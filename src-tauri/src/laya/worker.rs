//! One thread owns MLX and the loaded checkpoint; commands send it jobs and
//! wait off the async runtime (AGENTS.md §4). Keeping every MLX call on one
//! thread also keeps its default stream and Metal state in one place.

use super::engine::{Asked, Engine, Question};
use std::path::PathBuf;
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex};
use std::time::Instant;

type Reply<T> = Sender<Result<T, String>>;

enum Job {
    Load { id: String, dir: PathBuf, reply: Reply<f64> },
    Unload { reply: Reply<()> },
    Ask { state: String, questions: Vec<Question>, reply: Reply<Asked> },
}

pub struct Worker {
    tx: Sender<Job>,
    loaded: Arc<Mutex<Option<String>>>,
}

impl Worker {
    /// `metallib`: the bundled `mlx.metallib` (release builds). Dev and test
    /// builds leave it unset and MLX finds the one its build produced.
    pub fn spawn(metallib: Option<PathBuf>) -> Self {
        let (tx, rx) = channel::<Job>();
        let loaded = Arc::new(Mutex::new(None));
        let shared = loaded.clone();
        std::thread::Builder::new()
            .name("laya".into())
            .spawn(move || {
                if let Some(path) = metallib.filter(|p| p.is_file()) {
                    match mlx_rs::metal::set_metallib_path(path.to_string_lossy()) {
                        Ok(()) => println!("[laya] metallib {}", path.display()),
                        Err(e) => eprintln!("[laya] couldn't use {}: {e}", path.display()),
                    }
                }
                let mut engine: Option<Engine> = None;
                for job in rx {
                    match job {
                        Job::Load { id, dir, reply } => {
                            let started = Instant::now();
                            engine = None;
                            *shared.lock().unwrap() = None;
                            let res = Engine::load(&dir).map(|e| {
                                engine = Some(e);
                                *shared.lock().unwrap() = Some(id);
                                started.elapsed().as_secs_f64() * 1e3
                            });
                            let _ = reply.send(res);
                        }
                        Job::Unload { reply } => {
                            engine = None;
                            *shared.lock().unwrap() = None;
                            let _ = reply.send(Ok(()));
                        }
                        Job::Ask { state, questions, reply } => {
                            let res = match &engine {
                                Some(e) => e.ask(&state, &questions),
                                None => Err("No Laya model is loaded.".into()),
                            };
                            let _ = reply.send(res);
                        }
                    }
                }
            })
            .expect("spawn the laya thread");
        Self { tx, loaded }
    }

    fn ask<T>(&self, job: impl FnOnce(Reply<T>) -> Job) -> Result<T, String> {
        let (reply, rx) = channel();
        self.tx.send(job(reply)).map_err(|_| "the Laya thread has stopped".to_string())?;
        rx.recv().map_err(|_| "the Laya thread has stopped".to_string())?
    }

    /// Loads the checkpoint in `dir`, replacing any loaded one. Returns load ms.
    pub fn load(&self, id: &str, dir: PathBuf) -> Result<f64, String> {
        self.ask(|reply| Job::Load { id: id.to_string(), dir, reply })
    }

    pub fn unload(&self) -> Result<(), String> {
        self.ask(|reply| Job::Unload { reply })
    }

    pub fn questions(&self, state: String, questions: Vec<Question>) -> Result<Asked, String> {
        self.ask(|reply| Job::Ask { state, questions, reply })
    }

    pub fn loaded(&self) -> Option<String> {
        self.loaded.lock().unwrap().clone()
    }
}
