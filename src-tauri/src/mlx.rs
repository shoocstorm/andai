//! The one thread that owns MLX and every model loaded on it: the Laya
//! decision checkpoint (laya/) and the native chat models (llm/). Commands
//! send it jobs and wait off the async runtime (AGENTS.md §4). Keeping every
//! MLX call on one thread keeps its default stream and Metal state in one
//! place; the agent never decides while an answer streams, so a job queued
//! behind a generation waits at most for that generation.
//!
//! MLX exists only for Apple Silicon (`cfg(mlx)`, set by build.rs). Elsewhere
//! this is an empty state and the commands that need it say so.

#[cfg(mlx)]
use std::sync::mpsc::{channel, Sender};
#[cfg(mlx)]
use std::sync::{Arc, Mutex};
#[cfg(mlx)]
use tauri::{AppHandle, Manager};

/// What is loaded where, by checkpoint id.
#[cfg(mlx)]
#[derive(Debug, Default, Clone)]
pub struct Loaded {
    pub laya: Option<String>,
    /// Per native LLM slot (`llm::Slot`): chat, decider.
    pub llm: [Option<String>; 2],
}

/// The models the MLX thread holds. Only jobs touch it, on that thread.
#[cfg(mlx)]
#[derive(Default)]
pub struct Models {
    pub laya: Option<(String, crate::laya::engine::Engine)>,
    pub llm: [Option<(String, crate::llm::engine::Engine)>; 2],
}

#[cfg(mlx)]
impl Models {
    fn loaded(&self) -> Loaded {
        Loaded { laya: self.laya.as_ref().map(|(id, _)| id.clone()), llm: [0, 1].map(|i| self.llm[i].as_ref().map(|(id, _)| id.clone())) }
    }
}

#[cfg(mlx)]
type Job = Box<dyn FnOnce(&mut Models) + Send>;

#[cfg(mlx)]
pub struct Thread {
    tx: Sender<Job>,
    loaded: Arc<Mutex<Loaded>>,
}

#[cfg(mlx)]
impl Thread {
    /// `metallib`: the bundled `mlx.metallib` (release builds). Dev and test
    /// builds leave it unset and MLX finds the one its build produced.
    pub fn spawn(metallib: Option<std::path::PathBuf>) -> Self {
        let (tx, rx) = channel::<Job>();
        std::thread::Builder::new()
            .name("mlx".into())
            .spawn(move || {
                if let Some(path) = metallib.filter(|p| p.is_file()) {
                    match mlx_rs::metal::set_metallib_path(path.to_string_lossy()) {
                        Ok(()) => println!("[mlx] metallib {}", path.display()),
                        Err(e) => eprintln!("[mlx] couldn't use {}: {e}", path.display()),
                    }
                }
                let mut models = Models::default();
                for job in rx {
                    job(&mut models);
                }
            })
            .expect("spawn the MLX thread");
        Self { tx, loaded: Arc::default() }
    }

    /// Runs `f` on the MLX thread and waits for its result. What's loaded is
    /// updated before the result is sent, so a caller that sees the result
    /// also sees the new `loaded()`.
    pub fn run<T: Send + 'static>(&self, f: impl FnOnce(&mut Models) -> Result<T, String> + Send + 'static) -> Result<T, String> {
        let (reply, rx) = channel();
        let loaded = self.loaded.clone();
        let job: Job = Box::new(move |models| {
            let res = f(models);
            *loaded.lock().unwrap() = models.loaded();
            let _ = reply.send(res);
        });
        self.tx.send(job).map_err(|_| "the MLX thread has stopped".to_string())?;
        rx.recv().map_err(|_| "the MLX thread has stopped".to_string())?
    }

    pub fn loaded(&self) -> Loaded {
        self.loaded.lock().unwrap().clone()
    }
}

/// App state: the MLX thread, started on first use.
#[derive(Default)]
pub struct Mlx {
    #[cfg(mlx)]
    thread: Mutex<Option<Arc<Thread>>>,
}

#[cfg(mlx)]
impl Mlx {
    pub fn thread(&self, app: &AppHandle) -> Arc<Thread> {
        let mut t = self.thread.lock().unwrap();
        t.get_or_insert_with(|| {
            // Release bundles ship mlx.metallib as a resource (tauri.laya.conf.json).
            let metallib = app.path().resource_dir().ok().map(|d| d.join("mlx.metallib"));
            Arc::new(Thread::spawn(metallib))
        })
        .clone()
    }

    /// Without starting the thread: nothing is loaded before it runs.
    pub fn loaded(&self) -> Loaded {
        self.thread.lock().unwrap().as_ref().map(|t| t.loaded()).unwrap_or_default()
    }
}
