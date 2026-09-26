// The dev server port for the e2e and eval harnesses. `tauri dev` loads
// http://localhost:<port>, and "localhost" can reach a server on 127.0.0.1 or
// [::1]: with a developer's own `tauri dev` holding 1420 on one of them, a
// harness on the same port sometimes loaded that app (no harness in it) and
// waited until its timeout (seen 2026-09-26). Harness runs therefore use their
// own port, and with it their own webview storage (models download once more).
export const HARNESS_PORT = 1431;

/** Env and `tauri dev` args that serve and load the UI on `port`. */
export function devOnPort(port = HARNESS_PORT) {
  return {
    env: { ANDAI_DEV_PORT: String(port) },
    args: ['--config', JSON.stringify({ build: { devUrl: `http://localhost:${port}` } })],
  };
}
