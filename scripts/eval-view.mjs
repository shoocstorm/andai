#!/usr/bin/env bun
// Serves the agent eval viewer (scripts/eval-viewer.html) with the reports in
// eval/, so its dropdowns list every run without picking files by hand (a
// file:// page can't list a folder).
//
//   bun run eval:view              # http://127.0.0.1:4178, opens the browser
//   EVAL_VIEW_PORT=5000 bun run eval:view
//
// Loopback only, read-only, and it serves nothing but the viewer and
// eval/*.json by plain file name.
import { spawn } from 'node:child_process';
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { join, resolve } from 'node:path';

const root = resolve(import.meta.dirname, '..');
const dir = join(root, 'eval');
const viewer = join(root, 'scripts', 'eval-viewer.html');
// `+` joins setups in report names (`item15-v6-1.7b-mlx+en.json`); no path separators.
const REPORT = /^[\w.+-]+\.json$/;

const json = (body, status = 200) => new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } });

function handle(req) {
  const { pathname } = new URL(req.url);
  if (pathname === '/') return new Response(readFileSync(viewer), { headers: { 'content-type': 'text/html; charset=utf-8' } });
  if (pathname === '/reports') {
    const files = existsSync(dir) ? readdirSync(dir).filter((f) => REPORT.test(f)) : [];
    return json(files.map((name) => ({ name, mtime: statSync(join(dir, name)).mtimeMs })).sort((a, b) => a.mtime - b.mtime));
  }
  const name = decodeURIComponent(pathname.replace(/^\/reports\//, ''));
  if (pathname.startsWith('/reports/') && REPORT.test(name) && existsSync(join(dir, name))) {
    return new Response(readFileSync(join(dir, name)), { headers: { 'content-type': 'application/json' } });
  }
  return json({ error: 'not found' }, 404);
}

const preferred = Number(process.env.EVAL_VIEW_PORT ?? 4178);
let server;
try {
  server = Bun.serve({ hostname: '127.0.0.1', port: preferred, fetch: handle });
} catch {
  server = Bun.serve({ hostname: '127.0.0.1', port: 0, fetch: handle }); // taken: any free port
}
const url = `http://127.0.0.1:${server.port}/`;
console.log(`[eval:view] ${url}  (reports from ${dir}; Ctrl+C to stop)`);
const opener = process.platform === 'darwin' ? ['open', [url]] : process.platform === 'win32' ? ['cmd', ['/c', 'start', '', url]] : ['xdg-open', [url]];
if (!process.env.EVAL_VIEW_NO_OPEN) spawn(opener[0], opener[1], { stdio: 'ignore', detached: true }).on('error', () => {});
