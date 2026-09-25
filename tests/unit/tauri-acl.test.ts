// Guards the Tauri command contract (see AGENTS.md → Tauri commands). Release
// builds load the UI from http://localhost, a "remote" origin, so a command
// that is registered but not declared in build.rs + granted in the capability
// silently fails ONLY in the shipped app. This keeps all four lists in sync.
import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';

const ROOT = join(__dirname, '../..');
const read = (p: string) => readFileSync(join(ROOT, p), 'utf8');
const rustFiles = readdirSync(join(ROOT, 'src-tauri/src')).map((f) => read(`src-tauri/src/${f}`));

const defined = new Set(
  rustFiles.flatMap((src) => [...src.matchAll(/#\[tauri::command\]\s*pub(?:\(crate\))?\s*(?:async\s+)?fn\s+(\w+)|#\[tauri::command\]\s*(?:async\s+)?fn\s+(\w+)/g)].map((m) => m[1] ?? m[2])),
);
const handlerBlock = read('src-tauri/src/lib.rs').match(/generate_handler!\[([\s\S]*?)\]/)![1];
const registered = new Set([...handlerBlock.matchAll(/(?:\w+::)?(\w+),?/g)].map((m) => m[1]));
const commandsArray = read('src-tauri/build.rs').match(/const COMMANDS[^=]*=\s*&\[([\s\S]*?)\];/)![1];
const declared = new Set([...commandsArray.matchAll(/"(\w+)"/g)].map((m) => m[1]));
const permitted = new Set(
  (JSON.parse(read('src-tauri/capabilities/default.json')).permissions as string[])
    .filter((p) => p.startsWith('allow-'))
    .map((p) => p.slice(6).replace(/-/g, '_')),
);
const invoked = new Set(
  readdirSync(join(ROOT, 'src'), { recursive: true })
    .map(String)
    .filter((f) => /\.tsx?$/.test(f) && !f.includes('.test.'))
    .flatMap((f) => [...read(`src/${f}`).matchAll(/(?:invoke|call)(?:<[^>]*>)?\(\s*'(\w+)'/g)].map((m) => m[1])),
);

const sorted = (s: Set<string>) => [...s].sort();

describe('tauri command ACL', () => {
  it('finds the commands', () => {
    expect(defined.size).toBeGreaterThanOrEqual(8);
  });
  it('every #[tauri::command] is registered in generate_handler!', () => {
    expect(sorted(registered)).toEqual(sorted(defined));
  });
  it('every command is declared in build.rs (AppManifest)', () => {
    expect(sorted(declared)).toEqual(sorted(defined));
  });
  it('every command is granted in capabilities/default.json', () => {
    expect(sorted(permitted)).toEqual(sorted(defined));
  });
  it('every command the frontend invokes exists', () => {
    expect([...invoked].filter((c) => !defined.has(c))).toEqual([]);
  });
  it('the capability allows the release UI origin', () => {
    const cap = JSON.parse(read('src-tauri/capabilities/default.json'));
    const port = read('src-tauri/src/lib.rs').match(/UI_PORT: u16 = (\d+)/)![1];
    expect(cap.remote.urls).toContain(`http://localhost:${port}/*`);
  });
});
