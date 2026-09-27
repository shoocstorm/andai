#!/usr/bin/env bun
// The app version lives in four manifests that must always agree
// (bun.lock doesn't record the root package's version):
//
//   package.json               "version"
//   src-tauri/tauri.conf.json  "version"   ← what the built .app reports
//   src-tauri/Cargo.toml       [package] version
//   src-tauri/Cargo.lock       the "andai" package entry
//
//   bun scripts/version.mjs            print the version (fails if they disagree)
//   bun scripts/version.mjs check      same, quiet on success
//   bun scripts/version.mjs check 1.2.3 | v1.2.3   also require that exact version (CI: tag == version)
//   bun scripts/version.mjs set 1.2.3  write it to every manifest, then re-check
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';

const root = resolve(import.meta.dirname, '..');
const path = (p) => resolve(root, p);
const SEMVER = /^\d+\.\d+\.\d+$/;

const json = {
  read: (p) => JSON.parse(readFileSync(path(p), 'utf8')),
  write: (p, v) => writeFileSync(path(p), `${JSON.stringify(v, null, 2)}\n`),
};

// [package] version is the first `version = "…"` line in Cargo.toml
const cargoTomlRe = /^(version\s*=\s*")([^"]+)(")/m;
// the [[package]] block whose name is "andai" in Cargo.lock. \r?\n: Git for
// Windows checks text files out with CRLF (autocrlf), so the manifests read
// differently there (AGENTS.md §2).
const cargoLockRe = /(\[\[package\]\]\r?\nname = "andai"\r?\nversion = ")([^"]+)(")/;

export function readVersions() {
  const toml = readFileSync(path('src-tauri/Cargo.toml'), 'utf8').match(cargoTomlRe);
  const cargoLock = readFileSync(path('src-tauri/Cargo.lock'), 'utf8').match(cargoLockRe);
  return {
    'package.json': json.read('package.json').version,
    'src-tauri/tauri.conf.json': json.read('src-tauri/tauri.conf.json').version,
    'src-tauri/Cargo.toml': toml?.[2],
    'src-tauri/Cargo.lock (andai)': cargoLock?.[2],
  };
}

function setVersion(v) {
  const pkg = json.read('package.json');
  pkg.version = v;
  json.write('package.json', pkg);

  const conf = json.read('src-tauri/tauri.conf.json');
  conf.version = v;
  json.write('src-tauri/tauri.conf.json', conf);

  for (const [file, re] of [
    ['src-tauri/Cargo.toml', cargoTomlRe],
    ['src-tauri/Cargo.lock', cargoLockRe],
  ]) {
    const src = readFileSync(path(file), 'utf8');
    if (!re.test(src)) fail(`could not find the version in ${file}`);
    writeFileSync(path(file), src.replace(re, `$1${v}$3`));
  }
}

function check(expected) {
  const versions = readVersions();
  const distinct = new Set(Object.values(versions));
  const problems = [];
  if (distinct.size !== 1 || distinct.has(undefined)) problems.push('manifests disagree');
  const [version] = distinct;
  if (expected && version !== expected) problems.push(`expected ${expected}, manifests say ${version}`);
  if (version && !SEMVER.test(version)) problems.push(`${version} is not X.Y.Z`);
  if (problems.length) {
    console.error(`version check failed: ${problems.join('; ')}`);
    for (const [file, v] of Object.entries(versions)) console.error(`  ${file.padEnd(34)} ${v ?? '(missing)'}`);
    process.exit(1);
  }
  return version;
}

function fail(msg) {
  console.error(`error: ${msg}`);
  process.exit(1);
}

const normalize = (v) => (v ? v.replace(/^v/, '') : v);

if (import.meta.url === `file://${process.argv[1]}`) {
  const [cmd = 'print', arg] = process.argv.slice(2);
  if (cmd === 'print') console.log(check());
  else if (cmd === 'check') check(normalize(arg));
  else if (cmd === 'set') {
    const v = normalize(arg);
    if (!v || !SEMVER.test(v)) fail(`'${arg ?? ''}' is not a valid X.Y.Z version`);
    setVersion(v);
    check(v);
    console.log(`version set to ${v} in all manifests`);
  } else fail(`unknown command '${cmd}' (print | check [version] | set <version>)`);
}
