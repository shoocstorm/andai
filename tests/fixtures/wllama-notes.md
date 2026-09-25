# wllama deployment notes

## Cross-origin isolation

wllama runs llama.cpp as WebAssembly. Multi-threaded inference needs
`SharedArrayBuffer`, which browsers only expose to cross-origin isolated pages.
Serve the app shell with both headers:

- `Cross-Origin-Opener-Policy: same-origin`
- `Cross-Origin-Embedder-Policy: require-corp`

Without them wllama silently falls back to a single thread.

## Model cache

Models are cached in the Origin Private File System (OPFS), keyed by URL,
so each GGUF downloads once.
