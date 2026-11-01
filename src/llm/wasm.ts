// Which wllama build this webview can run, decided by compiling, not by
// asking (AGENTS.md §2, docs/platform-facts.md → wllama).
//
// wllama picks its default build (Memory64 + JSPI) when `WebAssembly.Suspending`
// exists and `new WebAssembly.Memory({ address: 'i64', initial: 1n })` doesn't
// throw. WebKit in macOS 27.0.1 (and so, presumably, iOS 27) has JSPI and
// accepts that descriptor, but still can't compile a Memory64 module ("Memory64
// is not enabled"). wllama then loads the default build, its abort reaches only
// console.log, and the load never settles: the 0.1.7 warm-up that timed out on a
// real iPhone and in WKWebView. Up to wllama 3.8.1 the check is the same.

import { Wllama } from '@wllama/wllama';

/** `(module (memory i64 1))`: a Memory64 memory section and nothing else. */
const MEMORY64_MODULE = new Uint8Array([0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x05, 0x03, 0x01, 0x04, 0x01]);

type Wasm = Pick<typeof WebAssembly, 'validate'> & { Suspending?: unknown };

/** Whether the default (Memory64 + JSPI) build can run here: both features, measured. */
export function defaultBuildRuns(wasm: Wasm | null = typeof WebAssembly === 'undefined' ? null : WebAssembly): boolean {
  if (!wasm) return false;
  try {
    return !!wasm.Suspending && wasm.validate(MEMORY64_MODULE);
  } catch {
    return false;
  }
}

type Resources = ReturnType<Wllama['getWorkerResources']>;

/**
 * wllama, but on the compat build whenever the default one can't compile here.
 * `setCompat` must be given the compat assets, as before; this only corrects
 * wllama's choice between them.
 */
export class AndaiWllama extends Wllama {
  private compatAssets: { worker: string; wasm: string } | null = null;

  override setCompat(compat: Parameters<Wllama['setCompat']>[0], mode?: Parameters<Wllama['setCompat']>[1]): void {
    super.setCompat(compat, mode);
    this.compatAssets = compat && typeof compat === 'object' && typeof compat.worker === 'string' ? { worker: compat.worker, wasm: compat.wasm } : null;
  }

  override getWorkerResources(): Resources {
    const r = super.getWorkerResources();
    return forceCompat(r, this.compatAssets, defaultBuildRuns());
  }
}

/** PURE: the worker resources to use, given wllama's choice and what was measured. */
export function forceCompat(r: Resources, compat: { worker: string; wasm: string } | null, defaultRuns: boolean): Resources {
  if (r.compat || defaultRuns || !compat) return r;
  return { ...r, wasmPath: new URL(compat.wasm, globalThis.location?.href).href, jsPath: compat.worker, compat: true };
}
