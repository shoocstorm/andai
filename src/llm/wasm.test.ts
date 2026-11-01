import { describe, expect, it, vi } from 'vitest';

vi.mock('@wllama/wllama', () => ({
  Wllama: class {
    compat: unknown = null;
    setCompat(c: unknown) {
      this.compat = c;
    }
    getWorkerResources() {
      // what wllama 3.6–3.8 returns in WebKit 27: it thinks the default build runs
      return { wasmPath: 'http://localhost/wllama/default/wllama.wasm', compat: false };
    }
  },
}));

const { AndaiWllama, defaultBuildRuns, forceCompat } = await import('./wasm');

describe('defaultBuildRuns', () => {
  it('needs JSPI and a Memory64 module that validates, measured, not assumed', () => {
    // WebKit in macOS 27.0.1, probed with JavaScriptCore: Suspending exists, Memory64 doesn't compile
    expect(defaultBuildRuns({ Suspending: function () {}, validate: () => false })).toBe(false);
    expect(defaultBuildRuns({ validate: () => true })).toBe(false);
    expect(defaultBuildRuns({ Suspending: function () {}, validate: () => true })).toBe(true);
    expect(
      defaultBuildRuns({
        Suspending: 1,
        validate: () => {
          throw new Error('no');
        },
      }),
    ).toBe(false);
    expect(defaultBuildRuns(null)).toBe(false);
  });

  it('hands validate a well-formed Memory64 module', () => {
    let bytes: BufferSource | undefined;
    defaultBuildRuns({ Suspending: 1, validate: (b) => ((bytes = b), true) });
    // (module (memory i64 1)): memory section, flags 0x04 = 64-bit index
    expect([...(bytes as Uint8Array)]).toEqual([0, 0x61, 0x73, 0x6d, 1, 0, 0, 0, 5, 3, 1, 4, 1]);
    // and the same module with a 32-bit memory is valid wasm everywhere
    expect(WebAssembly.validate(new Uint8Array([0, 0x61, 0x73, 0x6d, 1, 0, 0, 0, 5, 3, 1, 0, 1]))).toBe(true);
  });
});

describe('forceCompat', () => {
  const compat = { worker: 'http://localhost/wllama/compat/wllama.js', wasm: '/wllama/compat/wllama.wasm' };
  const def = { wasmPath: 'http://localhost/wllama/default/wllama.wasm', compat: false };

  it('moves to the compat build when the default one cannot run here', () => {
    expect(forceCompat(def, compat, false)).toEqual({ wasmPath: 'http://localhost:3000/wllama/compat/wllama.wasm', jsPath: compat.worker, compat: true });
  });

  it("keeps wllama's choice when it already chose compat, the default runs, or no compat assets were given", () => {
    const already = { wasmPath: 'x', jsPath: 'y', compat: true };
    expect(forceCompat(already, compat, false)).toBe(already);
    expect(forceCompat(def, compat, true)).toBe(def);
    expect(forceCompat(def, null, false)).toBe(def);
  });

  it('is what AndaiWllama hands its worker, with the assets given to setCompat', () => {
    const w = new AndaiWllama({ default: '/d.wasm' }, {});
    w.setCompat(compat);
    // Node (Vitest) has JSPI and Memory64; make it the WebKit 27 that hung 0.1.7
    const validate = vi.spyOn(WebAssembly, 'validate').mockReturnValue(false);
    vi.stubGlobal('WebAssembly', Object.assign(Object.create(WebAssembly), { Suspending: function () {}, validate }));
    expect(w.getWorkerResources()).toMatchObject({ compat: true, jsPath: compat.worker });
    validate.mockReturnValue(true);
    expect(w.getWorkerResources()).toMatchObject({ compat: false, wasmPath: def.wasmPath });
    vi.unstubAllGlobals();
    validate.mockRestore();
  });
});
