// PURE: how a wllama model is loaded on this device, and what a failed load
// tells the user. Phones get a smaller memory footprint from the start:
// iOS kills a Safari tab that grows past its memory limit, and the page
// reloads with no error to catch (AGENTS.md §2, docs/platform-facts.md →
// Web deployment). Qwen3 0.6B at its full 4,096-token context needs about
// 1.1 GB of WebAssembly memory (604 MB weights + 448 MB KV cache + 30 MB
// compute, measured in iOS Safari on the simulator); on the CPU with 2,048
// tokens it needs about 860 MB.

export type Device = {
  /** A phone or tablet browser: tighter memory, no window to resize. */
  mobile: boolean;
  ios: boolean;
};

type Nav = Pick<Navigator, 'userAgent'> & { maxTouchPoints?: number };

export function detectDevice(nav: Nav | undefined): Device {
  if (!nav) return { mobile: false, ios: false };
  const ua = nav.userAgent;
  // iPadOS reports itself as a Mac; only its touch screen tells it apart.
  const ios = /iPhone|iPad|iPod/.test(ua) || (/Macintosh/.test(ua) && (nav.maxTouchPoints ?? 0) > 1);
  return { mobile: ios || /Android|Mobile/.test(ua), ios };
}

/**
 * How much a load asks for, from most to least. Level 0 is the catalog's
 * context with layers on WebGPU when there is one; 1 is CPU only (wllama
 * then doesn't start WebGPU at all) at 2,048 tokens; 2 is CPU at 1,024.
 */
export type LoadPlan = { level: number; n_ctx: number; gpu: boolean };
export const MAX_LEVEL = 2;

export function loadPlan(nCtx: number, level: number): LoadPlan {
  if (level <= 0) return { level: 0, n_ctx: nCtx, gpu: true };
  if (level === 1) return { level: 1, n_ctx: Math.min(nCtx, 2048), gpu: false };
  return { level: MAX_LEVEL, n_ctx: Math.min(nCtx, 1024), gpu: false };
}

/** Where a load starts: phones at level 1, or lower than any level that failed here before. */
export function startLevel(device: Device, failedBefore: number | undefined): number {
  const base = device.mobile ? 1 : 0;
  return Math.min(MAX_LEVEL, Math.max(base, failedBefore ?? 0));
}

/** A one-line summary of a plan, for the progress line and the error. */
export function planText(p: LoadPlan): string {
  return `${p.gpu ? 'GPU' : 'CPU only'}, ${p.n_ctx.toLocaleString('en-US')}-token context`;
}

const MEMORY = /out of memory|\bOOM\b|bad_alloc|failed to allocate|could not allocate|unable to allocate|Cannot allocate|memory access out of bounds|RangeError.*(memory|buffer|allocation)|Maximum call stack|Aborted\(/i;
const STORAGE = /QuotaExceeded|quota|NoModificationAllowed|not enough space|storage/i;
const NETWORK = /Failed to fetch|NetworkError|network|Load failed|ERR_|HTTP (4|5)\d\d|status (4|5)\d\d/i;
const GPU = /webgpu|GPUDevice|adapter|device lost/i;

export type LoadFailure = {
  /** What the user reads. */
  message: string;
  /** Whether a lighter plan could help (memory or GPU trouble). */
  retryLighter: boolean;
};

/** A failed load, said plainly. `raw` is the thrown error's text. */
export function describeLoadError(raw: string, name: string, plan: LoadPlan | null, device: Device): LoadFailure {
  const where = device.ios ? 'this iPhone' : device.mobile ? 'this phone' : 'this computer';
  if (/integrity check/.test(raw)) return { message: raw, retryLighter: false };
  if (/timed out warming up/.test(raw)) {
    return {
      message: `${name} didn't finish warming up in 3 minutes on ${where}, so loading was stopped. Close other tabs and try again, or pick a smaller model.`,
      retryLighter: true,
    };
  }
  if (MEMORY.test(raw)) {
    return {
      message: `${name} needs more memory than the browser gives this page on ${where}${plan ? ` (tried ${planText(plan)})` : ''}. Close other tabs and apps and try again, or pick a smaller model.`,
      retryLighter: true,
    };
  }
  if (GPU.test(raw) && plan?.gpu) {
    return { message: `${name} couldn't start on the GPU (${raw.slice(0, 160)}).`, retryLighter: true };
  }
  if (STORAGE.test(raw)) {
    return {
      message: `There isn't enough storage space in this browser to keep ${name}. Free up space on ${where}, or remove another downloaded model in Settings, then try again.`,
      retryLighter: false,
    };
  }
  if (NETWORK.test(raw)) {
    return { message: `${name} couldn't be downloaded: the connection failed (${raw.slice(0, 120)}). Check the network and try again; the download resumes.`, retryLighter: false };
  }
  return { message: `${name} couldn't be loaded: ${raw.slice(0, 300)}`, retryLighter: true };
}

/** The message after the page itself was closed mid-load (a crash leaves no error to catch). */
export function crashMessage(name: string, plan: LoadPlan, next: LoadPlan | null, device: Device): string {
  const what = device.ios ? 'iOS closed the page' : 'The page was closed';
  const cause = `${what} while ${name} was warming up (${planText(plan)}), most likely because it ran out of memory.`;
  return next
    ? `${cause} The next try uses less memory (${planText(next)}). Close other tabs and apps first.`
    : `${cause} It already used the smallest setting Andai has, so this device can't run ${name} in the browser. Try a smaller model.`;
}
