// mediapipe_loader.js
//
// Load MediaPipe tasks-vision (vendored first, jsDelivr CDN fallback).
// Detectors are built LAZILY on first use — the page load itself only
// pays the ~250 ms bundle+wasm+fileset cost. Each task then awaits the
// ensure*() methods it needs before touching a detector.
//
// Detector build times (CPU delegate) are ~1 s each; a task that uses
// two detectors pays ~2 s the first time it kicks off, then instant on
// every subsequent action.

const MEDIAPIPE_VERSION = "0.10.21";

// All vendored assets live next to this module in web_interface/shared/
// (../mediapipe_js, ../models), so resolve them against import.meta.url —
// that keeps the loader independent of which app (signsim/, game/, …) or
// URL depth is doing the importing.
const VENDOR_BUNDLE = new URL("../mediapipe_js/tasks-vision/vision_bundle.mjs", import.meta.url).href;
const VENDOR_WASM   = new URL("../mediapipe_js/tasks-vision/wasm", import.meta.url).href;
const CDN_BUNDLE    = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/vision_bundle.mjs`;
const CDN_WASM      = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/wasm`;

const HAND_MODEL_PATH = new URL("../models/hand_landmarker.task", import.meta.url).href;
// Lite pose model (~5 MB, ~5× smaller than heavy). Accuracy is more than
// sufficient for arm/wrist tracking — swap to pose_landmarker.task if
// you need the heavy variant.
const POSE_MODEL_PATH = new URL("../models/pose_landmarker_lite.task", import.meta.url).href;

// Flip to "GPU" to trade slow (~28 s) startup for faster per-frame
// inference. CPU keeps startup at ~5 s total across all four detectors,
// and per-frame still fits under the 15 fps Word budget (67 ms).
const DELEGATE = "CPU";

let cachedPromise = null;

async function loadBundle() {
  let bundleUrl, wasmUrl, source;
  try {
    const head = await fetch(VENDOR_BUNDLE, { method: "HEAD" });
    if (head.ok) {
      bundleUrl = VENDOR_BUNDLE;   // already absolute (import.meta.url-resolved)
      wasmUrl   = VENDOR_WASM;
      source    = "vendored";
    }
  } catch (_) { /* fall through to CDN */ }

  if (!bundleUrl) {
    bundleUrl = CDN_BUNDLE;
    wasmUrl   = CDN_WASM;
    source    = "cdn";
  }

  const mod     = await import(bundleUrl);
  const fileset = await mod.FilesetResolver.forVisionTasks(wasmUrl);
  console.log(`[mediapipe] using ${source} bundle (${bundleUrl})`);
  return { mod, fileset, source };
}

async function timed(label, fn) {
  const t0 = performance.now();
  const out = await fn();
  const dt = performance.now() - t0;
  console.log(`[mediapipe] ${label}: ${dt.toFixed(0)} ms`);
  return out;
}

// Monotonic timestamp source for detectForVideo() calls. MediaPipe
// requires strictly increasing timestamps per detector; sharing a single
// counter across all detectors keeps callers simple and avoids
// per-detector bookkeeping when a page runs multiple sequences in a row.
function makeTsSource() {
  let last = 0;
  return () => {
    const now = Math.round(performance.now());
    last = Math.max(last + 1, now);
    return last;
  };
}

// Build a memoized "ensure" function for one detector slot.
// `kind` ∈ {"hand", "pose"}, `mode` ∈ {"IMAGE", "VIDEO"}.
// The first call kicks off the actual createFromOptions; concurrent
// calls await the same promise; on resolve, the built detector is also
// pinned onto `mp[key]` for downstream sync access
// (e.g. `mp.handVideo.detectForVideo(...)` inside a rAF loop).
function makeEnsurer(mp, kind, mode) {
  const key = kind + mode[0] + mode.slice(1).toLowerCase();  // handImage / handVideo / poseImage / poseVideo
  let pending = null;
  return function ensure() {
    if (mp[key])   return Promise.resolve(mp[key]);
    if (pending)   return pending;

    const modelPath = kind === "hand" ? HAND_MODEL_PATH : POSE_MODEL_PATH;
    const Cls       = kind === "hand" ? mp.mod.HandLandmarker : mp.mod.PoseLandmarker;
    const baseOptions = { modelAssetPath: modelPath, delegate: DELEGATE };
    const opts = kind === "hand"
      ? { baseOptions, runningMode: mode, numHands: 2 }
      : { baseOptions, runningMode: mode, numPoses: 1 };

    pending = timed(`build ${key}`, () => Cls.createFromOptions(mp.fileset, opts))
      .then(d => { mp[key] = d; return d; });
    return pending;
  };
}

export function loadMediaPipe() {
  if (cachedPromise) return cachedPromise;
  cachedPromise = (async () => {
    const t0     = performance.now();
    const bundle = await timed("load bundle+wasm+fileset", () => loadBundle());
    const mp = {
      ...bundle,
      nextTs:    makeTsSource(),
      handImage: null, handVideo: null,
      poseImage: null, poseVideo: null,
    };
    mp.ensureHandImage = makeEnsurer(mp, "hand", "IMAGE");
    mp.ensureHandVideo = makeEnsurer(mp, "hand", "VIDEO");
    mp.ensurePoseImage = makeEnsurer(mp, "pose", "IMAGE");
    mp.ensurePoseVideo = makeEnsurer(mp, "pose", "VIDEO");
    console.log(
      `[mediapipe] core init: ${(performance.now() - t0).toFixed(0)} ms ` +
      `(detectors build lazily on first use)`,
    );
    return mp;
  })();
  return cachedPromise;
}
