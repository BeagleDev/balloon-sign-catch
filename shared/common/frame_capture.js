// frame_capture.js
//
// Helpers for the Word task's frame pipeline:
//   1. `captureJpegBlob` — turn a canvas into a small JPEG blob for
//      later result-video playback (~60KB per 640×480 frame at q=0.75).
//   2. `decodeVideoFrames` — walk an uploaded video frame-by-frame via
//      requestVideoFrameCallback, invoking a callback with the current
//      frame drawn on a canvas so the caller can run detectors on it.

const DEFAULT_JPEG_QUALITY = 0.75;

// Snapshot a canvas as a JPEG blob (Promise). Small enough that keeping
// a few hundred in memory is fine (~10MB for 200 frames at ~50KB each).
export function captureJpegBlob(canvas, quality = DEFAULT_JPEG_QUALITY) {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      blob => { blob ? resolve(blob) : reject(new Error("toBlob returned null")); },
      "image/jpeg",
      quality,
    );
  });
}

// Load an image file into an HTMLImageElement via object URL.
export function loadImageFromFile(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload  = () => resolve({ img, url });
    img.onerror = e  => { URL.revokeObjectURL(url); reject(e); };
    img.src = url;
  });
}

// Load a video file into a hidden HTMLVideoElement. Caller owns cleanup
// via `revoke()` after use.
export async function loadVideoFromFile(file) {
  const url   = URL.createObjectURL(file);
  const video = document.createElement("video");
  video.src      = url;
  video.muted    = true;
  video.playsInline = true;
  video.crossOrigin  = "anonymous";
  await new Promise((resolve, reject) => {
    video.onloadedmetadata = resolve;
    video.onerror = e => reject(new Error(`Video failed to load: ${e?.message || e}`));
  });
  return {
    video,
    revoke: () => URL.revokeObjectURL(url),
  };
}

// Iterate every frame of a loaded video by pausing and manually seeking.
// This is slower than requestVideoFrameCallback-with-playback, but gives
// the caller a deterministic sample: exactly the same `currentTime`
// values on every run of the same file. Same video in → same landmark
// sequence out.
//
// For each sampled frame:
//   1. Seek to `t` and wait for `seeked`.
//   2. Draw the current frame onto `canvas` (sized to video dimensions,
//      or downscaled if `maxWidth` is smaller than the source).
//   3. Await the caller's `onFrame({ canvas, ctx, mediaTime, frameIdx })`.
// Progress is reported via `onProgress(percent 0..100)` if provided.
//
// `fps` sets the sample rate (default 30). For most sign-language clips
// 20–30 fps is plenty; DTW handles the rate mismatch anyway.
// `maxWidth` (optional) caps the canvas width; if the source is wider,
// the frame is downscaled (aspect-ratio preserved) before detection +
// JPEG capture. Cuts JPEG sizes and downstream memory dramatically for
// big source videos, with negligible impact on landmark accuracy since
// MediaPipe resizes to ~256px internally anyway.
export async function decodeVideoFrames({
  video,
  canvas,
  onFrame,
  onProgress,
  fps = 30,
  maxWidth = null,
}) {
  const srcW = video.videoWidth;
  const srcH = video.videoHeight;
  let w = srcW, h = srcH;
  if (maxWidth && srcW > maxWidth) {
    w = maxWidth;
    h = Math.round(srcH * maxWidth / srcW);
  }
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width  = w;
    canvas.height = h;
  }
  const ctx = canvas.getContext("2d");
  const duration = (isFinite(video.duration) && video.duration > 0) ? video.duration : null;
  const step = 1 / fps;

  video.playbackRate = 1;
  video.muted        = true;
  video.pause();
  await seekTo(video, 0);

  let frameIdx = 0;
  for (let t = 0; duration === null || t < duration + step * 0.5; t += step) {
    if (Math.abs(video.currentTime - t) > 1e-4) {
      await seekTo(video, Math.min(t, duration === null ? t : duration));
    }
    ctx.drawImage(video, 0, 0, w, h);
    await onFrame({ canvas, ctx, mediaTime: t, frameIdx });
    frameIdx++;
    if (onProgress && duration) {
      onProgress(Math.min(100, (t / duration) * 100));
    }
    if (duration === null) break;
  }
  if (onProgress) onProgress(100);
  return frameIdx;
}

function seekTo(video, t) {
  return new Promise((resolve, reject) => {
    const onSeeked = () => { cleanup(); resolve(); };
    const onError  = () => { cleanup(); reject(new Error("seek error")); };
    const cleanup  = () => {
      video.removeEventListener("seeked", onSeeked);
      video.removeEventListener("error",  onError);
    };
    video.addEventListener("seeked", onSeeked, { once: true });
    video.addEventListener("error",  onError,  { once: true });
    video.currentTime = t;
  });
}
