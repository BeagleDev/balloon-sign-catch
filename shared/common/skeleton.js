// skeleton.js
//
// Port of landmarks.py: extract a 75-landmark xy sequence per frame
// (33 pose + 21 left hand + 21 right hand), shoulder-center + hip-scale
// normalize, and EMA-smooth over time. Used by Word task to build the
// two sequences fed to DTW.

export const POSE_LANDMARK_COUNT = 33;
export const HAND_LANDMARK_COUNT = 21;
export const TOTAL_LANDMARKS     = 75;

// Absolute landmark indices in the 75-array.
export const LEFT_HAND_START  = POSE_LANDMARK_COUNT;               // 33
export const RIGHT_HAND_START = POSE_LANDMARK_COUNT + HAND_LANDMARK_COUNT; // 54

// Pose landmark indices used for normalization.
const L_SHOULDER = 11;
const R_SHOULDER = 12;
const L_HIP      = 23;
const R_HIP      = 24;

const ZERO = () => ({ x: 0, y: 0 });

// Result adapter for MediaPipe HandLandmarker.detectForVideo / .detect —
// returns {left: [{x,y},…]|null, right: […]|null}.
export function splitHandsByHandedness(result) {
  const out = { left: null, right: null };
  if (!result || !result.landmarks) return out;
  for (let i = 0; i < result.landmarks.length; i++) {
    const label = result.handednesses[i][0].categoryName;
    const xy = result.landmarks[i].map(p => ({ x: p.x, y: p.y }));
    if (label === "Left") out.left = xy;
    else if (label === "Right") out.right = xy;
  }
  return out;
}

// Result adapter for MediaPipe PoseLandmarker — returns a 33-length array
// of {x,y} or null when nothing detected.
export function extractPoseXY(result) {
  if (!result || !result.landmarks || result.landmarks.length === 0) return null;
  return result.landmarks[0].map(p => ({ x: p.x, y: p.y }));
}

// Combine pose + hand results into the flat 75-landmark image-space array.
// Missing landmarks become (0, 0), matching landmarks.py:extract_combined_xy_image.
export function combineXYImage(poseXY, handSplit) {
  const out = new Array(TOTAL_LANDMARKS);

  if (poseXY) {
    for (let i = 0; i < POSE_LANDMARK_COUNT; i++) out[i] = { x: poseXY[i].x, y: poseXY[i].y };
  } else {
    for (let i = 0; i < POSE_LANDMARK_COUNT; i++) out[i] = ZERO();
  }

  const left  = handSplit && handSplit.left  ? handSplit.left  : null;
  const right = handSplit && handSplit.right ? handSplit.right : null;
  for (let i = 0; i < HAND_LANDMARK_COUNT; i++) {
    out[LEFT_HAND_START  + i] = left  ? { x: left[i].x,  y: left[i].y  } : ZERO();
    out[RIGHT_HAND_START + i] = right ? { x: right[i].x, y: right[i].y } : ZERO();
  }

  return out;
}

// Shoulder-centered, hip-scaled normalization — mirror of
// landmarks.py:normalize_skeleton_xy. Returns a NEW 75-length array.
// If either shoulder or hip is missing (0,0), returns a zero skeleton.
const EPS_SCALE = 1e-6;
export function normalizeSkeletonXY(xyImage) {
  const ls = xyImage[L_SHOULDER];
  const rs = xyImage[R_SHOULDER];
  const lh = xyImage[L_HIP];
  const rh = xyImage[R_HIP];

  const shoulderMissing = (ls.x === 0 && ls.y === 0) || (rs.x === 0 && rs.y === 0);
  const hipMissing      = (lh.x === 0 && lh.y === 0) || (rh.x === 0 && rh.y === 0);
  if (shoulderMissing || hipMissing) {
    return Array.from({ length: TOTAL_LANDMARKS }, ZERO);
  }

  const root = { x: 0.5 * (ls.x + rs.x), y: 0.5 * (ls.y + rs.y) };
  const hipC = { x: 0.5 * (lh.x + rh.x), y: 0.5 * (lh.y + rh.y) };
  let scale  = Math.hypot(root.x - hipC.x, root.y - hipC.y);
  if (scale < EPS_SCALE) scale = EPS_SCALE;

  const out = new Array(TOTAL_LANDMARKS);
  for (let i = 0; i < TOTAL_LANDMARKS; i++) {
    const p = xyImage[i];
    if (p.x === 0 && p.y === 0) {
      out[i] = ZERO();
    } else {
      out[i] = { x: (p.x - root.x) / scale, y: (p.y - root.y) / scale };
    }
  }
  return out;
}

// EMA smoother matching landmarks.py:TemporalSmoother — per-landmark
// blend with the previous smoothed skeleton. `alpha` weights the current
// frame; higher = less smoothing (default 0.7).
export class TemporalSmoother {
  constructor(alpha = 0.7) {
    this.alpha = alpha;
    this.prev  = null;
  }

  reset() { this.prev = null; }

  apply(current) {
    if (this.prev === null) {
      this.prev = current.map(p => ({ x: p.x, y: p.y }));
      return current.map(p => ({ x: p.x, y: p.y }));
    }
    const a = this.alpha;
    const b = 1 - a;
    const out = new Array(current.length);
    for (let i = 0; i < current.length; i++) {
      const c = current[i];
      const p = this.prev[i];
      out[i] = { x: a * c.x + b * p.x, y: a * c.y + b * p.y };
    }
    this.prev = out.map(p => ({ x: p.x, y: p.y }));
    return out;
  }
}

// Mirror a 75-landmark xy array horizontally in image-normalized space.
// Zero landmarks (undetected) stay zero. Handedness LABELS are NOT
// swapped — a "Left" hand's landmarks just get their x-coord flipped in
// place; anatomical meaning is preserved.
export function flipXYImage(xyImage) {
  return xyImage.map(p => (p.x === 0 && p.y === 0) ? { x: 0, y: 0 } : { x: 1 - p.x, y: p.y });
}

// Convenience: given raw pose + hand results for a single frame, produce
// the tuple (xyImage, xySkel). Callers own the smoother lifecycle.
// Pass `{ flipX: true }` to mirror the x-coordinates before normalizing;
// use this when the caller displays a mirrored (selfie) view so stored
// landmarks live in the same coordinate space as what's shown.
export function frameSkeleton(poseResult, handResult, smoother, opts = {}) {
  const poseXY   = extractPoseXY(poseResult);
  const handSplit = splitHandsByHandedness(handResult);
  let xyImage    = combineXYImage(poseXY, handSplit);
  if (opts.flipX) xyImage = flipXYImage(xyImage);
  const xySkel   = normalizeSkeletonXY(xyImage);
  const smoothed = smoother ? smoother.apply(xySkel) : xySkel;
  return { xyImage, xySkel: smoothed };
}

// Detected checks (any nonzero point → present).
export function isLandmarkDetected(pt) {
  return pt && (pt.x !== 0 || pt.y !== 0);
}
