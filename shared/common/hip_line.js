// hip_line.js
//
// Port of the hip-line geometry in test_dtw_lasso_similarity.py: a "hands
// must be raised above the hips to count" boundary, extended sideways and
// lifted up a bit from the raw hip landmarks, used to find the start/end
// of a signing attempt (at least one wrist crossing up, then neither
// wrist still up).
//
// Takes a 75-slot xyImage array ({x,y} objects, as built by
// skeleton.js:combineXYImage) — NOT the normalized xySkel, since this
// works in raw image-normalized coordinates like the Python version.

const L_SHOULDER = 11, R_SHOULDER = 12;
const L_HIP = 23, R_HIP = 24;
const L_WRIST = 15, R_WRIST = 16;

const EPS_SCALE = 1e-6;
const HIP_EXT_FACTOR  = 0.5;  // extend the line by this fraction of hip-width, each side
const HIP_LIFT_FACTOR = 0.3;  // shift the whole line up by this fraction of torso height

function isZero(p) {
  return p.x === 0 && p.y === 0;
}

// -> [p1, p2] | null. Endpoints extended by HIP_EXT_FACTOR times the hip
// length on each side, then shifted up by HIP_LIFT_FACTOR times the torso
// height (shoulder-center to hip-center distance). null if the hips
// weren't detected; the lift is simply skipped if the shoulders weren't.
export function extendedHipPoints(xyImage) {
  if (!xyImage) return null;
  const lh = xyImage[L_HIP], rh = xyImage[R_HIP];
  if (isZero(lh) || isZero(rh)) return null;

  const dx = rh.x - lh.x, dy = rh.y - lh.y;
  let p1 = { x: lh.x - HIP_EXT_FACTOR * dx, y: lh.y - HIP_EXT_FACTOR * dy };
  let p2 = { x: rh.x + HIP_EXT_FACTOR * dx, y: rh.y + HIP_EXT_FACTOR * dy };

  const ls = xyImage[L_SHOULDER], rs = xyImage[R_SHOULDER];
  if (!isZero(ls) && !isZero(rs)) {
    const shoulderC = { x: 0.5 * (ls.x + rs.x), y: 0.5 * (ls.y + rs.y) };
    const hipC = { x: 0.5 * (lh.x + rh.x), y: 0.5 * (lh.y + rh.y) };
    const torsoHeight = Math.hypot(shoulderC.x - hipC.x, shoulderC.y - hipC.y);
    const lift = HIP_LIFT_FACTOR * torsoHeight;
    p1 = { x: p1.x, y: p1.y - lift };
    p2 = { x: p2.x, y: p2.y - lift };
  }
  return [p1, p2];
}

// True if the wrist landmark (not the 21-point hand -- that can drop out
// even when the arm/wrist itself is clearly tracked) is above the hip
// line, the line's y interpolated/extrapolated to the wrist's x.
export function wristAboveHipLine(xyImage, wristIdx, hipLine) {
  if (!xyImage || !hipLine) return false;
  const w = xyImage[wristIdx];
  if (isZero(w)) return false;
  const [p1, p2] = hipLine;
  if (Math.abs(p2.x - p1.x) < EPS_SCALE) return false;  // degenerate line -> can't tell
  const slope = (p2.y - p1.y) / (p2.x - p1.x);
  return w.y < p1.y + slope * (w.x - p1.x);
}

export function anyWristAboveHipline(xyImage) {
  const hipLine = extendedHipPoints(xyImage);
  return wristAboveHipLine(xyImage, L_WRIST, hipLine) ||
         wristAboveHipLine(xyImage, R_WRIST, hipLine);
}

// -> [start, end], an inclusive frame-index range: start is the first
// frame at least one wrist rises above the hip line; end is the first
// frame after that where neither wrist is above it anymore (both hands
// down), so a hand that's still mid-sign doesn't get the window cut
// short just because the other hand dropped first (or the sequence's
// last frame, if a wrist never comes back down). null if no wrist is
// ever above it at all.
export function findActiveCrop(xyImages) {
  const anyFlags = xyImages.map(anyWristAboveHipline);
  const start = anyFlags.indexOf(true);
  if (start === -1) return null;
  let end = xyImages.length - 1;
  for (let i = start + 1; i < xyImages.length; i++) {
    if (!anyFlags[i]) { end = i; break; }
  }
  return [start, end];
}
