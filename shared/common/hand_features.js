// hand_features.js
//
// JS port of ../hand_features.py — Similarity Method v2.1.
// Keep these two files in sync.
//
// Per-hand: 21 verdict slots; TIPs combine 3 sub-checks with AND;
//   5 landmarks inherit from another landmark.
// Inter-hand: 6 direction-sensitive vector checks between matched
//   landmark pairs (wrists + 5 fingertip pairs).
// Per-frame score: weighted 45/45/10 when ref has both hands;
//   100·(matched/21) when ref has only one hand.

export const FEATURES = {
  0:  ["palm_orientation"],
  1:  ["inherit", 2],
  2:  ["flexion", 0, 1, 2],
  3:  ["flexion", 1, 2, 3],
  4:  ["tip_combo",
        ["flexion",  2, 3, 4],
        ["distance", "mcp_tip", 2, 4],
        ["distance", "adj_tip", 4, 8]],
  5:  ["distance", "adj_mcp", 5, 9],
  6:  ["flexion", 0, 5, 6],
  7:  ["inherit", 8],
  8:  ["tip_combo",
        ["flexion",  5, 6, 8],
        ["distance", "mcp_tip", 5, 8],
        ["distance", "adj_tip", 8, 12]],
  9:  ["distance", "adj_mcp", 9, 13],
  10: ["flexion", 0, 9, 10],
  11: ["inherit", 12],
  12: ["tip_combo",
        ["flexion",  9, 10, 12],
        ["distance", "mcp_tip", 9, 12],
        ["distance", "adj_tip", 12, 16]],
  13: ["distance", "adj_mcp", 13, 17],
  14: ["flexion", 0, 13, 14],
  15: ["inherit", 16],
  16: ["tip_combo",
        ["flexion",  13, 14, 16],
        ["distance", "mcp_tip", 13, 16],
        ["distance", "adj_tip", 16, 20]],
  17: ["distance", "adj_mcp", 13, 17],
  18: ["flexion", 0, 17, 18],
  19: ["inherit", 20],
  20: ["tip_combo",
        ["flexion",  17, 18, 20],
        ["distance", "mcp_tip", 17, 20],
        ["distance", "adj_tip", 16, 20]],
};

export const DEFAULT_TOLERANCES = {
  palm_orientation: 25.0,
  flexion:          20.0,
  mcp_tip:          0.15,
  adj_tip:          0.20,
  adj_mcp:          0.08,
};

// Inter-hand pairs: [leftLM, rightLM, toleranceKind].
export const INTER_HAND_PAIRS = [
  [0,  0,  "wrist"],
  [4,  4,  "tip"],
  [8,  8,  "tip"],
  [12, 12, "tip"],
  [16, 16, "tip"],
  [20, 20, "tip"],
];

export const DEFAULT_INTER_HAND_TOLERANCES = {
  wrist: 0.35,
  tip:   0.30,
};

export const SCORE_WEIGHTS = {
  left_hand:  45.0,
  right_hand: 45.0,
  inter_hand: 10.0,
};
export const HAND_SLOTS  = 21;
export const INTER_SLOTS = INTER_HAND_PAIRS.length;

const EPS = 1e-9;
const RAD2DEG = 180.0 / Math.PI;

function vsub(a, b) { return { x: a.x - b.x, y: a.y - b.y }; }
function vdot(a, b) { return a.x * b.x + a.y * b.y; }
function vnorm(a)   { return Math.hypot(a.x, a.y); }
function dist(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }

function angleBetween(v1, v2) {
  const n1 = vnorm(v1), n2 = vnorm(v2);
  if (n1 < EPS || n2 < EPS) return 0.0;
  let c = vdot(v1, v2) / (n1 * n2);
  if (c < -1) c = -1; else if (c > 1) c = 1;
  return Math.acos(c) * RAD2DEG;
}

function signedAngleFromDown(v) {
  return Math.atan2(v.x, v.y) * RAD2DEG;
}

function wrapSigned(d) {
  return ((d + 180) % 360 + 360) % 360 - 180;
}

export function handDetected(handXY) {
  if (!handXY || handXY.length !== 21) return false;
  for (const p of handXY) if (p.x !== 0 || p.y !== 0) return true;
  return false;
}

function palmLength(handXY) {
  return Math.max(dist(handXY[0], handXY[9]), EPS);
}

function evalAtomic(spec, handXY, palmLen) {
  const kind = spec[0];
  if (kind === "palm_orientation") {
    return ["palm_orientation", signedAngleFromDown(vsub(handXY[9], handXY[0]))];
  }
  if (kind === "flexion") {
    const p = spec[1], j = spec[2], c = spec[3];
    return ["flexion", angleBetween(vsub(handXY[p], handXY[j]),
                                    vsub(handXY[c], handXY[j]))];
  }
  if (kind === "distance") {
    const subKind = spec[1], a = spec[2], b = spec[3];
    return [subKind, dist(handXY[a], handXY[b]) / palmLen];
  }
  throw new Error(`Unexpected atomic feature kind: ${kind}`);
}

export function computeHandFeatures(handXY) {
  if (!handDetected(handXY)) return null;
  const palmLen = palmLength(handXY);
  const feats = new Array(21);
  for (const k of Object.keys(FEATURES)) {
    const lm = parseInt(k, 10);
    const spec = FEATURES[lm];
    const kind = spec[0];
    if (kind === "inherit") {
      feats[lm] = ["inherit", spec[1]];
    } else if (kind === "tip_combo") {
      const subs = [];
      for (let i = 1; i < spec.length; i++) {
        subs.push(evalAtomic(spec[i], handXY, palmLen));
      }
      feats[lm] = ["tip_combo", subs];
    } else {
      feats[lm] = evalAtomic(spec, handXY, palmLen);
    }
  }
  return feats;
}

function atomicPass(refVal, testVal, tol) {
  const kind = refVal[0];
  let d = testVal[1] - refVal[1];
  if (kind === "palm_orientation") d = wrapSigned(d);
  return Math.abs(d) <= tol[kind];
}

export function compareHandFeatures(refFeats, testFeats, tolerances = null) {
  if (refFeats === null || testFeats === null) return null;
  const tol = Object.assign({}, DEFAULT_TOLERANCES, tolerances || {});
  const correct = new Array(21).fill(false);

  for (let lm = 0; lm < 21; lm++) {
    const ref = refFeats[lm], test = testFeats[lm];
    const kind = ref[0];
    if (kind === "inherit") continue;
    if (kind === "tip_combo") {
      let allPass = true;
      for (let i = 0; i < ref[1].length; i++) {
        if (!atomicPass(ref[1][i], test[1][i], tol)) { allPass = false; break; }
      }
      correct[lm] = allPass;
    } else {
      correct[lm] = atomicPass(ref, test, tol);
    }
  }

  for (let lm = 0; lm < 21; lm++) {
    if (refFeats[lm][0] === "inherit") {
      correct[lm] = correct[refFeats[lm][1]];
    }
  }

  return correct;
}

// ==================================================
// Inter-hand: 6 direction-sensitive vector checks
// ==================================================
export function computeInterHandVectors(leftXY, rightXY) {
  if (!handDetected(leftXY) || !handDetected(rightXY)) return null;
  const palmL = dist(leftXY[0], leftXY[9]);
  const palmR = dist(rightXY[0], rightXY[9]);
  const meanPalm = 0.5 * (palmL + palmR);
  if (meanPalm < EPS) return null;
  const out = new Array(INTER_HAND_PAIRS.length);
  for (let i = 0; i < INTER_HAND_PAIRS.length; i++) {
    const [li, ri, kind] = INTER_HAND_PAIRS[i];
    out[i] = [kind, {
      x: (rightXY[ri].x - leftXY[li].x) / meanPalm,
      y: (rightXY[ri].y - leftXY[li].y) / meanPalm,
    }];
  }
  return out;
}

export function compareInterHandVectors(refVecs, testVecs, tolerances = null) {
  if (refVecs === null || testVecs === null) return null;
  const tol = Object.assign({}, DEFAULT_INTER_HAND_TOLERANCES, tolerances || {});
  const n = INTER_HAND_PAIRS.length;
  const correct = new Array(n).fill(false);
  for (let i = 0; i < n; i++) {
    const [kindR, vr] = refVecs[i];
    const [, vt]      = testVecs[i];
    const dx = vt.x - vr.x;
    const dy = vt.y - vr.y;
    correct[i] = Math.hypot(dx, dy) <= tol[kindR];
  }
  return correct;
}

// ==================================================
// Per-frame weighted score
// ==================================================
function sumBools(arr) {
  let n = 0;
  for (const b of arr) if (b) n++;
  return n;
}

export function scoreFrame({ leftCorrect, rightCorrect, interCorrect,
                             refHasLeft, refHasRight }) {
  if (refHasLeft && refHasRight) {
    let score = 0;
    if (leftCorrect)  score += SCORE_WEIGHTS.left_hand  * (sumBools(leftCorrect)  / HAND_SLOTS);
    if (rightCorrect) score += SCORE_WEIGHTS.right_hand * (sumBools(rightCorrect) / HAND_SLOTS);
    if (interCorrect) {
      const n = interCorrect.length > 0 ? interCorrect.length : INTER_SLOTS;
      score += SCORE_WEIGHTS.inter_hand * (sumBools(interCorrect) / n);
    }
    return score;
  }
  if (refHasLeft) {
    if (!leftCorrect) return 0;
    return 100 * sumBools(leftCorrect) / HAND_SLOTS;
  }
  if (refHasRight) {
    if (!rightCorrect) return 0;
    return 100 * sumBools(rightCorrect) / HAND_SLOTS;
  }
  return null;
}
