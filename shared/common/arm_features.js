// arm_features.js
//
// Word task arm-shape features: two elbow angles + four distances
// (wrist ↔ same-side shoulder, wrist ↔ nose), computed in the same
// shoulder-centered / hip-scaled skeleton space as skeleton.js.
//
// Six verdict slots per aligned frame:
//   L elbow angle   (11-13-15)
//   R elbow angle   (12-14-16)
//   L wrist ↔ L shoulder distance  (15-11)
//   R wrist ↔ R shoulder distance  (16-12)
//   L wrist ↔ nose distance         (15-0)
//   R wrist ↔ nose distance         (16-0)
//
// Verdicts are computed on the normalized skeleton (xySkel from
// skeleton.js) so distances are already in torso-scaled units.

import { isLandmarkDetected } from "./skeleton.js";

const EPS = 1e-9;
const RAD2DEG = 180.0 / Math.PI;

export const ARM_TOLERANCES = {
  elbow_angle: 20.0,   // degrees
  distance:    0.15,   // shoulder-hip torso units
};

// Slot ordering — used to index the correctness array and to map slot
// verdicts back to pose landmarks for visualization.
export const ARM_SLOTS = [
  { name: "left_elbow_angle",     kind: "elbow_angle", a: 11, j: 13, c: 15, involves: [11, 13, 15] },
  { name: "right_elbow_angle",    kind: "elbow_angle", a: 12, j: 14, c: 16, involves: [12, 14, 16] },
  { name: "left_wrist_shoulder",  kind: "distance",    p: 15, q: 11,        involves: [15, 11]     },
  { name: "right_wrist_shoulder", kind: "distance",    p: 16, q: 12,        involves: [16, 12]     },
  { name: "left_wrist_nose",      kind: "distance",    p: 15, q:  0,        involves: [15,  0]     },
  { name: "right_wrist_nose",     kind: "distance",    p: 16, q:  0,        involves: [16,  0]     },
];

function angleAt(a, j, c) {
  const v1x = a.x - j.x, v1y = a.y - j.y;
  const v2x = c.x - j.x, v2y = c.y - j.y;
  const n1 = Math.hypot(v1x, v1y);
  const n2 = Math.hypot(v2x, v2y);
  if (n1 < EPS || n2 < EPS) return null;
  let cos = (v1x * v2x + v1y * v2y) / (n1 * n2);
  if (cos < -1) cos = -1; else if (cos > 1) cos = 1;
  return Math.acos(cos) * RAD2DEG;
}

function distance(p, q) {
  return Math.hypot(p.x - q.x, p.y - q.y);
}

// Compute the six arm features for a single skeleton frame. Returns an
// array of {kind, value|null}. `null` value → the input landmarks
// required for that slot were missing (undetected).
export function computeArmFeatures(xySkel) {
  if (!xySkel) return null;
  const out = [];
  for (const slot of ARM_SLOTS) {
    if (slot.kind === "elbow_angle") {
      const A = xySkel[slot.a], J = xySkel[slot.j], C = xySkel[slot.c];
      if (!isLandmarkDetected(A) || !isLandmarkDetected(J) || !isLandmarkDetected(C)) {
        out.push({ kind: slot.kind, value: null });
      } else {
        out.push({ kind: slot.kind, value: angleAt(A, J, C) });
      }
    } else {
      // distance
      const P = xySkel[slot.p], Q = xySkel[slot.q];
      if (!isLandmarkDetected(P) || !isLandmarkDetected(Q)) {
        out.push({ kind: slot.kind, value: null });
      } else {
        out.push({ kind: slot.kind, value: distance(P, Q) });
      }
    }
  }
  return out;
}

// Compare two feature arrays produced by computeArmFeatures. Returns a
// parallel array of {true, false, null} — null means the slot is not
// scoreable (either side missing).
export function compareArmFeatures(refFeats, testFeats, tolerances = null) {
  if (!refFeats || !testFeats) return null;
  const tol = Object.assign({}, ARM_TOLERANCES, tolerances || {});
  const out = new Array(ARM_SLOTS.length);
  for (let i = 0; i < ARM_SLOTS.length; i++) {
    const r = refFeats[i], t = testFeats[i];
    if (r.value === null || t.value === null) { out[i] = null; continue; }
    const d = Math.abs(t.value - r.value);
    out[i] = d <= tol[r.kind];
  }
  return out;
}

// Build a per-pose-landmark correctness map from an arm-verdict array.
// Semantics: a pose landmark is `true` iff every arm slot involving it
// passed (AND), `false` if any involving slot failed, `null` if no
// scoreable slot involves it.
export function armVerdictsToPoseCorrectness(armVerdicts) {
  const perLM = new Array(33).fill(undefined);
  if (!armVerdicts) return perLM.map(_ => null);

  for (let i = 0; i < ARM_SLOTS.length; i++) {
    const v = armVerdicts[i];
    if (v === null) continue;
    for (const lm of ARM_SLOTS[i].involves) {
      if (v === false) {
        perLM[lm] = false;
      } else if (perLM[lm] === undefined) {
        perLM[lm] = true;
      }
    }
  }
  return perLM.map(v => (v === undefined ? null : v));
}
