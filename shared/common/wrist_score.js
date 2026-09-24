// wrist_score.js
//
// Port of test_dtw_lasso_similarity.py's wrist<->nose scoring: per-side
// normalized wrist<->nose distance (on the shoulder-center + hip-scale
// normalized skeleton, skeleton.js:normalizeSkeletonXY), pure linear decay
// from 100 at zero difference to 0 at a fixed max difference, no flat
// tolerance plateau. A side only counts if the *reference* has a hand
// there (mirrors the hand score's own gate).

const NOSE = 0;
const L_WRIST = 15, R_WRIST = 16;
export const WRIST_DIST_MAX = 1.0;  // normalized (hip-scale) distance diff at which the score hits 0

function isZero(p) {
  return p.x === 0 && p.y === 0;
}

// Normalized wrist<->nose distance on xySkel (normalizeSkeletonXY output).
// null if the whole frame's skeleton is undetected -- normalizeSkeletonXY
// zeroes the entire array when shoulders/hips aren't detected, so a
// single all-zero check covers it (mirrors the Python version exactly).
export function wristNoseDistance(xySkel, wristIdx) {
  if (!xySkel || xySkel.every(isZero)) return null;
  const w = xySkel[wristIdx], n = xySkel[NOSE];
  return Math.hypot(w.x - n.x, w.y - n.y);
}

export function wristSideScore(diff) {
  return 100 * Math.max(0, 1 - diff / WRIST_DIST_MAX);
}

// refHands/testHands: { left: bool, right: bool } -- whether the
// reference/test has a hand on that side (gates which sides count, same
// as the hand score's own convention).
// -> { score: number|null, sideScores: {left,right}, verdicts: {left,right} }
// score is the average across required sides; null iff no side qualifies.
export function scoreWristPair(refXYSkel, testXYSkel, refHasHand) {
  let totalW = 0, sumScore = 0;
  const sideScores = { left: null, right: null };
  const verdicts   = { left: null, right: null };

  for (const [side, wristIdx] of [["left", L_WRIST], ["right", R_WRIST]]) {
    if (!refHasHand[side]) continue;  // no reference hand on this side -> don't score its wrist either
    const refDist = wristNoseDistance(refXYSkel, wristIdx);
    if (refDist === null) continue;
    totalW++;
    const testDist = wristNoseDistance(testXYSkel, wristIdx);
    const sideScore = testDist !== null
      ? wristSideScore(Math.abs(refDist - testDist))
      : 0;  // required but missing on the test side -> fail
    sumScore += sideScore;
    sideScores[side] = sideScore;
    verdicts[side] = sideScore >= 75;
  }

  if (totalW === 0) return { score: null, sideScores, verdicts };
  return { score: sumScore / totalW, sideScores, verdicts };
}
