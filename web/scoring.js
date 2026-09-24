// scoring.js
//
// Per-attempt scoring: DTW-align a captured live sequence against a
// word's precomputed reference sequence, then combine hand + wrist scores
// exactly like test_dtw_lasso_similarity.py's main loop (0.6*hand +
// 0.4*wrist per pair, renormalized if one side is missing that pair),
// and report the "pairs above PAIR_PASS_THRESHOLD%" percentage used to
// decide correct/not.
//
// PAIR_PASS_THRESHOLD was lowered from 75 to 50: with the deployment
// hand-rules bundle (thresholds tuned on dataset_hand_shapes' curated,
// single-frame hand-shape photos, not live-webcam-vs-precomputed-clip
// pairs) and a 0.6*hand + 0.4*wrist blend, hitting 75 on both dimensions
// simultaneously most frames was demanding enough that pairSimilarityPct
// came out ~0 even on reasonable attempts -- see CLAUDE.md's Balloon Game
// section.

import { compareHands } from "../shared/common/hand_features_v3.js";
import { runDTW, FULL_DTW_JOINTS } from "../shared/common/dtw.js";
import { scoreWristPair } from "../shared/common/wrist_score.js";

const HAND_LANDMARK_COUNT = 21;
const TOTAL_WEIGHTS = { hand: 0.6, wrist: 0.4 };
export const PAIR_PASS_THRESHOLD = 50;
export const CORRECT_THRESHOLD_PCT = 50;

// -> { score: number|null, verdicts: {left,right} }. Same convention as
// score_pair() / game/score.js's v3 branch: a hand the *reference* has is
// required; missing on the test side counts as a full fail for that
// hand's 21 slots. null iff the reference has neither hand.
function scoreHandPair(refHands, testHands, rules, agg) {
  let total = 0, passed = 0;
  const verdicts = { left: null, right: null };
  for (const side of ["left", "right"]) {
    const refH = refHands[side];
    if (!refH) continue;
    total += HAND_LANDMARK_COUNT;
    const testH = testHands[side];
    if (testH) {
      const res = compareHands(refH, testH, rules, agg);
      verdicts[side] = res.verdicts;
      for (const v of res.verdicts) if (v) passed++;
    } else {
      verdicts[side] = new Array(HAND_LANDMARK_COUNT).fill(false);
    }
  }
  if (total === 0) return { score: null, verdicts };
  return { score: 100 * passed / total, verdicts };
}

// refSeq/liveSeq: arrays of { hands: {left,right}, xySkel } (same shape
// build_sequence() produces in the Python script).
// -> { avg, pairSimilarityPct, pairs: [{ri,ti,total,hand,wrist}, …] }
// avg/pairSimilarityPct are null if no pair in the DTW path was scoreable
// (reference has no hand at all -- shouldn't happen for real clips, but
// mirrors the Python script's "no scoreable pairs" case).
export function scoreAttempt(refSeq, liveSeq, rules, agg = "majority") {
  const { path } = runDTW(
    refSeq.map(f => f.xySkel),
    liveSeq.map(f => f.xySkel),
    FULL_DTW_JOINTS,
  );

  const pairs = [];
  for (const [ri, ti] of path) {
    const ref = refSeq[ri], live = liveSeq[ti];
    const handRes = scoreHandPair(ref.hands, live.hands, rules, agg);
    if (handRes.score === null) continue;  // nothing to score this pair

    const refHasHand = { left: !!ref.hands.left, right: !!ref.hands.right };
    const wristRes = scoreWristPair(ref.xySkel, live.xySkel, refHasHand);

    let total;
    if (wristRes.score === null) {
      total = handRes.score;
    } else {
      const parts = [
        [TOTAL_WEIGHTS.hand, handRes.score],
        [TOTAL_WEIGHTS.wrist, wristRes.score],
      ];
      const wSum = parts.reduce((s, [w]) => s + w, 0);
      total = parts.reduce((s, [w, v]) => s + w * v, 0) / wSum;
    }

    pairs.push({ ri, ti, total, hand: handRes.score, wrist: wristRes.score, handVerdicts: handRes.verdicts });
  }

  if (pairs.length === 0) return { avg: null, pairSimilarityPct: null, pairs };

  const avg = pairs.reduce((s, p) => s + p.total, 0) / pairs.length;
  const abovePassThreshold = pairs.filter(p => p.total >= PAIR_PASS_THRESHOLD).length;
  const pairSimilarityPct = 100 * abovePassThreshold / pairs.length;
  return { avg, pairSimilarityPct, pairs };
}

export function isCorrect(pairSimilarityPct) {
  return pairSimilarityPct !== null && pairSimilarityPct >= CORRECT_THRESHOLD_PCT;
}
