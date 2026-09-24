// dtw.js
//
// Classical DTW over skeleton sequences. Port of the curated joint set +
// frame distance metric used by dtw_compare_skeleton.py (fastdtw in
// Python, exact DP here — sequences are short enough that O(T·N) time
// and memory are fine).

// Absolute landmark indices in the 75-array. Matches the joint-set
// selected in dtw_compare_skeleton.py.
const WRIST_JOINTS = [15, 16];
const HAND_JOINTS = [
  // Left hand (33..53) — 15 joints from thumb_mcp/index_pip onward.
  35, 36, 37,
  39, 40, 41,
  43, 44, 45,
  47, 48, 49,
  51, 52, 53,
  // Right hand (54..74) — 15 joints, same layout.
  56, 57, 58,
  60, 61, 62,
  64, 65, 66,
  68, 69, 70,
  72, 73, 74,
];
export const SELECTED_JOINTS = [...WRIST_JOINTS, ...HAND_JOINTS].sort((a, b) => a - b);

// Wider joint set used by test_dtw_lasso_similarity.py / balloon_game:
// both arms {shoulders 11,12; elbows 13,14; wrists 15,16} plus both FULL
// hands {33..74}, not just the curated 15-per-hand subset above.
const ARM_JOINTS = [11, 12, 13, 14, 15, 16];
const FULL_HAND_JOINTS = Array.from({ length: 42 }, (_, i) => 33 + i);  // 33..74
export const FULL_DTW_JOINTS = [...new Set([...ARM_JOINTS, ...FULL_HAND_JOINTS])].sort((a, b) => a - b);

// Distance between two frames, restricted to `jointSet` (default
// SELECTED_JOINTS): mean of per-joint Euclidean distances, skipping any
// joint that's undetected (all-zero) in either frame -- matches
// frame_distance() in test_dtw_lasso_similarity.py. 1.0 is a moderate
// placeholder cost for the rare case where no joint in the set is valid
// in either frame.
export function frameDistance(f1, f2, jointSet = SELECTED_JOINTS) {
  let sum = 0, n = 0;
  for (const j of jointSet) {
    const a = f1[j], b = f2[j];
    if ((a.x === 0 && a.y === 0) || (b.x === 0 && b.y === 0)) continue;
    sum += Math.hypot(a.x - b.x, a.y - b.y);
    n++;
  }
  return n ? sum / n : 1.0;
}

// Classical DTW. `ref`/`test` are frame arrays; `jointSet` (default
// SELECTED_JOINTS) picks which frameDistance() call site above to use.
// Returns:
//   { distance, path: [[t_ref, t_test], …] }
// path is monotonically non-decreasing in both indices.
export function runDTW(ref, test, jointSet = SELECTED_JOINTS) {
  const T = ref.length, N = test.length;
  if (T === 0 || N === 0) return { distance: 0, path: [] };

  // Cost matrix (row = ref index, col = test index). Flat Float64Array.
  const INF = Infinity;
  const cost = new Float64Array((T + 1) * (N + 1)).fill(INF);
  cost[0] = 0;

  // Frame-distance cache to avoid recomputing on backtrack.
  const stride = N;
  const local = new Float64Array(T * N);
  for (let i = 0; i < T; i++) {
    for (let k = 0; k < N; k++) {
      local[i * stride + k] = frameDistance(ref[i], test[k], jointSet);
    }
  }

  for (let i = 1; i <= T; i++) {
    for (let k = 1; k <= N; k++) {
      const d = local[(i - 1) * stride + (k - 1)];
      const a = cost[(i - 1) * (N + 1) + k];       // insertion (advance ref)
      const b = cost[i * (N + 1) + (k - 1)];       // deletion (advance test)
      const c = cost[(i - 1) * (N + 1) + (k - 1)]; // match
      cost[i * (N + 1) + k] = d + Math.min(a, b, c);
    }
  }

  // Backtrace from (T, N) → (1, 1).
  const path = [];
  let i = T, k = N;
  while (i > 0 && k > 0) {
    path.push([i - 1, k - 1]);
    const a = cost[(i - 1) * (N + 1) + k];
    const b = cost[i * (N + 1) + (k - 1)];
    const c = cost[(i - 1) * (N + 1) + (k - 1)];
    if (c <= a && c <= b) { i--; k--; }
    else if (a <= b)      { i--; }
    else                  { k--; }
  }
  path.reverse();

  return { distance: cost[T * (N + 1) + N], path };
}
