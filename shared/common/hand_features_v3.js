// hand_features_v3.js
//
// Browser mirror of hand_features_v3.py. Consumes a JSON rule bundle
// exported by LASSO_regression/export_rules.py (handcraft_v2_1,
// lasso_selected, or lasso_all) and produces a per-landmark verdict
// array with the same shape/spec as compareHandFeatures() —
// drop-in for the existing drawHand(ctx, xy, correctness, …).
//
// Landmarks: array of 21 objects {x, y} in image-normalized coords.
//
// Aggregators (per landmark):
//   "majority"           pass if ≥ 50% of rules on that landmark pass  (default)
//   "weighted_majority"  pass if Σ|coef|·pass ≥ Σ|coef|·fail
//   "and"                pass only if every rule on that landmark passes
//   "or"                 pass if any rule on that landmark passes

export const EPS  = 1e-9;
export const N_LM = 21;

// ------------------------------------------------------------------
// Geometric primitives
// ------------------------------------------------------------------
function _dist(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }

function _angle3pt(a, b, c) {
  const v1x = a.x - b.x, v1y = a.y - b.y;
  const v2x = c.x - b.x, v2y = c.y - b.y;
  const n1 = Math.hypot(v1x, v1y);
  const n2 = Math.hypot(v2x, v2y);
  if (n1 < EPS || n2 < EPS) return 0;
  let cos = (v1x * v2x + v1y * v2y) / (n1 * n2);
  if (cos > 1) cos = 1; else if (cos < -1) cos = -1;
  return Math.acos(cos) * 180 / Math.PI;
}

function _signedAngleFromDown(vx, vy) {
  return Math.atan2(vx, vy) * 180 / Math.PI;
}

function _wrapSigned(d) {
  return ((d + 180) % 360 + 360) % 360 - 180;
}

function _boneOrientationLocal(a, b, refA, refB) {
  const vx = b.x - a.x, vy = b.y - a.y;
  const rx = refB.x - refA.x, ry = refB.y - refA.y;
  const nV = Math.hypot(vx, vy);
  const nR = Math.hypot(rx, ry);
  if (nV < EPS || nR < EPS) return 0;
  let local = Math.atan2(vy, vx) - Math.atan2(ry, rx);
  const TWO_PI = 2 * Math.PI;
  local = ((local + Math.PI) % TWO_PI + TWO_PI) % TWO_PI - Math.PI;
  return local * 180 / Math.PI;
}

// ------------------------------------------------------------------
// Kind dispatch
// ------------------------------------------------------------------
export function computeFeature(kind, params, lm) {
  switch (kind) {
    case "signed_angle_from_down": {
      const a = lm[params.a], b = lm[params.b];
      return _signedAngleFromDown(b.x - a.x, b.y - a.y);
    }
    case "angle_3pt":
      return _angle3pt(lm[params.a], lm[params.j], lm[params.c]);
    case "distance":
      return _dist(lm[params.a], lm[params.b]);
    case "distance_over_palm": {
      const palm = _dist(lm[0], lm[9]);
      return palm > EPS ? _dist(lm[params.a], lm[params.b]) / palm : 0;
    }
    case "distance_over_pair": {
      const denom = _dist(lm[params.p], lm[params.q]);
      return denom > EPS ? _dist(lm[params.a], lm[params.b]) / denom : 0;
    }
    case "bone_orientation_local":
      return _boneOrientationLocal(
        lm[params.a], lm[params.b],
        lm[params.ref_a], lm[params.ref_b],
      );
    case "extension_ratio": {
      let total = 0;
      for (const [p, q] of params.bones) total += _dist(lm[p], lm[q]);
      return total > EPS ? _dist(lm[params.mcp], lm[params.tip]) / total : 0;
    }
    default:
      throw new Error(`unknown feature kind: ${kind}`);
  }
}

// ------------------------------------------------------------------
// Rule bundle loader
// ------------------------------------------------------------------
export async function loadRules(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`load rules ${url}: ${res.status}`);
  const bundle = await res.json();
  if (!bundle.features) throw new Error(`${url}: not a rule bundle`);
  return bundle;
}

// ------------------------------------------------------------------
// Aggregation
// ------------------------------------------------------------------
function _aggregate(passes, coefs, agg) {
  if (passes.length === 0) return true;
  switch (agg) {
    case "majority": {
      let n = 0; for (const v of passes) if (v) n++;
      return n * 2 >= passes.length;
    }
    case "weighted_majority": {
      let pos = 0, neg = 0;
      for (let i = 0; i < passes.length; i++) {
        (passes[i] ? (pos += coefs[i]) : (neg += coefs[i]));
      }
      return pos >= neg;
    }
    case "and": return passes.every(Boolean);
    case "or":  return passes.some(Boolean);
    default: throw new Error(`unknown agg: ${agg}`);
  }
}

// ------------------------------------------------------------------
// Compare two hands using a rule bundle
// ------------------------------------------------------------------
export function compareHands(refLm, testLm, rules, agg = "majority") {
  if (!refLm || refLm.length < N_LM || !testLm || testLm.length < N_LM) {
    throw new Error("compareHands: both hands need 21 landmarks");
  }

  const perRule = new Array(rules.features.length);
  for (let i = 0; i < rules.features.length; i++) {
    const rule = rules.features[i];
    const va = computeFeature(rule.kind, rule.params, refLm);
    const vb = computeFeature(rule.kind, rule.params, testLm);
    let diff = va - vb;
    if (rule.comparison === "signed_wrap") diff = _wrapSigned(diff);
    const d   = Math.abs(diff);
    const thr = Number(rule.threshold);
    const dir = rule.direction || "<=";
    const passed = dir === "<=" ? (d <= thr) : (d >= thr);
    perRule[i] = { name: rule.name, kind: rule.kind, d, threshold: thr, direction: dir, passed };
  }

  const perLmPass = Array.from({ length: N_LM }, () => []);
  const perLmCoef = Array.from({ length: N_LM }, () => []);
  for (let i = 0; i < rules.features.length; i++) {
    const rule = rules.features[i];
    const w    = Math.abs(Number(rule.coef ?? 1.0));
    for (const lmIdx of rule.involves) {
      if (lmIdx >= 0 && lmIdx < N_LM) {
        perLmPass[lmIdx].push(perRule[i].passed);
        perLmCoef[lmIdx].push(w);
      }
    }
  }

  const verdicts = new Array(N_LM).fill(true);
  for (let lm = 0; lm < N_LM; lm++) {
    verdicts[lm] = _aggregate(perLmPass[lm], perLmCoef[lm], agg);
  }

  let passes = 0;
  for (const r of perRule) if (r.passed) passes++;
  const passRatio = perRule.length ? passes / perRule.length : 1;

  return { verdicts, passRatio, perRule };
}
