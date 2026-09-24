// balloon_game/app.js

import { loadMediaPipe } from "../shared/common/mediapipe_loader.js";
import {
  frameSkeleton, TemporalSmoother,
  LEFT_HAND_START, RIGHT_HAND_START, HAND_LANDMARK_COUNT,
} from "../shared/common/skeleton.js";
import {
  anyWristAboveHipline, extendedHipPoints,
} from "../shared/common/hip_line.js";
import { loadRules } from "../shared/common/hand_features_v3.js";
import { scoreAttempt, isCorrect } from "./scoring.js";

const LIVE_SAMPLE_MS  = 1000 / 3;  // live attempt buffer: same ~3 fps density
const FALL_MS         = 20000;  // a real attempt needs much longer than the mockup's 4.2s tap window
const BALLOON_APPEAR_MS = 500;  // held static (full balloon + word, not yet falling) before the drop starts
const HINT_REVEAL_FRACTION = 0.4;  // hint video (if toggled on) starts once the balloon has fallen this far down the frame
const TARGET_BALLOONS = 3;
const LIVES           = 3;
// advanceWord() only ends the round once caught reaches TARGET_BALLOONS or
// missed reaches LIVES -- a mixed run (e.g. miss, miss, catch) can use up
// TARGET_BALLOONS words without either counter getting there, so the word
// pool must cover the worst case: TARGET_BALLOONS-1 catches interleaved
// with LIVES-1 misses, plus the one deciding attempt.
const WORD_POOL_SIZE  = TARGET_BALLOONS + LIVES - 1;
const COLORS          = ["a", "b", "c", "d"];
const AGG              = "majority";

// ------------------------------------------------------------------
// DOM
// ------------------------------------------------------------------
const btnHint       = document.getElementById("btn-hint");
const btnRestart    = document.getElementById("btn-restart");
const hintPanel     = document.getElementById("hint-panel");
const hintVideo     = document.getElementById("hint-video");
const numCaught     = document.getElementById("num-caught");
const numLives      = document.getElementById("num-lives");
const pipsCaught    = document.getElementById("pips-caught");
const pipsLives     = document.getElementById("pips-lives");
const stageRow      = document.getElementById("stage-row");
const viewport      = document.getElementById("viewport");
const webcamEl      = document.getElementById("webcam");
const canvas        = document.getElementById("canvas");
const recBadge      = document.getElementById("rec-badge");
const hipHint       = document.getElementById("hip-hint");
const overlayLoading = document.getElementById("overlay-loading");
const loadingText   = document.getElementById("loading-text");
const overlayStart  = document.getElementById("overlay-start");
const btnStart      = document.getElementById("btn-start");
const overlayEnd    = document.getElementById("overlay-end");
const endTitle      = document.getElementById("end-title");
const endText       = document.getElementById("end-text");
const btnAgain      = document.getElementById("btn-again");
const debugPanel    = document.getElementById("debug-panel");

const queryParams = new URLSearchParams(location.search);

// ?debug=1 -- shows every DTW-aligned pair's hand/wrist/total score after
// each attempt (see finishAttempt()), instead of just pass/fail.
const DEBUG = queryParams.get("debug") === "1";
debugPanel.hidden = !DEBUG;

// ?rules=<source> -- same convention as game/ and signsim/ (see CLAUDE.md);
// defaults to the relaxed bundle (lasso_selected's thresholds x1.5) since
// the stricter default made pairSimilarityPct come out ~0 even on
// reasonable attempts -- override with e.g. ?rules=lasso_selected.
const KNOWN_RULES = new Set(["handcraft", "lasso_selected", "lasso_all", "lasso2"]);
const rulesParam = queryParams.get("rules");
const RULES_NAME = KNOWN_RULES.has(rulesParam) ? rulesParam : "lasso2";

// ------------------------------------------------------------------
// Pip icons -- plain emoji instead of the mockup's hand-drawn SVGs.
// Caught pips: unfilled U+26AA (white circle) -> filled U+1F388 (balloon).
// Life pips:   remaining U+1F4A3 (bomb) -> lost U+1F4A5 (collision/burst).
// ------------------------------------------------------------------
const PIP_CAUGHT_EMPTY = "⚪";
const PIP_CAUGHT_FULL  = "\u{1F388}";
const PIP_LIFE_OK      = "\u{1F4A3}";
const PIP_LIFE_LOST    = "\u{1F4A5}";

// ------------------------------------------------------------------
// State
// ------------------------------------------------------------------
const state = {
  phase: "loading",   // loading | start | playing | won | lost
  mp: null,
  rules: null,
  words: [],
  refData: {},         // word -> { videoURL, frames: [{xySkel, hands}] }
  wordIndex: 0,
  caught: 0,
  missed: 0,
  hintOn: false,
  balloon: null,        // { el, start, duration, vpHeight, resolved }
  attempt: null,         // { subState: "waitingReady"|"idle"|"recording", buffer, lastSample, smoother }
                         //   waitingReady: hands not yet confirmed down
                         //   idle:         ready, watching for at least one wrist up
                         //   recording:    capturing an attempt, watching for both wrists down
  raf: null,
};

// ------------------------------------------------------------------
// Small local helpers (URL video load, hand-landmark slice, timing)
// ------------------------------------------------------------------
// Same console-timing convention as mediapipe_loader.js's timed() -- logs
// how long an already-started promise took to settle, so boot()'s steps
// show up as [boot] lines for a real per-step breakdown.
async function timed(label, promise) {
  const t0 = performance.now();
  try {
    return await promise;
  } finally {
    console.log(`[boot] ${label}: ${(performance.now() - t0).toFixed(0)} ms`);
  }
}

// Pull the two 21-point hands back out of a flipped 75-slot xyImage
// (skeleton.js:combineXYImage layout) -- both the reference (precompute)
// and live paths already build this array via frameSkeleton(), so hand
// landmarks for scoring come from the SAME flipped space as xySkel
// instead of needing a second, separately-flipped copy.
function handsFromXYImage(xyImage) {
  const left  = xyImage.slice(LEFT_HAND_START,  LEFT_HAND_START  + HAND_LANDMARK_COUNT);
  const right = xyImage.slice(RIGHT_HAND_START, RIGHT_HAND_START + HAND_LANDMARK_COUNT);
  const hasLeft  = left.some(p => p.x !== 0 || p.y !== 0);
  const hasRight = right.some(p => p.x !== 0 || p.y !== 0);
  return { left: hasLeft ? left : null, right: hasRight ? right : null };
}

// HandLandmarker (numHands: 2) and PoseLandmarker (numPoses: 1) run as
// independent detectors -- MediaPipe caps hands at 2 TOTAL across the whole
// frame, not "2 hands belonging to the one tracked pose". With a second
// person in frame, a bystander's hand can fill one of those 2 slots (or
// outright displace the player's own hand) and get scored as if it were
// the player's. Reject any hand whose own wrist (its landmark 0) lands
// implausibly far from the tracked pose's corresponding wrist (15 left,
// 16 right) -- the two models' wrist estimates for the SAME real hand
// normally agree within a couple percent of frame size, while two
// different people standing apart don't.
const L_WRIST_IDX = 15, R_WRIST_IDX = 16;
const HAND_WRIST_MAX_DIST = 0.08;  // normalized (image-space) fraction of frame size

function isZeroPoint(p) {
  return p.x === 0 && p.y === 0;
}

function filterStrayHands(xyImage, xySkel) {
  const outImg  = xyImage.slice();
  const outSkel = xySkel.slice();
  for (const [poseWristIdx, handStart] of [[L_WRIST_IDX, LEFT_HAND_START], [R_WRIST_IDX, RIGHT_HAND_START]]) {
    const poseWrist = xyImage[poseWristIdx];
    const handWrist = xyImage[handStart];  // hand's own landmark 0 = its wrist
    if (isZeroPoint(poseWrist) || isZeroPoint(handWrist)) continue;  // nothing to compare -- leave as-is
    const dist = Math.hypot(poseWrist.x - handWrist.x, poseWrist.y - handWrist.y);
    if (dist > HAND_WRIST_MAX_DIST) {
      for (let i = 0; i < HAND_LANDMARK_COUNT; i++) {
        outImg[handStart + i]  = { x: 0, y: 0 };
        outSkel[handStart + i] = { x: 0, y: 0 };
      }
    }
  }
  return { xyImage: outImg, xySkel: outSkel };
}

// ------------------------------------------------------------------
// Word list + precompute
// ------------------------------------------------------------------
async function pickWords() {
  const res = await fetch("word_list.json");
  const manifest = await res.json();
  const pool = manifest.words.slice();
  const picked = [];
  for (let i = 0; i < WORD_POOL_SIZE && pool.length; i++) {
    const idx = Math.floor(Math.random() * pool.length);
    picked.push(pool.splice(idx, 1)[0]);
  }
  return picked;
}

// pairsToXY: [[x,y], ...] (JSON-friendly, as precompute_skeletons.py writes
// it) -> [{x,y}, ...], matching the {x,y} object shape frameSkeleton()
// produces and scoring.js/dtw.js expect.
function pairsToXY(pairs) {
  return pairs.map(([x, y]) => ({ x, y }));
}

async function precomputeWord(word) {
  const url = `dataset/${encodeURIComponent(word)}.mp4`;
  const res = await fetch(`dataset/skeletons/${encodeURIComponent(word)}.json`);
  if (!res.ok) {
    throw new Error(`precomputed skeleton for "${word}" not found (${res.status}) -- run precompute_skeletons.py`);
  }
  const data = await res.json();
  const frames = data.frames.map(f => ({
    xySkel: pairsToXY(f.xySkel),
    hands: {
      left:  f.hands.left  ? pairsToXY(f.hands.left)  : null,
      right: f.hands.right ? pairsToXY(f.hands.right) : null,
    },
  }));
  return { videoURL: url, frames };
}

async function precomputeAll(words) {
  const out = {};
  await Promise.all(words.map(async word => {
    out[word] = await timed(`precomputeWord "${word}"`, precomputeWord(word));
  }));
  return out;
}

// ------------------------------------------------------------------
// Webcam
// ------------------------------------------------------------------
async function setupWebcam() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false,
  });
  webcamEl.srcObject = stream;
  await new Promise(resolve => { webcamEl.onloadedmetadata = resolve; });
  await webcamEl.play();
  canvas.width  = webcamEl.videoWidth  || 1280;
  canvas.height = webcamEl.videoHeight || 720;
}

// ------------------------------------------------------------------
// HUD (pips + numbers)
// ------------------------------------------------------------------
function makePip(kind) {
  const p = document.createElement("span");
  p.className = kind === "life" ? "pip life" : "pip";
  p.textContent = kind === "life" ? PIP_LIFE_OK : PIP_CAUGHT_EMPTY;
  return p;
}

function buildPips() {
  pipsCaught.innerHTML = "";
  for (let i = 0; i < TARGET_BALLOONS; i++) pipsCaught.appendChild(makePip("caught"));
  pipsLives.innerHTML = "";
  for (let i = 0; i < LIVES; i++) pipsLives.appendChild(makePip("life"));
}

function renderHud() {
  numCaught.textContent = state.caught;
  numLives.textContent  = Math.max(LIVES - state.missed, 0);

  const caughtPips = pipsCaught.children;
  for (let i = 0; i < caughtPips.length; i++) {
    const filled = i < state.caught;
    caughtPips[i].classList.toggle("filled", filled);
    caughtPips[i].textContent = filled ? PIP_CAUGHT_FULL : PIP_CAUGHT_EMPTY;
  }
  const livePips = pipsLives.children;
  for (let i = 0; i < livePips.length; i++) {
    const lost = i < state.missed;
    livePips[i].classList.toggle("lost", lost);
    livePips[i].textContent = lost ? PIP_LIFE_LOST : PIP_LIFE_OK;
  }
}

// ------------------------------------------------------------------
// Hint video
// ------------------------------------------------------------------
// The hint video is sized to exactly 0.75 * the webcam viewport's own
// rendered height, with width left to the browser (object-fit isn't
// involved -- the video's native aspect ratio just determines its width at
// that height), so there's never any cropping or letterboxing. Since
// #hint-video's height is set directly (not the panel around it), the
// panel auto-sizes to hug the video via its own flex/padding, and --
// because #hint-panel isn't a flex sibling `.viewport` grows into -- toggling
// it on/off never changes the webcam viewport's size (see .viewport's fixed
// flex-basis in style.css).
function syncHintVideoHeight() {
  hintVideo.style.height = (viewport.getBoundingClientRect().height * 0.75) + "px";
}

function updateHintVideo() {
  const word = state.words[state.wordIndex];
  const ref  = word && state.refData[word];
  if (!ref) { hintPanel.hidden = true; return; }

  if (hintVideo.getAttribute("src") !== ref.videoURL) hintVideo.src = ref.videoURL;

  if (state.hintOn && state.phase === "playing" && state.balloon && state.balloon.hintReached) {
    hintPanel.hidden = false;
    syncHintVideoHeight();
    hintVideo.currentTime = 0;
    hintVideo.play().catch(() => {});
  } else {
    hintPanel.hidden = true;
    hintVideo.pause();
  }
}

window.addEventListener("resize", () => {
  if (!hintPanel.hidden) syncHintVideoHeight();
});

function toggleHint() {
  state.hintOn = !state.hintOn;
  btnHint.classList.toggle("active", state.hintOn);
  btnHint.setAttribute("aria-pressed", state.hintOn ? "true" : "false");
  // Left-align the stage row (instead of centering it) while hint mode is
  // on, so the webcam shifts over to make room for the hint panel beside
  // it rather than staying centered with the panel tacked onto one side.
  stageRow.classList.toggle("hint-on", state.hintOn);
  updateHintVideo();
}

// ------------------------------------------------------------------
// Feedback banner + burst particles (mirrors the mockup's effects)
// ------------------------------------------------------------------
// opts.top overlays the default top:34% position with the viewport's top
// center (used for the correct/incorrect labels); opts.pos overlays it with an
// explicit viewport-local pixel position (used for "Missed!", at the
// balloon).
function showFeedback(good, text, opts = {}) {
  const f = document.createElement("div");
  f.className = "feedback " + (good ? "show-good" : "show-bad");
  if (opts.top) f.classList.add("feedback--top");
  if (opts.pos) {
    f.style.left = opts.pos.x + "px";
    f.style.top  = opts.pos.y + "px";
  }
  f.textContent = text;
  viewport.appendChild(f);
  setTimeout(() => f.remove(), 900);
}

function burstAt(xPx, yPx, good) {
  const wrap = document.createElement("div");
  wrap.className = "burst" + (good ? "" : " bad");
  wrap.style.left = xPx + "px";
  wrap.style.top  = yPx + "px";

  const ring = document.createElement("div");
  ring.className = "burst-ring";
  wrap.appendChild(ring);

  const n = good ? 10 : 8;
  for (let i = 0; i < n; i++) {
    const shard = document.createElement("div");
    shard.className = "shard";
    const angle = (Math.PI * 2 * i) / n + Math.random() * 0.4;
    const dist  = 46 + Math.random() * 30;
    shard.animate(
      [
        { transform: "translate(-50%,-50%) translate(0,0)", opacity: 1 },
        { transform: `translate(-50%,-50%) translate(${Math.cos(angle) * dist}px,${Math.sin(angle) * dist}px)`, opacity: 0 },
      ],
      { duration: 550, easing: "cubic-bezier(.2,.7,.3,1)" },
    );
    wrap.appendChild(shard);
  }
  viewport.appendChild(wrap);
  setTimeout(() => wrap.remove(), 600);
}

// ------------------------------------------------------------------
// Balloon lifecycle
// ------------------------------------------------------------------
function spawnBalloon() {
  const word  = state.words[state.wordIndex];
  const color = COLORS[state.wordIndex % COLORS.length];

  const el = document.createElement("div");
  el.className = "balloon balloon--" + color;
  el.innerHTML =
    `<div class="balloon-body"><span class="balloon-word">${word}</span></div>` +
    '<div class="balloon-knot"></div>' +
    '<div class="balloon-string"></div>';

  const xPct = 30 + Math.random() * 40;
  el.style.left = xPct + "%";
  // NOT appended to the viewport yet -- it only appears (and its fall
  // timer only starts) once the hip line is confirmed visible, via
  // revealBalloon() (called from tick() as soon as hipVisible is true).

  const vpRect = viewport.getBoundingClientRect();
  // hintReached: true once the balloon's fall has passed HINT_REVEAL_FRACTION
  // of the viewport height -- the hint video (if toggled on) only starts
  // then, not the instant the balloon appears (see updateBalloonPosition()).
  state.balloon = { el, start: null, duration: FALL_MS, vpHeight: vpRect.height, resolved: false, revealed: false, hintReached: false };
  // Always start a new balloon requiring hands-down confirmation first --
  // covers both the very first balloon and a balloon that spawns right
  // after a timeout interrupted a still-in-progress attempt (hands may
  // still be up at that instant).
  state.attempt = { subState: "waitingReady", buffer: [], lastSample: 0, smoother: new TemporalSmoother() };

  updateHintVideo();
}

function revealBalloon(now) {
  const b = state.balloon;
  if (!b || b.revealed) return;
  b.revealed = true;
  // The balloon (full body + word) appears and holds still at its starting
  // position for BALLOON_APPEAR_MS before it starts falling -- updateBalloonPosition()
  // skips moving it while `now < b.start`, so the fall itself (and its
  // FALL_MS budget) only really begins once this hold is over.
  b.start = now + BALLOON_APPEAR_MS;
  viewport.appendChild(b.el);
  // The base transform's -140% vertical offset (see updateBalloonPosition)
  // is relative to the element's OWN height, so it leaves the balloon's
  // bottom edge sitting 0.4 * elHeight above wherever `y` alone would put
  // it -- measure the real rendered height (only known once it's in the
  // DOM) so the fall distance can compensate and the balloon's bottom
  // actually reaches the viewport floor at progress 1, instead of
  // stopping short and exploding mid-air.
  b.elHeight = b.el.getBoundingClientRect().height;
  // Hides any hint video still showing/playing for the PREVIOUS balloon's
  // word -- this new balloon's hintReached is false, so updateHintVideo()
  // won't show it again until updateBalloonPosition() flips that once this
  // one's fallen HINT_REVEAL_FRACTION of the way down.
  updateHintVideo();
}

function popBalloon(good) {
  const b = state.balloon;
  b.resolved = true;
  b.el.classList.add(good ? "pop-good" : "pop-bad");

  const rect   = b.el.getBoundingClientRect();
  const vpRect = viewport.getBoundingClientRect();
  const cx = rect.left - vpRect.left + rect.width / 2;
  const cy = Math.min(rect.top - vpRect.top + rect.height / 2, vpRect.height - 16);
  burstAt(cx, cy, good);
  showFeedback(good, good ? "✔ ถูกต้อง" : "Missed!", good ? { top: true } : { pos: { x: cx, y: cy } });

  setTimeout(() => {
    b.el.remove();
    state.balloon = null;
    advanceWord();
  }, 320);
}

function catchBalloon() {
  const b = state.balloon;
  if (!b || b.resolved) return;
  state.caught++;
  renderHud();
  popBalloon(true);
}

function missBalloon() {
  const b = state.balloon;
  if (!b || b.resolved) return;
  // A live attempt may be mid-flight when the timer runs out -- drop it,
  // the balloon exploding takes precedence. Go back to "waitingReady"
  // (not "idle") since hands may still be up right now; spawnBalloon()
  // resets this properly for the next word regardless, but this also
  // covers the dead window before that happens (e.g. game-over).
  if (state.attempt && state.attempt.subState === "recording") {
    state.attempt.subState = "waitingReady";
    recBadge.hidden = true;
  }
  state.missed++;
  renderHud();
  popBalloon(false);
}

function advanceWord() {
  if (state.caught >= TARGET_BALLOONS) return endGame(true);
  if (state.missed >= LIVES) return endGame(false);
  state.wordIndex++;
  spawnBalloon();
}

// ------------------------------------------------------------------
// Attempt scoring
// ------------------------------------------------------------------
function finishAttempt() {
  const attempt = state.attempt;
  attempt.subState = "idle";
  recBadge.hidden = true;
  // Too short to be a real attempt (e.g. a single noisy pose blip) --
  // ignore it rather than scoring near-nothing as "not correct".
  if (attempt.buffer.length < 2) return;

  const word   = state.words[state.wordIndex];
  const refSeq = state.refData[word].frames;
  const result = scoreAttempt(refSeq, attempt.buffer, state.rules, AGG);
  const correct = isCorrect(result.pairSimilarityPct);

  if (DEBUG) renderDebugPanel(word, result, correct);

  if (correct) {
    catchBalloon();
  } else {
    showFeedback(false, "X ผิด", { top: true });
  }
}

// Dumps every DTW-aligned pair's hand/wrist/total score -- the same numbers
// scoreAttempt() computes internally but normally throws away in favor of
// just correct/incorrect -- so a scoring issue can be diagnosed without
// re-instrumenting the code. Appends (doesn't replace) so every attempt
// across the whole play stays visible, not just the latest one -- cleared
// only at the start of a new round (see startPlaying()).
function renderDebugPanel(word, result, correct) {
  const fmt = v => (v === null ? "  -  " : v.toFixed(1).padStart(5));
  const lines = [
    `word: ${word}`,
    `avg: ${fmt(result.avg)}   pairSimilarityPct: ${fmt(result.pairSimilarityPct)}   ` +
      (correct ? "CORRECT" : "not correct"),
    "",
    " ri  ti | hand  wrist  total",
    "-----------------------------",
  ];
  for (const p of result.pairs) {
    lines.push(
      `${String(p.ri).padStart(3)} ${String(p.ti).padStart(3)} | ` +
      `${fmt(p.hand)} ${fmt(p.wrist)} ${fmt(p.total)}`,
    );
  }
  const block = lines.join("\n");
  debugPanel.textContent += (debugPanel.textContent ? "\n\n" : "") + block;
  debugPanel.scrollTop = debugPanel.scrollHeight;
}

// ------------------------------------------------------------------
// Live loop
// ------------------------------------------------------------------
// Same bone pairs as test_dtw_lasso_similarity.py's HAND_CONNECTIONS.
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];
const HIP_LINE_COLOR       = "#9AA0A6";  // grey
const BONE_COLOR           = "rgba(220,220,220,.85)";
const LANDMARK_GREY        = "#9AA0A6";  // dots outside the active attempt window
const LEFT_HAND_COLOR      = "#37B7A0";
const RIGHT_HAND_COLOR     = "#8C7AE6";

// Raw per-frame pose landmarks are noisy enough that the hip line visibly
// wobbles/flickers every frame. Two independent smoothing passes, both
// applied ONLY to what's drawn -- gating (hipVisible/anyUp below) and the
// attempt buffer keep using the raw per-frame landmarks unchanged, so
// recording timing/scoring accuracy isn't affected.
//
// 1) Position: EMA-smooth the drawn xyImage so the line/skeleton don't
//    visibly jitter frame to frame. Zero-aware, unlike TemporalSmoother:
//    a point only blends with its previous value when BOTH are actually
//    detected -- otherwise it snaps straight to the raw value. Plain EMA
//    would blend a just-vanished landmark's (0,0) toward its last real
//    position, decaying asymptotically instead of ever hitting exact zero,
//    so a hand leaving the frame would drift toward the canvas corner and
//    hover there instead of just disappearing.
function makeOverlaySmoother(alpha = 0.6) {
  let prev = null;
  return current => {
    if (!prev) { prev = current.map(p => ({ x: p.x, y: p.y })); return prev; }
    const out = current.map((c, i) => {
      const p = prev[i];
      const cZero = c.x === 0 && c.y === 0;
      const pZero = p.x === 0 && p.y === 0;
      if (cZero || pZero) return { x: c.x, y: c.y };
      return { x: alpha * c.x + (1 - alpha) * p.x, y: alpha * c.y + (1 - alpha) * p.y };
    });
    prev = out.map(p => ({ x: p.x, y: p.y }));
    return out;
  };
}
const overlaySmoother = makeOverlaySmoother();
// 2) Presence: a single noisy frame can flip hipLandmarksConfident() true
//    for just an instant -- since revealBalloon()/subState transitions are
//    edge-triggered on ONE true frame, that's enough to reveal a balloon
//    even though the line looked "not really there" to the player. Only
//    trust a change once it holds for HIP_VISIBLE_DEBOUNCE_FRAMES straight
//    frames, for both the drawn line and the ready/recording gating that
//    reads the same signal.
function makeDebouncer(framesRequired) {
  let stable = false;
  let streak = 0;
  return raw => {
    if (raw === stable) { streak = 0; return stable; }
    streak++;
    if (streak >= framesRequired) { stable = raw; streak = 0; }
    return stable;
  };
}
const HIP_VISIBLE_DEBOUNCE_FRAMES = 4;
// Reassigned (not just used) at the start of every round -- see
// startPlaying() -- so a round can't inherit a `stable: true` left over
// from however the *previous* round happened to end. tick() stops
// entirely between rounds (endGame() cancels the raf loop), so nothing
// re-samples the player's real position during that idle window; without
// a fresh debouncer, a replay where the player never physically left frame
// would satisfy "hip visible" on literally the first live frame, skipping
// the wait it's meant to enforce.
let hipVisibleDebounced = makeDebouncer(HIP_VISIBLE_DEBOUNCE_FRAMES);

// MediaPipe's PoseLandmarker keeps outputting a full 33-point skeleton
// (a "best guess") for as long as VIDEO mode's tracker believes there's a
// person in frame -- it doesn't just go empty the moment the hips are
// unclear, so extendedHipPoints() alone (a purely geometric check) isn't
// enough to catch "nobody's really there" or "the tracker's guessing".
// Each landmark also carries a visibility score (0..1); require the hip
// landmarks specifically to be confidently visible before trusting them.
const MIN_HIP_VISIBILITY = 0.7;
const L_HIP_IDX = 23, R_HIP_IDX = 24;

function hipLandmarksConfident(poseRes) {
  if (!poseRes || !poseRes.landmarks || poseRes.landmarks.length === 0) return false;
  const lm = poseRes.landmarks[0];
  const lh = lm[L_HIP_IDX], rh = lm[R_HIP_IDX];
  if (!lh || !rh) return false;
  const visOf = p => (typeof p.visibility === "number" ? p.visibility : 1);
  return visOf(lh) >= MIN_HIP_VISIBILITY && visOf(rh) >= MIN_HIP_VISIBILITY;
}

function drawMirroredFrame() {
  const ctx = canvas.getContext("2d");
  ctx.save();
  ctx.setTransform(-1, 0, 0, 1, canvas.width, 0);
  ctx.drawImage(webcamEl, 0, 0, canvas.width, canvas.height);
  ctx.restore();
}

// xyImage is already flipX:true'd (frameSkeleton), so drawing its x/y
// straight against the canvas lines up with the mirrored video underneath
// -- no extra mirroring needed here, unlike a raw MediaPipe result.
// `visible` is the SAME confidence-gated check (hipVisible in tick()) that
// drives the state machine and the "please stand..." hint -- draw only
// when it's true, so the line, the hint text, and the actual game logic
// can never disagree about whether the hip line has really been found.
function drawHipLineOverlay(ctx, xyImage, visible) {
  if (!visible) return;
  const line = extendedHipPoints(xyImage);
  if (!line) return;
  const [p1, p2] = line;
  const x1 = canvas.width * p1.x, y1 = canvas.height * p1.y;
  const x2 = canvas.width * p2.x, y2 = canvas.height * p2.y;

  // Dark outline underneath so the line reads clearly against any
  // background brightness/color, same trick as the Python script's
  // outlined on-screen text.
  ctx.strokeStyle = "rgba(0,0,0,.6)";
  ctx.lineWidth = 11;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();

  ctx.strokeStyle = HIP_LINE_COLOR;
  ctx.lineWidth = 7;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
}

function drawHandSkeleton(ctx, handXY, color) {
  if (!handXY) return;
  const pts = handXY.map(p => (p.x === 0 && p.y === 0)
    ? null
    : [canvas.width * p.x, canvas.height * p.y]);

  ctx.strokeStyle = BONE_COLOR;
  ctx.lineWidth = 3;
  for (const [a, b] of HAND_CONNECTIONS) {
    if (pts[a] && pts[b]) {
      ctx.beginPath();
      ctx.moveTo(pts[a][0], pts[a][1]);
      ctx.lineTo(pts[b][0], pts[b][1]);
      ctx.stroke();
    }
  }

  ctx.fillStyle = color;
  for (const p of pts) {
    if (!p) continue;
    ctx.beginPath();
    ctx.arc(p[0], p[1], 7, 0, Math.PI * 2);
    ctx.fill();
  }
}

// Hip line + both hand skeletons, same visual language as
// test_dtw_lasso_similarity.py's draw_hip_line/draw_hand (left = teal,
// right = violet -- this app's own balloon accent colors, since there's
// no per-landmark correct/incorrect verdict to color by outside of a
// just-finished attempt). Dots are grey outside the active attempt window
// (`active` = attempt.subState === "recording", i.e. from the crop start --
// a wrist crossing above the hip line -- to the crop stop -- both wrists
// back down) and switch to their per-hand color for the window itself, so
// the skeleton visually marks what's actually being scored.
function drawOverlay(xyImage, visible, active) {
  const ctx = canvas.getContext("2d");
  drawHipLineOverlay(ctx, xyImage, visible);
  const hands = handsFromXYImage(xyImage);
  drawHandSkeleton(ctx, hands.left,  active ? LEFT_HAND_COLOR  : LANDMARK_GREY);
  drawHandSkeleton(ctx, hands.right, active ? RIGHT_HAND_COLOR : LANDMARK_GREY);
}

// Piecewise-linear easing: the first FAST_T fraction of the fall's TIME
// covers FAST_D of its DISTANCE (steep), and the remaining time covers the
// rest of the distance (shallow) -- "drops fast at first, then slows down",
// while still landing at exactly 1 when t=1 so the FALL_MS budget/timeout
// is unaffected. `t`/return value are both fractions in [0,1].
const FAST_T = 0.4, FAST_D = 0.6;
function fallEase(t) {
  if (t <= FAST_T) return (t / FAST_T) * FAST_D;
  return FAST_D + ((t - FAST_T) / (1 - FAST_T)) * (1 - FAST_D);
}

function updateBalloonPosition(now) {
  const b = state.balloon;
  if (!b || b.resolved || !b.revealed) return;
  const elapsed = now - b.start;
  if (elapsed < 0) return;  // still in the appear-hold -- stay put at the CSS default position
  const progress = Math.min(elapsed / b.duration, 1);
  // +0.4*elHeight compensates the base -140% transform offset (see
  // revealBalloon) so the balloon's bottom edge -- not some point above
  // it -- is what reaches the viewport floor at progress 1. fallEase()
  // shapes progress (elapsed TIME fraction) into a distance fraction --
  // missBalloon()'s timeout below still keys off the raw `progress`, so
  // this only changes how the balloon looks moving, not the deadline.
  const y    = fallEase(progress) * (b.vpHeight + 0.4 * b.elHeight);
  const sway = Math.sin(elapsed / 750) * 16;
  b.el.style.transform = `translate(calc(-50% + ${sway}px), calc(-140% + ${y}px))`;
  if (!b.hintReached && y >= b.vpHeight * HINT_REVEAL_FRACTION) {
    b.hintReached = true;
    updateHintVideo();  // hint (if toggled on) only starts once the balloon is 25% down the frame
  }
  if (progress >= 1) missBalloon();
}

function tick(now) {
  if (state.phase !== "playing") return;

  try {
    drawMirroredFrame();

    const ts      = state.mp.nextTs();
    const handRes = state.mp.handVideo.detectForVideo(webcamEl, ts);
    const poseRes = state.mp.poseVideo.detectForVideo(webcamEl, ts);

    // filterStrayHands() drops any detected hand that isn't actually
    // attached to the tracked pose (see its own comment) -- applied here,
    // before anything downstream (drawing, the attempt buffer, hip-line
    // gating) ever sees raw.xyImage/raw.xySkel, so a bystander's hand can't
    // slip into a scored attempt.
    const combined = frameSkeleton(poseRes, handRes, null, { flipX: true });
    const raw = filterStrayHands(combined.xyImage, combined.xySkel);

    // anyWristAboveHipline() alone can't tell "confirmed both down" apart
    // from "no hip line at all" (nobody in frame, pose lost) -- both read
    // as false. Gate the ready/start transitions on the hip line actually
    // being visible (geometrically computable AND confidently detected --
    // see hipLandmarksConfident()) so an empty or unclear frame can never
    // look like "hands down", and a spurious low-confidence pose guess
    // can never look like "hands up". Debounced (see hipVisibleDebounced
    // above) so a single noisy frame can't reveal a balloon (or flip the
    // hint text) the player never actually got confirmed-ready for.
    const rawHipVisible = hipLandmarksConfident(poseRes) && extendedHipPoints(raw.xyImage) !== null;
    const hipVisible = hipVisibleDebounced(rawHipVisible);
    const anyUp = anyWristAboveHipline(raw.xyImage);
    const handsConfirmedDown = hipVisible && !anyUp;
    const handsConfirmedUp   = hipVisible && anyUp;

    hipHint.hidden = hipVisible;

    const attempt = state.attempt;
    const smoothedXY = overlaySmoother(raw.xyImage);
    drawOverlay(smoothedXY, hipVisible, !!attempt && attempt.subState === "recording");

    if (attempt) {
      // The balloon (and its fall timer) stays hidden until the hip line
      // itself is visible -- independent of the arming transition below,
      // which additionally requires hands confirmed down.
      if (attempt.subState === "waitingReady" && hipVisible) {
        revealBalloon(now);
      }

      if (attempt.subState === "waitingReady" && handsConfirmedDown) {
        attempt.subState = "idle";  // both hands confirmed down -- armed, watching for a wrist to rise
      }

      if (attempt.subState === "idle" && handsConfirmedUp) {
        attempt.subState  = "recording";
        attempt.buffer     = [];
        attempt.lastSample = 0;
        attempt.smoother.reset();
        recBadge.hidden = false;
      }

      if (attempt.subState === "recording") {
        if (now - attempt.lastSample >= LIVE_SAMPLE_MS) {
          const xySkel = attempt.smoother.apply(raw.xySkel);
          attempt.buffer.push({ xySkel, hands: handsFromXYImage(raw.xyImage) });
          attempt.lastSample = now;
        }
        // Loose stop condition on purpose: if the hip line drops out
        // mid-recording (person stepped out of frame), stopping rather
        // than continuing to buffer garbage is the safe default.
        if (!anyUp) finishAttempt();
      }
    }

    updateBalloonPosition(now);
  } catch (err) {
    // Never let one bad frame silently freeze the whole game -- log it and
    // keep the loop alive so the next frame gets a chance to recover.
    console.error("tick() error:", err);
  }

  state.raf = requestAnimationFrame(tick);
}

// ------------------------------------------------------------------
// Phases
// ------------------------------------------------------------------
function startPlaying() {
  state.phase     = "playing";
  state.wordIndex = 0;
  state.caught    = 0;
  state.missed    = 0;
  // Fresh debouncer every round -- see its declaration for why: otherwise
  // a replay can inherit a stale "hip visible" reading from the end of the
  // previous round and skip the wait entirely.
  hipVisibleDebounced = makeDebouncer(HIP_VISIBLE_DEBOUNCE_FRAMES);
  if (DEBUG) debugPanel.textContent = "";  // fresh log per round, not per attempt
  // Defensive: a balloon whose pop cleanup didn't run (e.g. an error mid-
  // attempt) could otherwise leave a stray, still-falling element behind
  // for a fresh round to reveal next to.
  viewport.querySelectorAll(".balloon").forEach(el => el.remove());
  renderHud();
  overlayStart.hidden = true;
  spawnBalloon();
  state.raf = requestAnimationFrame(tick);
}

// Restart the round with the SAME state.words (no re-picking/re-fetching) --
// cancel any in-flight raf loop first, since startPlaying() would otherwise
// start a second one running alongside it.
function restartGame() {
  if (state.raf) cancelAnimationFrame(state.raf);
  overlayStart.hidden = true;
  overlayEnd.hidden   = true;
  recBadge.hidden     = true;
  startPlaying();
}

function endGame(won) {
  state.phase = won ? "won" : "lost";
  if (state.raf) cancelAnimationFrame(state.raf);
  recBadge.hidden = true;
  hintPanel.hidden = true;
  hipHint.hidden = true;
  endTitle.textContent = won ? "คุณชนะ !" : "Game Over";
  endText.hidden = true;
  overlayEnd.hidden = false;
}

async function playAgain() {
  overlayEnd.hidden   = true;
  overlayLoading.hidden = false;
  loadingText.textContent = "กำลังโหลดคำใหม่...";

  const words = await pickWords();
  state.words   = words;
  state.refData = await precomputeAll(words);

  overlayLoading.hidden = true;
  overlayStart.hidden = false;
}

async function boot() {
  const bootT0 = performance.now();
  console.log(`[boot] rules bundle: ${RULES_NAME}`);
  try {
    const [mp, rules, words] = await Promise.all([
      timed("loadMediaPipe (bundle+wasm+fileset)", loadMediaPipe()),
      timed("loadRules", loadRules(`../shared/rules/hand_rules_${RULES_NAME}.json`)),
      timed("pickWords", pickWords()),
      timed("setupWebcam (incl. permission prompt)", setupWebcam()),
    ]);
    state.mp    = mp;
    state.rules = rules;
    state.words = words;

    // Pre-warm the VIDEO-mode detectors here (loading screen already shown)
    // so startPlaying()'s tick() loop never hits a null handVideo/poseVideo —
    // ensure*() is memoized, so this is a no-op if something else built them.
    // No IMAGE-mode detectors to build anymore: reference sequences are
    // fetched pre-baked from dataset/skeletons/ (precompute_skeletons.py).
    const [refData] = await Promise.all([
      timed(`precomputeAll (${words.length} words total)`, precomputeAll(words)),
      timed("ensureHandVideo", mp.ensureHandVideo()),
      timed("ensurePoseVideo", mp.ensurePoseVideo()),
    ]);
    state.refData = refData;

    console.log(`[boot] TOTAL: ${(performance.now() - bootT0).toFixed(0)} ms`);

    overlayLoading.hidden = true;
    overlayStart.hidden = false;
    btnHint.disabled = false;
    btnRestart.disabled = false;
  } catch (err) {
    console.error(err);
    loadingText.textContent = "Something went wrong: " + (err.message || err) +
      ". Reload the page to try again (and allow camera access).";
  }
}

btnStart.addEventListener("click", startPlaying);
btnRestart.addEventListener("click", restartGame);
btnAgain.addEventListener("click", playAgain);
btnHint.addEventListener("click", toggleHint);

// Pips + counts don't depend on anything boot() loads -- build them
// immediately so ⚪⚪⚪/💣💣💣 show next to the "0/3"/"3/3" labels from the
// very first paint, instead of staying empty through the whole loading
// screen.
buildPips();
renderHud();
boot();
