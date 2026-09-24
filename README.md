# Balloon Sign Catch

A browser-based sign-language game: sign the Thai word shown on each falling
balloon before it hits the ground. Runs on client-side (MediaPipe 
for hand + pose via webcam)

## Setup

### 1. Add the reference videos

Copy word videos into `web/dataset/`, named to
match the word exactly as it appears in `web/word_list.json` and in
`web/dataset/skeletons/` (e.g. `web/dataset/สวัสดี.mp4` for the entry
`สวัสดี`).

### 2. Serve it

```bash
cd balloon_game
python3 -m http.server 8000
```

Then open **`http://localhost:8000/web/`**.

