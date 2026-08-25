# ISL ⇄ TEXT — Indian Sign Language Recognition

A production engineering platform for **Indian Sign Language** recognition:
webcam frames become MediaPipe landmarks, a temporal Transformer reads the
movement, and a gated state machine commits words into sentences.

> No fabricated accuracy. 95% is the engineering **target** — numbers ship only
> with the run that produced them, in `logs/metrics.json` on a held-out split.

## Why this exists

ISL and ASL are different languages with different phonology. Training on
WLASL produces WLASL results. This pipeline is built for ISL data (INCLUDE,
AI4Bharat) and refuses to mix languages — every sample is tagged with its
dataset version.

## The pipeline — ten stages

| # | Stage | Contract |
|---|-------|----------|
| 01 | Webcam capture | BGR · H×W×3 @ 30 fps, degrades to `CAMERA_OFFLINE` |
| 02 | Frame routing | BGR→RGB, mirrored, latest frame only |
| 03 | MediaPipe Holistic | 2×21×3 hands · 33 pose · face subset (12 pts) |
| 04 | Landmark normalization | wrist origin → unit scale → canonical roll |
| 05 | Temporal buffer | `deque(maxlen=60)` → (T, 186), min 30 frames |
| 06 | Spatial encoder | (B,T,186) → (B,T,256) linear + LayerNorm |
| 07 | Temporal transformer | 4 layers × 8 heads, attention pooling |
| 08 | Confidence filter | tempered softmax → hard threshold 0.60, abstain |
| 09 | Smoothing + stability | EMA α=0.35 → 12 stable → commit |
| 10 | Sentence builder | duplicate suppression, undo, clear, reset |

## Feature schema (186-D, asserted not assumed)

| Block | Content | Dims |
|-------|---------|------|
| Left hand | 21 × xyz, wrist-normalized | 63 |
| Right hand | 21 × xyz, wrist-normalized | 63 |
| Pose subset | 11 joints × xyz | 33 |
| Face subset | 12 points × xy | 24 |
| Presence flags | left / right / face visible | 3 |

## Quick start

```bash
git clone <this-repo> && cd isl-translator
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

```bash
# 02 — prepare data (your own recordings, consent-gated)
python scripts/prepare_dataset.py --signer anon-01 --session s01 \
    --labels HELLO,THANK-YOU --consent
python scripts/extract_landmarks.py --cfg configs/config.yaml   # cached .npz, incremental

# 03 — train
python scripts/train_model.py --cfg configs/config.yaml --epochs 60
# → checkpoints/best_model.pth + logs/metrics.json + confusion_matrix.png

# 04 — evaluate (held-out split)
python scripts/evaluate_model.py --split test --checkpoint checkpoints/best_model.pth

# 05 — run the app
streamlit run app.py          # webcam, feedback loop
docker compose up --build     # or containerized

# 06 — test & CI
pytest -q                     # data / model / inference / utils suites
gh workflow run tests.yml
```

## Promotion gates — all four, or no deploy

1. **Test metrics pass** — macro-F1 ≥ target on held-out set
2. **Regression tests pass** — pytest green on CI
3. **Latency budget** — p95 inference ≤ 40 ms on target device
4. **Human approval** — curator sign-off recorded in the registry

Automation can request training and report results; it can never promote. The
CI workflow stops at "upload candidate" — a human moves it forward, and the
previous production model is archived, never overwritten.

```
models/
├── candidate/     ← training output lands here
├── production/    ← write-locked, one approved model
└── archive/       ← every superseded version, forever
```

## Feedback → retraining loop

`Streamlit UI → Feedback API → Make.com / n8n → Sheet / DB → data review →
retrain queue → evaluation → model registry → approval gate → production`.

Feedback is always persisted locally first (`logs/feedback.jsonl`), so a dead
webhook never loses a correction — replay it with `scripts/replay_feedback.py`.

## Privacy & ethics — non-negotiable

- Recorded sessions require explicit consent; consent metadata travels with the data.
- Frames are processed locally. Only prediction metadata ever leaves the device, and only to a webhook you configure.
- Sign language is a full language. Gloss output is not a grammatical translation; never market it as one.
- Deaf signers review approved samples before any retraining. No silent dataset growth.

## License

MIT — see [LICENSE](LICENSE).
