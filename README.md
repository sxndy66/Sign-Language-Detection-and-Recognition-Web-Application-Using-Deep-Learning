# ISL ⇄ TEXT — Indian Sign Language Recognition Platform

A production engineering platform for **Indian Sign Language** recognition:
webcam frames become MediaPipe landmarks, a temporal Transformer reads the
movement, and a gated state machine commits words into sentences.

> No fabricated accuracy. 95% is the engineering **target** — numbers ship only
> with the run that produced them, in `metrics.json` on a held-out split.

## What's in this repository

| Path | What it is |
|------|------------|
| [`index.html`](./index.html) | The ISL ⇄ TEXT platform site — open it in a browser. Includes a live, in-browser demo (MediaPipe Hands WASM + few-shot prototype classifier). |
| [`isl-translator/`](./isl-translator/) | The full Python engineering project the site documents: `src/`, `configs/`, `scripts/`, `tests/`, `app.py`, Docker, CI. |

## The pipeline — ten stages

webcam → frame routing → MediaPipe Holistic → landmark normalization →
temporal buffer → spatial encoder → temporal transformer → confidence filter →
EMA smoothing + stability → sentence builder.

Feature schema is a fixed **186-D** row (63 left hand + 63 right hand + 33 pose
+ 24 face + 3 presence flags), asserted in tests, never assumed.

## Run the Python project

```bash
cd isl-translator
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# train / evaluate / run
python scripts/train_model.py --cfg configs/config.yaml --epochs 60
python scripts/evaluate_model.py --split test --checkpoint checkpoints/best_model.pth
streamlit run app.py
docker compose up --build
pytest -q
```

See [`isl-translator/README.md`](./isl-translator/README.md) for the full
quickstart, promotion gates, feedback loop, and field manual.

## View the site locally

```bash
python3 -m http.server 8080   # then open http://localhost:8080
```

## License

MIT — see [LICENSE](./LICENSE).
