"""ISL ⇄ TEXT — Streamlit app: live webcam, inference state machine, feedback.

Run locally:  streamlit run app.py
Or containerized:  docker compose up --build
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from src.feedback import FeedbackRecorder, WebhookNotifier, record_feedback
from src.inference import InferenceEngine
from src.model import SequenceModelConfig, build_model
from src.preprocessing import extract_features
from src.utils import load_config, log, select_device
from src.webcam import CameraOfflineError, Webcam

st.set_page_config(page_title="ISL ⇄ TEXT", page_icon="🤟", layout="wide")

CFG_PATH = Path(__file__).parent / "configs" / "config.yaml"


@st.cache_resource
def load_classes_and_model(cfg: dict):
    classes_path = Path(cfg["data"]["classes_file"])
    classes = [line.strip() for line in classes_path.read_text(encoding="utf-8")
               if line.strip() and not line.startswith("#")]

    model_cfg = SequenceModelConfig(
        feature_dimension=int(cfg["model"]["feature_dimension"]),
        num_classes=int(cfg["model"]["num_classes"]),
        embedding_dim=int(cfg["model"]["embedding_dim"]),
        num_heads=int(cfg["model"]["num_heads"]),
        transformer_layers=int(cfg["model"]["transformer_layers"]),
        ff_dim=int(cfg["model"]["ff_dim"]),
        dropout=float(cfg["model"]["dropout"]),
    )
    model = build_model(cfg["model"]["name"], model_cfg)
    # Load production weights when present; otherwise run with the fresh model.
    prod = Path(cfg["paths"]["models_production"])
    candidates = sorted(prod.glob("*.pth"))
    if candidates:
        import torch
        model.load_state_dict(torch.load(candidates[-1], map_location="cpu"))
        log.info("loaded production weights: %s", candidates[-1].name)
    return classes, model


def main() -> None:
    cfg = load_config(CFG_PATH)
    st.title(cfg["app"]["title"])
    st.caption("Real webcam → MediaPipe landmarks → temporal transformer → committed text.")

    classes, model = load_classes_and_model(cfg)
    engine = InferenceEngine(
        model=model,
        classes=classes,
        device=select_device(),
        sequence_length=int(cfg["inference"]["sequence_length"]),
        min_frames=int(cfg["inference"]["min_frames"]),
        confidence_threshold=float(cfg["inference"]["confidence_threshold"]),
        temperature=float(cfg["inference"]["temperature"]),
        ema_alpha=float(cfg["inference"]["ema_alpha"]),
        stability_window=int(cfg["inference"]["stability_window"]),
        cooldown_seconds=float(cfg["inference"]["cooldown_seconds"]),
    )
    recorder = FeedbackRecorder(cfg["feedback"]["local_log"])
    notifier = WebhookNotifier(cfg["feedback"].get("webhook_url"))

    run = st.checkbox("Start camera", value=False)
    sentence_box = st.empty()
    feedback_word = st.selectbox("Correct a wrong prediction (optional)",
                                 ["—"] + classes)

    if feedback_word != "—":
        record_feedback(recorder, notifier, "prediction_feedback",
                        {"correction": feedback_word, "shown_sentence": engine.sentence.sentence})
        st.success("Feedback recorded — thank you. A human reviews it before retraining.")

    if not run:
        st.info("Enable the checkbox to open the webcam. Frames stay on this device.")
        return

    try:
        import mediapipe as mp
        holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    except ImportError:
        st.error("mediapipe is not installed — see the troubleshooting section.")
        return

    with Webcam(index=int(cfg["camera"]["index"]),
                width=int(cfg["camera"]["width"]),
                height=int(cfg["camera"]["height"])) as cam:
        frame_ph = st.empty()
        try:
            cam.open()
        except CameraOfflineError:
            st.error("CAMERA_OFFLINE — check OS permissions for this app.")
            return

        while True:
            ok, frame_bgr = cam.read()
            if not ok:
                frame_ph.warning("CAMERA_OFFLINE — frame stream stalled.")
                time.sleep(0.2)
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            display = cv2.flip(frame_bgr, 1)
            res = holistic.process(frame_rgb)
            frame_rgb.flags.writeable = True

            features = extract_features(res)
            engine.push(features)
            pred = engine.predict()
            if pred.committed:
                record_feedback(recorder, notifier, "prediction_feedback",
                                {"word": pred.word, "confidence": round(pred.confidence, 4)})

            label = pred.word or ("no hands" if pred.status == "NO_HANDS" else "uncertain")
            cv2.putText(display, f"{label}  {pred.confidence:.2f}",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            sentence_box.code(engine.sentence.sentence or "(sentence empty)")
            frame_ph.image(cv2.cvtColor(display, cv2.COLOR_BGR2RGB), channels="RGB")

            time.sleep(1 / float(cfg["camera"]["fps"]))


if __name__ == "__main__":
    main()
