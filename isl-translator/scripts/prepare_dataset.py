"""Record labelled webcam clips per signer, with consent and session metadata.

Consent is recorded per session; frames are never uploaded. Signer metadata
makes signer-aware splitting work on your data too.

Usage:
    python scripts/prepare_dataset.py --source include --out data/raw \
        --signer anon-01 --session s01 --labels HELLO,THANK-YOU
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from src.webcam import Webcam  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record labelled ISL sign clips.")
    p.add_argument("--source", default="webcam", help="capture source")
    p.add_argument("--out", default="data/raw", help="output directory")
    p.add_argument("--signer", required=True, help="anonymized signer id")
    p.add_argument("--session", required=True, help="session id")
    p.add_argument("--labels", required=True, help="comma-separated glosses")
    p.add_argument("--seconds-per-sign", type=float, default=2.0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--consent", action="store_true",
                   help="confirm recorded consent (required)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.consent:
        print("Recording requires explicit consent. Re-run with --consent after "
              "the signer has agreed. Frames are never uploaded.")
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    meta = {
        "signer_id": args.signer,
        "session_id": args.session,
        "consent": True,
        "source": args.source,
        "labels": [l.strip() for l in args.labels.split(",") if l.strip()],
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out / f"{args.signer}_{args.session}_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")

    with Webcam() as cam:
        cam.open()
        for label in meta["labels"]:
            print(f"Record '{label}' now — capturing {args.seconds_per_sign}s …")
            writer = None
            n_frames = int(args.seconds_per_sign * args.fps)
            for _ in range(n_frames):
                ok, frame = cam.read()
                if not ok:
                    continue
                if writer is None:
                    h, w = frame.shape[:2]
                    path = out / f"{args.signer}_{args.session}_{label}.avi"
                    writer = cv2.VideoWriter(str(path),
                                             cv2.VideoWriter_fourcc(*"MJPG"),
                                             args.fps, (w, h))
                writer.write(frame)
            if writer is not None:
                writer.release()
            time.sleep(0.3)
    print("Done. Metadata:", meta)


if __name__ == "__main__":
    main()
