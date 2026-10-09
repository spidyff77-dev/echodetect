#!/usr/bin/env python3
"""
EchoDetect - Inference Script
Classifies a sound file using a trained EchoDetect model (SVM or Random Forest).
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
import numpy as np
import soundfile as sf

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from config import CLASSES, MODELS_DIR, SAMPLE_RATE, N_MFCC, INCLUDE_DELTAS
from echodetect.features import extract_features_from_file
from echodetect.models import load_model


def parse_args():
    parser = argparse.ArgumentParser(description="EchoDetect: Run inference on an audio file")
    parser.add_argument("--audio", type=str, required=True,
                        help="Path to the WAV audio file to classify")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained model .joblib file (default: saved_models/svm_model.joblib or rf_model.joblib)")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Number of top predictions to display (default: 3)")
    return parser.parse_args()


def predict(audio_path: str | Path, model_path: Optional[str | Path] = None, top_k: int = 3):
    audio_path = Path(audio_path)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)

    # Locate model
    if model_path is None:
        svm_cand = MODELS_DIR / "svm_model.joblib"
        rf_cand = MODELS_DIR / "rf_model.joblib"
        if svm_cand.exists():
            model_path = svm_cand
        elif rf_cand.exists():
            model_path = rf_cand
        else:
            print("Error: No trained model found in saved_models/. Please train a model first using train.py.")
            sys.exit(1)
    else:
        model_path = Path(model_path)

    print(f"[*] Loading model from: {model_path}")
    pipeline, metadata = load_model(model_path)

    sr = metadata.get("sample_rate", SAMPLE_RATE)
    n_mfcc = metadata.get("n_mfcc", N_MFCC)
    include_deltas = metadata.get("include_deltas", INCLUDE_DELTAS)
    classes_dict = metadata.get("classes", CLASSES)

    # Inspect audio file
    info = sf.info(str(audio_path))
    print(f"[*] Audio Info: {audio_path.name}")
    print(f"    - Original Sample Rate: {info.samplerate} Hz")
    print(f"    - Channels: {info.channels} ({'mono' if info.channels == 1 else 'stereo'})")
    print(f"    - Duration: {info.duration:.2f} seconds")

    # Extract features
    print(f"[*] Extracting MFCC features (sr={sr}, n_mfcc={n_mfcc}, deltas={include_deltas})...")
    features = extract_features_from_file(
        audio_path=audio_path,
        sr=sr,
        n_mfcc=n_mfcc,
        include_deltas=include_deltas
    )
    features_2d = features.reshape(1, -1)

    # Run inference
    pred_class_id = pipeline.predict(features_2d)[0]
    pred_class_name = classes_dict.get(pred_class_id, f"Unknown ({pred_class_id})")

    print("\n" + "=" * 50)
    print(f"  PREDICTED CLASS: {pred_class_name.upper()} (ID: {pred_class_id})")
    print("=" * 50)

    # Probabilities
    if hasattr(pipeline, "predict_proba"):
        probas = pipeline.predict_proba(features_2d)[0]
        model_classes = pipeline.classes_
        
        # Rank by probability
        ranked_indices = np.argsort(probas)[::-1][:top_k]
        print(f"\nTop {top_k} Predictions:")
        for rank, idx in enumerate(ranked_indices, 1):
            cid = model_classes[idx]
            cname = classes_dict.get(cid, f"Class {cid}")
            prob = probas[idx] * 100
            bar_len = int(prob / 2.5)  # 40 chars max
            bar = "█" * bar_len + "░" * (40 - bar_len)
            print(f"  {rank}. {cname:<18} [{bar}] {prob:5.1f}%")

    print("=" * 50 + "\n")
    return pred_class_id, pred_class_name


if __name__ == "__main__":
    args = parse_args()
    predict(args.audio, args.model, args.top_k)
