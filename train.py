#!/usr/bin/env python3
"""
EchoDetect - Training Script
Trains SVM (RBF + Calibrated Probabilities) and Random Forest (200 trees)
using pre-defined UrbanSound8K folds (Folds 1-8 Train, Fold 9 Val, Fold 10 Test).
Extracts and caches features fold-by-fold on disk to avoid redundant processing.
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure Echodetect package is on sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from config import (
    CLASSES, CLASS_NAMES, SAMPLE_RATE, N_MFCC, N_FFT, HOP_LENGTH,
    INCLUDE_DELTAS, INCLUDE_SPECTRAL, MODELS_DIR, REPORTS_DIR,
    TRAIN_FOLDS, VAL_FOLDS, TEST_FOLDS, RANDOM_STATE,
    SVM_MODEL_PATH, RF_MODEL_PATH, EVAL_SUMMARY_JSON, TEST_METRICS_CSV
)
from echodetect.dataset import load_metadata, get_dataset_subset, get_train_val_test_split
from echodetect.features import MFCCExtractor
from echodetect.models import (
    build_svm_pipeline, build_rf_pipeline, evaluate_model, save_model
)
from echodetect.utils import plot_confusion_matrix, save_evaluation_artifacts


def parse_args():
    parser = argparse.ArgumentParser(description="EchoDetect: Train SVM & Random Forest on UrbanSound8K")
    parser.add_argument("--subset", type=int, default=None,
                        help="Train on a small subset of total samples (e.g. 300 for rapid testing)")
    parser.add_argument("--samples-per-class", type=int, default=None,
                        help="Number of samples to draw per class (e.g. 30 per class)")
    parser.add_argument("--model", type=str, choices=["svm", "rf", "both"], default="both",
                        help="Model to train: 'svm', 'rf', or 'both'")
    parser.add_argument("--tune-svm", action="store_true", default=False,
                        help="Perform lightweight validation search on fold 9 for SVM C parameter")
    parser.add_argument("--save-models", action="store_true", default=True,
                        help="Persist trained models and scalers to saved_models/")
    return parser.parse_args()


def plot_feature_importance(feature_importances, save_filename="rf_feature_importances.png"):
    """Plots top feature importances for Random Forest classifier."""
    if not feature_importances:
        return
    top_items = feature_importances[:15]
    names = [item["feature"] for item in reversed(top_items)]
    vals = [item["importance"] for item in reversed(top_items)]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor('#08090C')
    ax.set_facecolor('#15171D')

    clean_names = [n.replace('_', ' ') for n in names]
    bars = ax.barh(clean_names, vals, color='#D7FF64', edgecolor='#B4E83A', height=0.6, alpha=0.95)
    ax.set_title("Random Forest Top 15 Feature Importances", color='#F5F5F7', fontsize=12, fontweight='700', pad=12, fontfamily='sans-serif')
    ax.set_xlabel("Gini Importance (Mean Decrease in Impurity)", color='#A5A9B5', fontsize=9, labelpad=8, fontfamily='sans-serif')
    ax.tick_params(colors='#A5A9B5', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#252936')
    ax.grid(True, axis='x', color='#1F232E', linestyle='--', linewidth=0.6, alpha=0.7)
    plt.tight_layout()

    out_path = REPORTS_DIR / save_filename
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[✓] Feature importance plot saved to: {out_path}")


def main():
    args = parse_args()

    print("=" * 75)
    print("                ECHODETECT PRODUCTION ML TRAINING PIPELINE")
    print("=" * 75)
    print(f"[*] Configuration:")
    print(f"    - Sample Rate      : {SAMPLE_RATE} Hz")
    print(f"    - N_MFCC           : {N_MFCC}")
    print(f"    - Derivatives      : Deltas & Delta-Deltas (order 1 & 2)")
    print(f"    - Spectral Features: Centroid, Bandwidth, Rolloff, ZCR, Chroma STFT")
    print(f"    - Predefined Folds : Train: {TRAIN_FOLDS} | Val: {VAL_FOLDS} | Test: {TEST_FOLDS}")
    print(f"    - Selected Model(s): {args.model.upper()}")

    # 1. Load Metadata
    print("\n[Step 1/5] Loading UrbanSound8K metadata...")
    df = load_metadata()
    print(f"[✓] Total valid audio records in dataset: {len(df)}")

    extractor = MFCCExtractor(
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        include_deltas=INCLUDE_DELTAS,
        include_spectral=INCLUDE_SPECTRAL
    )
    feature_names = extractor.feature_names
    print(f"[✓] Feature dimension: {extractor.feature_dim} features per audio file")

    # 2. Dataset Splitting
    if args.subset or args.samples_per_class:
        print(f"\n[Step 2/5] Extracting custom subset for rapid experimentation...")
        subset_df = get_dataset_subset(
            df,
            samples_per_class=args.samples_per_class,
            total_samples=args.subset,
            random_state=RANDOM_STATE
        )
        print(f"[✓] Sampled {len(subset_df)} records across classes.")
        train_df, test_df = train_test_split(
            subset_df, test_size=0.2, random_state=RANDOM_STATE, stratify=subset_df['classID']
        )
        val_df = test_df  # Use test split for validation evaluation in subset mode

        print(f"    - Train set: {len(train_df)} samples")
        print(f"    - Test set : {len(test_df)} samples")

        # Extract features for subset
        print("\n[Step 3/5] Extracting features for subset...")
        X_train, y_train = extractor.extract_batch(
            train_df['file_path'].tolist(), train_df['classID'].tolist(), desc="Extracting Train"
        )
        X_val, y_val = extractor.extract_batch(
            val_df['file_path'].tolist(), val_df['classID'].tolist(), desc="Extracting Val"
        )
        X_test, y_test = extractor.extract_batch(
            test_df['file_path'].tolist(), test_df['classID'].tolist(), desc="Extracting Test"
        )
    else:
        # Full Fold-based split (Folds 1-8 Train, Fold 9 Val, Fold 10 Test)
        print("\n[Step 2/5] Preparing Predefined UrbanSound8K Folds (No Data Leakage)...")
        print(f"    - Training Folds   : {TRAIN_FOLDS} (folds 1 through 8)")
        print(f"    - Validation Fold  : {VAL_FOLDS} (fold 9)")
        print(f"    - Held-out Test Fold: {TEST_FOLDS} (fold 10)")

        # Load / Extract features fold by fold with disk caching
        print("\n[Step 3/5] Loading / caching features fold-by-fold from disk...")
        train_X_list, train_y_list = [], []
        for f in TRAIN_FOLDS:
            X_f, y_f, _ = extractor.get_or_extract_fold(f, df)
            train_X_list.append(X_f)
            train_y_list.append(y_f)

        X_train = np.vstack(train_X_list)
        y_train = np.concatenate(train_y_list)

        # Validation fold 9
        X_val, y_val, _ = extractor.get_or_extract_fold(VAL_FOLDS[0], df)

        # Held-out test fold 10
        X_test, y_test, _ = extractor.get_or_extract_fold(TEST_FOLDS[0], df)

    print(f"\n[✓] Feature matrices prepared successfully:")
    print(f"    - Train set shape     : {X_train.shape} ({len(y_train)} samples)")
    print(f"    - Validation set shape: {X_val.shape} ({len(y_val)} samples)")
    print(f"    - Test set shape      : {X_test.shape} ({len(y_test)} samples)")

    # 4. Model Training & Hyperparameter Tuning
    print("\n[Step 4/5] Training Classifiers...")
    results = {}

    # Support Vector Machine
    if args.model in ["svm", "both"]:
        print("\n" + "=" * 35 + " SUPPORT VECTOR MACHINE " + "=" * 35)
        best_c = 10.0
        if args.tune_svm:
            print("[*] Tuning SVM parameter C on validation fold 9...")
            candidate_c = [1.0, 5.0, 10.0, 20.0]
            best_val_acc = -1
            for c_val in candidate_c:
                temp_pipe = build_svm_pipeline({"C": c_val})
                temp_pipe.fit(X_train, y_train)
                val_acc = temp_pipe.score(X_val, y_val)
                print(f"    - C={c_val:<4} -> Validation Accuracy: {val_acc * 100:.2f}%")
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_c = c_val
            print(f"[✓] Selected best SVM C = {best_c}")

        svm_pipeline = build_svm_pipeline({"C": best_c})
        t0 = time.time()
        print(f"[*] Fitting SVM on training data ({len(X_train)} samples)...")
        svm_pipeline.fit(X_train, y_train)
        svm_time = time.time() - t0
        print(f"[✓] SVM training completed in {svm_time:.2f}s")

        # Evaluate on strictly held-out test fold
        svm_eval = evaluate_model(
            svm_pipeline, X_test, y_test,
            class_names=CLASS_NAMES, feature_names=feature_names
        )
        results["svm"] = {
            "name": "Support Vector Machine (RBF)",
            "pipeline": svm_pipeline,
            "fit_time": round(svm_time, 2),
            "metrics": svm_eval
        }

        print(f"\n--- Held-Out Test Fold Results (SVM) ---")
        print(f"Accuracy         : {svm_eval['accuracy'] * 100:.2f}%")
        print(f"Macro Precision  : {svm_eval['precision_macro']:.4f}")
        print(f"Macro Recall     : {svm_eval['recall_macro']:.4f}")
        print(f"Macro F1 Score   : {svm_eval['f1_macro']:.4f}")
        print(f"Weighted F1 Score: {svm_eval['f1_weighted']:.4f}\n")
        print("Classification Report:")
        print(svm_eval['classification_report_str'])

        if args.save_models:
            save_model(svm_pipeline, SVM_MODEL_PATH, metadata={
                "model_type": "svm",
                "accuracy": svm_eval['accuracy'],
                "f1_macro": svm_eval['f1_macro'],
                "feature_dim": extractor.feature_dim,
                "n_mfcc": N_MFCC,
                "sample_rate": SAMPLE_RATE,
                "classes": CLASSES
            })

        plot_confusion_matrix(
            svm_eval['confusion_matrix'],
            class_names=[CLASS_NAMES[c] for c in sorted(list(set(y_test)))],
            title="SVM Test Fold 10 Confusion Matrix",
            save_filename="svm_confusion_matrix.png"
        )

    # Random Forest Classifier
    if args.model in ["rf", "both"]:
        print("\n" + "=" * 35 + " RANDOM FOREST (200 TREES) " + "=" * 35)
        rf_pipeline = build_rf_pipeline()
        t0 = time.time()
        print(f"[*] Fitting Random Forest on training data ({len(X_train)} samples)...")
        rf_pipeline.fit(X_train, y_train)
        rf_time = time.time() - t0
        print(f"[✓] Random Forest training completed in {rf_time:.2f}s")

        # Evaluate on strictly held-out test fold
        rf_eval = evaluate_model(
            rf_pipeline, X_test, y_test,
            class_names=CLASS_NAMES, feature_names=feature_names
        )
        results["rf"] = {
            "name": "Random Forest (200 Trees)",
            "pipeline": rf_pipeline,
            "fit_time": round(rf_time, 2),
            "metrics": rf_eval
        }

        print(f"\n--- Held-Out Test Fold Results (Random Forest) ---")
        print(f"Accuracy         : {rf_eval['accuracy'] * 100:.2f}%")
        print(f"Macro Precision  : {rf_eval['precision_macro']:.4f}")
        print(f"Macro Recall     : {rf_eval['recall_macro']:.4f}")
        print(f"Macro F1 Score   : {rf_eval['f1_macro']:.4f}")
        print(f"Weighted F1 Score: {rf_eval['f1_weighted']:.4f}\n")
        print("Classification Report:")
        print(rf_eval['classification_report_str'])

        if args.save_models:
            save_model(rf_pipeline, RF_MODEL_PATH, metadata={
                "model_type": "rf",
                "accuracy": rf_eval['accuracy'],
                "f1_macro": rf_eval['f1_macro'],
                "feature_dim": extractor.feature_dim,
                "n_mfcc": N_MFCC,
                "sample_rate": SAMPLE_RATE,
                "classes": CLASSES
            })

        plot_confusion_matrix(
            rf_eval['confusion_matrix'],
            class_names=[CLASS_NAMES[c] for c in sorted(list(set(y_test)))],
            title="Random Forest Test Fold 10 Confusion Matrix",
            save_filename="rf_confusion_matrix.png"
        )

        plot_feature_importance(rf_eval['feature_importances'])

    # Step 5: Save Summary Artifacts
    print("\n[Step 5/5] Saving Evaluation Summary & Performance Artifacts...")
    summary_data = {
        "dataset": "UrbanSound8K",
        "total_samples": len(df),
        "feature_dim": extractor.feature_dim,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "evaluation_fold": "Fold 10 (Strictly Held Out)",
        "classes": CLASSES,
        "class_names": CLASS_NAMES
    }
    for k in ["svm", "rf"]:
        if k in results:
            summary_data[k] = results[k]

    save_evaluation_artifacts(summary_data)

    # Print final comparison table
    print("\n" + "=" * 75)
    print("                     FINAL MODEL COMPARISON (TEST FOLD 10)")
    print("=" * 75)
    print(f"{'Classifier':<30} | {'Accuracy':<10} | {'Macro F1':<10} | {'Weighted F1':<12} | {'Train Time':<10}")
    print("-" * 75)
    for k, v in results.items():
        m = v['metrics']
        print(f"{v['name']:<30} | {m['accuracy']*100:6.2f}%    | {m['f1_macro']:<10.4f} | {m['f1_weighted']:<12.4f} | {v['fit_time']}s")
    print("=" * 75)
    print("\n[✓] Training pipeline execution completed successfully!")


if __name__ == "__main__":
    main()
