#!/usr/bin/env python3
"""
EchoDetect - Pipeline Verification Script
Tests the full end-to-end ML workflow on a small subset of UrbanSound8K:
1. Dataset subset extraction
2. MFCC feature extraction (mean, std, deltas)
3. Model training (SVM and Random Forest)
4. Evaluation and confusion matrix plotting
5. Model saving and loading
6. Single-file audio inference
"""

import sys
import time
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from config import (
    CLASSES, CLASS_NAMES, SAMPLE_RATE, N_MFCC, N_FFT, HOP_LENGTH,
    MODELS_DIR, REPORTS_DIR, RANDOM_STATE
)
from echodetect.dataset import load_metadata, get_dataset_subset
from echodetect.features import MFCCExtractor, extract_features_from_file
from echodetect.models import (
    build_svm_pipeline, build_rf_pipeline, evaluate_model, save_model, load_model
)
from echodetect.utils import plot_confusion_matrix


def run_pipeline_test(samples_per_class: int = 10):
    print("=" * 70)
    print("         ECHODETECT PIPELINE TEST (SMALL SUBSET)")
    print("=" * 70)
    total_samples = samples_per_class * len(CLASSES)
    print(f"[*] Target: {samples_per_class} samples/class = {total_samples} samples across 10 classes\n")

    # Step 1: Load metadata and extract subset
    print("[1/6] Loading metadata and generating small balanced subset...")
    df = load_metadata()
    assert len(df) > 0, "Dataset metadata is empty!"
    
    subset_df = get_dataset_subset(df, samples_per_class=samples_per_class, random_state=RANDOM_STATE)
    print(f"    - Loaded subset: {len(subset_df)} samples")
    assert len(subset_df) == total_samples, f"Expected {total_samples} samples, got {len(subset_df)}"
    print("    [✓] Step 1 Passed: Balanced subset generated.")

    # Step 2: Feature extraction test
    print("\n[2/6] Testing MFCC feature extraction...")
    extractor = MFCCExtractor(sr=SAMPLE_RATE, n_mfcc=N_MFCC, include_deltas=True)
    expected_dim = extractor.feature_dim
    print(f"    - Feature dimension: {expected_dim}")

    # Test single file extraction
    test_audio = subset_df.iloc[0]['file_path']
    single_feat = extractor.extract_single(test_audio)
    assert single_feat.shape == (expected_dim,), f"Single feature shape mismatch: {single_feat.shape}"
    assert not np.isnan(single_feat).any(), "Single feature contains NaNs!"
    print(f"    - Single audio ({test_audio.name}) extracted: shape={single_feat.shape}, non-zero={np.count_nonzero(single_feat)}")

    # Split train/test
    train_df, test_df = train_test_split(
        subset_df,
        test_size=0.3,
        random_state=RANDOM_STATE,
        stratify=subset_df['classID']
    )
    print(f"    - Train split: {len(train_df)} samples, Test split: {len(test_df)} samples")

    # Extract batch
    t0 = time.time()
    X_train, y_train = extractor.extract_batch(
        train_df['file_path'].tolist(),
        labels=train_df['classID'].tolist(),
        desc="Extracting Train Features"
    )
    X_test, y_test = extractor.extract_batch(
        test_df['file_path'].tolist(),
        labels=test_df['classID'].tolist(),
        desc="Extracting Test Features"
    )
    extract_time = time.time() - t0
    print(f"    - Batch extraction completed in {extract_time:.2f}s ({extract_time/len(subset_df):.3f}s/file)")
    assert X_train.shape == (len(train_df), expected_dim)
    assert X_test.shape == (len(test_df), expected_dim)
    print("    [✓] Step 2 Passed: Feature extraction validated.")

    # Step 3: Train SVM & Random Forest
    print("\n[3/6] Training SVM and Random Forest models...")
    svm_pipe = build_svm_pipeline()
    rf_pipe = build_rf_pipeline()

    t_svm = time.time()
    svm_pipe.fit(X_train, y_train)
    svm_time = time.time() - t_svm
    print(f"    - SVM fitted in {svm_time:.2f}s")

    t_rf = time.time()
    rf_pipe.fit(X_train, y_train)
    rf_time = time.time() - t_rf
    print(f"    - Random Forest fitted in {rf_time:.2f}s")
    print("    [✓] Step 3 Passed: Models fitted successfully.")

    # Step 4: Evaluate Models
    print("\n[4/6] Evaluating models on test split...")
    svm_eval = evaluate_model(svm_pipe, X_test, y_test, class_names=CLASS_NAMES)
    rf_eval = evaluate_model(rf_pipe, X_test, y_test, class_names=CLASS_NAMES)

    print(f"    - SVM Accuracy: {svm_eval['accuracy']*100:.1f}%, Macro F1: {svm_eval['f1_macro']:.4f}")
    print(f"    - RF  Accuracy: {rf_eval['accuracy']*100:.1f}%, Macro F1: {rf_eval['f1_macro']:.4f}")
    
    # Generate confusion matrix plots
    svm_cm_path = plot_confusion_matrix(svm_eval['confusion_matrix'], title="SVM Test Confusion Matrix", save_filename="test_svm_cm.png")
    rf_cm_path = plot_confusion_matrix(rf_eval['confusion_matrix'], title="RF Test Confusion Matrix", save_filename="test_rf_cm.png")
    assert svm_cm_path.exists(), "SVM CM plot not generated"
    assert rf_cm_path.exists(), "RF CM plot not generated"
    print("    [✓] Step 4 Passed: Model evaluation and plots verified.")

    # Step 5: Model Serialization & Deserialization
    print("\n[5/6] Testing model saving and loading...")
    test_svm_path = MODELS_DIR / "test_svm_pipeline.joblib"
    save_model(svm_pipe, test_svm_path, metadata={"accuracy": svm_eval['accuracy']})
    assert test_svm_path.exists(), "Saved model file does not exist!"

    loaded_pipe, meta = load_model(test_svm_path)
    loaded_preds = loaded_pipe.predict(X_test[:5])
    orig_preds = svm_pipe.predict(X_test[:5])
    assert np.array_equal(loaded_preds, orig_preds), "Loaded model predictions do not match original model!"
    print(f"    - Saved and loaded model matches original predictions perfectly.")
    print("    [✓] Step 5 Passed: Model serialization verified.")

    # Step 6: Audio File Inference Test
    print("\n[6/6] Testing single audio file inference...")
    sample_file = test_df.iloc[0]['file_path']
    true_class = CLASSES[test_df.iloc[0]['classID']]
    sample_feat = extract_features_from_file(sample_file, sr=SAMPLE_RATE, n_mfcc=N_MFCC, include_deltas=True)
    pred_class_id = loaded_pipe.predict(sample_feat.reshape(1, -1))[0]
    pred_class = CLASSES[pred_class_id]
    probas = loaded_pipe.predict_proba(sample_feat.reshape(1, -1))[0]
    
    print(f"    - Sample file: {sample_file.name}")
    print(f"    - Ground Truth: {true_class}")
    print(f"    - Predicted   : {pred_class}")
    print(f"    - Class Probabilities: {np.round(probas, 3)}")
    print("    [✓] Step 6 Passed: End-to-end audio inference verified.")

    print("\n" + "=" * 70)
    print("              ALL PIPELINE TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = run_pipeline_test(samples_per_class=10)
    sys.exit(0 if success else 1)
