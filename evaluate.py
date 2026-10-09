#!/usr/bin/env python3
"""
EchoDetect - Model Evaluation Script
Evaluates a trained model checkpoint against specific folds or a subset of UrbanSound8K.
"""

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from config import CLASSES, CLASS_NAMES, MODELS_DIR, REPORTS_DIR, RANDOM_STATE
from echodetect.dataset import load_metadata, get_dataset_subset, get_fold_split
from echodetect.features import MFCCExtractor
from echodetect.models import load_model, evaluate_model
from echodetect.utils import plot_confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="EchoDetect: Evaluate a trained model")
    parser.add_argument("--model", type=str, default=str(MODELS_DIR / "svm_model.joblib"),
                        help="Path to trained model checkpoint (.joblib)")
    parser.add_argument("--test-fold", type=int, default=10,
                        help="UrbanSound8K fold to evaluate on (default: 10)")
    parser.add_argument("--subset", type=int, default=None,
                        help="Evaluate on a small subset (e.g. 100 samples) instead of full fold")
    parser.add_argument("--save-plot", action="store_true", default=True,
                        help="Save confusion matrix heatmap to reports/")
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model checkpoint not found at {model_path}")
        sys.exit(1)

    print(f"[*] Loading model from: {model_path}")
    pipeline, metadata = load_model(model_path)
    print(f"[✓] Model loaded. Stored metadata: {metadata}")

    sr = metadata.get("sample_rate", 22050)
    n_mfcc = metadata.get("n_mfcc", 40)
    include_deltas = metadata.get("include_deltas", True)

    # Load data
    print("\n[*] Loading UrbanSound8K metadata...")
    df = load_metadata()

    if args.subset:
        eval_df = get_dataset_subset(df, total_samples=args.subset, random_state=RANDOM_STATE)
        eval_desc = f"subset of {len(eval_df)} samples"
    else:
        _, eval_df = get_fold_split(df, test_folds=[args.test_fold])
        eval_desc = f"Fold {args.test_fold} ({len(eval_df)} samples)"

    print(f"[✓] Evaluating on: {eval_desc}")

    # Extract features or load from cache
    extractor = MFCCExtractor(sr=sr, n_mfcc=n_mfcc, include_deltas=include_deltas)
    if args.subset:
        X_eval, y_eval = extractor.extract_batch(
            eval_df['file_path'].tolist(),
            labels=eval_df['classID'].tolist(),
            desc="Extracting Evaluation Features"
        )
    else:
        X_eval, y_eval, _ = extractor.get_or_extract_fold(args.test_fold, df)

    # Evaluate
    results = evaluate_model(pipeline, X_eval, y_eval, class_names=CLASS_NAMES)

    print("\n" + "=" * 60)
    print(f"               EVALUATION REPORT: {model_path.name}")
    print("=" * 60)
    print(f"Target Set       : {eval_desc}")
    print(f"Accuracy         : {results['accuracy'] * 100:.2f}%")
    print(f"Macro F1 Score   : {results['f1_macro']:.4f}")
    print(f"Weighted F1 Score: {results['f1_weighted']:.4f}\n")
    print("Detailed Classification Breakdown:")
    print(results['classification_report'])

    if args.save_plot:
        plot_name = f"eval_{model_path.stem}_fold{args.test_fold if not args.subset else 'sub'}.png"
        plot_confusion_matrix(
            results['confusion_matrix'],
            class_names=[CLASS_NAMES[c] for c in sorted(list(set(y_eval)))],
            title=f"Evaluation Confusion Matrix: {model_path.name}",
            save_filename=plot_name
        )


if __name__ == "__main__":
    main()
