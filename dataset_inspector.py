#!/usr/bin/env python3
"""
EchoDetect - UrbanSound8K Dataset Inspector
Analyzes the metadata and audio characteristics of the UrbanSound8K dataset.
"""

import os
import sys
import pandas as pd
import soundfile as sf
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "UrbanSound8K"
METADATA_PATH = DATASET_DIR / "metadata" / "UrbanSound8K.csv"
AUDIO_DIR = DATASET_DIR / "audio"

def inspect_dataset():
    print("=" * 60)
    print("        ECHODETECT: URBANSOUND8K DATASET INSPECTION")
    print("=" * 60)
    
    if not METADATA_PATH.exists():
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        sys.exit(1)
        
    print(f"\n[1] Loading metadata from: {METADATA_PATH}")
    df = pd.read_csv(METADATA_PATH)
    
    print(f"    - Total records: {len(df)}")
    print(f"    - Columns: {list(df.columns)}")
    missing = df.isnull().sum()
    print(f"    - Missing values:\n{missing.to_string()}")
    
    # Class distribution
    print("\n[2] Class Distribution:")
    class_counts = df.groupby(['classID', 'class']).size().reset_index(name='count')
    for _, row in class_counts.iterrows():
        print(f"    - Class {row['classID']:2d}: {row['class']:<20} ({row['count']} samples)")
        
    # Fold distribution
    print("\n[3] Fold Distribution:")
    fold_counts = df.groupby('fold').size()
    for fold, count in fold_counts.items():
        print(f"    - Fold {fold:2d}: {count} audio files")
        
    # Salience distribution
    print("\n[4] Salience Distribution:")
    salience_counts = df['salience'].value_counts()
    salience_labels = {1: "foreground", 2: "background"}
    for sal, count in salience_counts.items():
        print(f"    - Salience {sal} ({salience_labels.get(sal, 'unknown')}): {count} samples")
        
    # Verify file existence on disk
    print("\n[5] Verifying Audio File Existence on Disk:")
    missing_files = []
    found_count = 0
    
    for idx, row in df.iterrows():
        fold_dir = AUDIO_DIR / f"fold{row['fold']}"
        audio_file = fold_dir / row['slice_file_name']
        if audio_file.exists():
            found_count += 1
        else:
            missing_files.append(str(audio_file))
            if len(missing_files) <= 5:
                print(f"    [!] Missing: {audio_file}")
                
    print(f"    - Verified {found_count} / {len(df)} files exist on disk.")
    if missing_files:
        print(f"    - Total missing files: {len(missing_files)}")
    else:
        print("    - All audio files are present and matched with metadata!")

    # Inspect audio properties for a sample of files
    print("\n[6] Sampling Audio File Properties (First 150 files across folds):")
    sample_df = df.sample(min(150, len(df)), random_state=42)
    sample_rates = Counter()
    channels_count = Counter()
    durations = []
    subtypes = Counter()
    
    for _, row in sample_df.iterrows():
        audio_path = AUDIO_DIR / f"fold{row['fold']}" / row['slice_file_name']
        try:
            info = sf.info(str(audio_path))
            sample_rates[info.samplerate] += 1
            channels_count[info.channels] += 1
            durations.append(info.duration)
            subtypes[info.subtype] += 1
        except Exception as e:
            print(f"    [!] Error reading {audio_path.name}: {e}")
            
    print(f"    - Sample Rates: {dict(sample_rates)}")
    print(f"    - Channels (1=mono, 2=stereo): {dict(channels_count)}")
    print(f"    - Audio Subtypes / Bit Formats: {dict(subtypes)}")
    if durations:
        print(f"    - Duration range: {min(durations):.2f}s to {max(durations):.2f}s (mean: {sum(durations)/len(durations):.2f}s)")
        
    print("\n" + "=" * 60)
    print("                 INSPECTION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    inspect_dataset()
