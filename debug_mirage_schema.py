from datasets import load_dataset
import json

try:
    print("Loading nthakur/mirage-bench (dev split)...")
    # Config 'en' for English
    ds = load_dataset("nthakur/mirage-bench", "en")
    print("Splits:", list(ds.keys()))
    # Check if 'corpus' is a split
    if 'corpus' in ds:
        print("Corpus found!")
    else:
        print("No corpus split found.")


except Exception as e:
    print(f"Error: {e}")
