from datasets import load_dataset
import json

try:
    print("Loading dgslibisey/MuSiQue (default, validation)...")
    ds = load_dataset("dgslibisey/MuSiQue", "default", split="validation")
    
    print("\nDataset Info:")
    print(ds)
    
    print("\nFirst Example Schema:")
    example = ds[0]
    # Prune paragraphs list for display to avoid huge output
    if 'paragraphs' in example:
        full_paragraphs = example['paragraphs']
        print(f"Total paragraphs: {len(full_paragraphs)}")
        example['paragraphs'] = full_paragraphs[:3] # Show first 3 only
        
    print(json.dumps(example, indent=2, default=str))

    # Verify extraction logic
    gold_passages = [p for p in full_paragraphs if p.get('is_supporting')]
    print(f"\nGold Passages found: {len(gold_passages)}")
    if gold_passages:
        print("Sample Gold:", gold_passages[0])

except Exception as e:
    print(f"Error: {e}")
