import os
import matplotlib.pyplot as plt
from datasets import load_dataset
import pandas as pd
import numpy as np

def inspect_dataset():
    print("Loading ai4bharat/MSMARCO-XI dataset (Hindi config)...")
    try:
        dataset = load_dataset("ai4bharat/MSMARCO-XI", data_files={"train": "train/hintrain.parquet"}, split="train", streaming=True)
        print("Loaded Hindi subset via 'train/hintrain.parquet' for inspection.")
    except Exception as e:
        print(f"Failed to load Hindi subset: {e}")
        try:
            dataset = load_dataset("ai4bharat/MSMARCO-XI", split="train", streaming=True)
            print("Loaded default subset.")
        except Exception as e2:
            print(f"Failed to load default subset: {e2}")
            return

    # Sample records
    max_queries = 500
    print(f"Sampling passages from the first {max_queries} queries for profiling...")
    
    passage_records = []
    query_lengths = []
    
    for i, item in enumerate(dataset):
        if i >= max_queries:
            break
        
        query = item.get("query", "")
        query_lengths.append(len(query))
        
        passages_dict = item.get("passages", {})
        translated_passages = passages_dict.get("Translated_passages", [])
        is_selected = passages_dict.get("is_selected", [])
        
        for idx, text in enumerate(translated_passages):
            selected = is_selected[idx] if idx < len(is_selected) else 0
            passage_records.append({
                "query_idx": i,
                "text": text,
                "selected": selected,
                "length_chars": len(text),
                "length_words": len(text.split())
            })

    if not passage_records:
        print("No passage records found or loaded.")
        return

    df = pd.DataFrame(passage_records)
    print(f"\nProfiled {len(df)} passages from {max_queries} queries.")
    
    print("\n--- Query Stats ---")
    print(f"Mean query character length: {np.mean(query_lengths):.2f}")
    
    print("\n--- Passage Length (characters) Stats ---")
    print(df['length_chars'].describe())
    
    print("\n--- Passage Word Count Stats ---")
    print(df['length_words'].describe())
    
    print("\n--- Relevance Stats ---")
    print(df['selected'].value_counts(normalize=True))

    # Plot distribution
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(df['length_chars'], bins=50, color='blue', alpha=0.7, edgecolor='black')
    plt.title('Passage Length (Chars) Distribution')
    plt.xlabel('Length in Characters')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    plt.hist(df['length_words'], bins=50, color='green', alpha=0.7, edgecolor='black')
    plt.title('Passage Word Count Distribution')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    
    os.makedirs('data', exist_ok=True)
    plot_path = 'data/passage_length_distribution.png'
    plt.savefig(plot_path)
    print(f"\nSaved passage length distribution plot to {plot_path}")

if __name__ == "__main__":
    inspect_dataset()
