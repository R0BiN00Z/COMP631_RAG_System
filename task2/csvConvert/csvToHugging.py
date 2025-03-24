import pandas as pd
from datasets import Dataset, load_dataset

def csv_to_huggingface(csv_path="xiaohongshu_data.csv", output_path="huggingface_corpus.csv"):
    # Load CSV
    df = pd.read_csv(csv_path)

    # Add 'title' column if missing
    if 'title' not in df.columns:
        df['title'] = ''

    # Rename 'content' to 'text'
    df = df.rename(columns={'content': 'text'})

    # Insert 'id' column
    df.insert(0, 'id', [f"doc{i}" for i in range(1, len(df)+1)])

    # Reorder and save
    df[['id', 'title', 'text']].to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ Saved processed data to {output_path}")

    # Load as Hugging Face dataset
    dataset = load_dataset('csv', data_files=output_path)
    corpus = dataset['train']
    print(f"✅ Loaded corpus with {len(corpus)} samples")

    # Print a sample for verification
    print("📌 Sample:", corpus[0])
    return corpus

if __name__ == "__main__":
    corpus = csv_to_huggingface("xiaohongshu_data.csv")
