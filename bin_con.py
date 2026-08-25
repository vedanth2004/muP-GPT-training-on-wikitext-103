import os
from tqdm import tqdm
import numpy as np
import tiktoken

# Define directory structures matching your setup
dataset_dir = 'wikitext-103-raw-v1/wikitext-103-raw'
data_dir = os.path.join('data', dataset_dir)
os.makedirs(data_dir, exist_ok=True)

# Map target outputs to your local raw file patterns
files = {
    'wiki.train.bin': os.path.join(dataset_dir, 'wiki.train.raw'),
    'wiki.valid.bin': os.path.join(dataset_dir, 'wiki.valid.raw'),
}

# Initialize the official GPT-2 Tiktoken encoder
enc = tiktoken.get_encoding("gpt2")

print("Starting tokenization and conversion to binary format...")

for bin_filename, raw_filepath in files.items():
    if not os.path.exists(raw_filepath):
        print(f"Error: Could not find raw file at {raw_filepath}. Skipping.")
        continue
        
    print(f"Processing {raw_filepath} -> {bin_filename}...")
    
    # Read the full raw text content
    with open(raw_filepath, 'r', encoding='utf-8') as f:
        data = f.read()
        
    # Encode text to GPT-2 token integer IDs and append End-Of-Text token
    all_tokens = enc.encode_ordinary(data)
    all_tokens.append(enc.eot_token)
    
    # Convert token arrays to uint16 NumPy configurations
    token_arr = np.array(all_tokens, dtype=np.uint16)
    
    # Write structural binary directly to disk 
    out_path = os.path.join(data_dir, bin_filename)
    with open(out_path, 'wb') as f:
        f.write(token_arr.tobytes())
        
    print(f"Saved {len(token_arr):,} tokens to {out_path}")

print("\nData preparation complete! You can now launch your train.py script.")
