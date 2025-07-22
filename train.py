# This file is the intitial training script for training a model on characters

# --NOTES--
# Re-check the blank index and ensure it's used consistently in encoding and character mapping
# The num_classes includes the blank character. If our characters are 0-9, a-z (36 total),
# then num_classes = 37. The indices should be 1-36 for characters and 0 for blank, OR
# 0-35 for characters and 36 for blank. Keras's ctc_batch_cost expects blank_index = num_classes - 1.
# So, our character indices should be 0 to num_classes - 2.
# Let's adjust the char_to_int and int_to_char mappings to reflect this.



import numpy as np
from PIL import Image
import pandas as pd
# Example: create dummy DataFrames with 'image_path' and 'text' columns
# You should replace this with your actual data loading logic
train_df = pd.DataFrame({
    'image_path': ['dummy_path1.png', 'dummy_path2.png'],
    'text': ['5', '0']
})
val_df = pd.DataFrame({
    'image_path': ['dummy_path3.png'],
    'text': ['4']
})
# Build character set from train_df and val_df (assuming single-label classification)
all_labels = np.unique(np.concatenate([train_df['text'].values, val_df['text'].values]))
all_chars = [str(label) for label in all_labels]  # Convert to string if needed
# Assign indices 0 to len(all_chars) - 1 to the characters
char_to_int = {char: i for i, char in enumerate(all_chars)}
int_to_char = {i: char for i, char in enumerate(all_chars)}
# Assign the blank index
blank_index = len(all_chars)  # This will be num_classes - 1
char_to_int[' '] = blank_index
int_to_char[blank_index] = ' '
num_classes = len(all_chars) + 1  # Total number of classes including blank
print(f"Updated Number of unique characters (including blank): {num_classes}")
print(f"Blank index: {blank_index}")
print(f"Char to int mapping sample: {list(char_to_int.items())[:5]}...")
print(f"Int to char mapping sample: {list(int_to_char.items())[:5]}...")
# Define MAX_TEXT_LEN as the maximum length of the text labels in your dataset
MAX_TEXT_LEN = 1  # For single digit classification; set higher if your labels are longer
# Re-define the encoding function to use the updated blank index for padding
def encode_label(text, char_map, max_len=MAX_TEXT_LEN, blank_idx=blank_index):
    encoded = [char_map.get(char, blank_idx) for char in text] # Use .get with default blank_idx for unknown chars
    # Pad labels with the blank index to a fixed length
    if len(encoded) < max_len:
        encoded += [blank_idx] * (max_len - len(encoded))
    else:
        encoded = encoded[:max_len] # Truncate if longer than max_len
    return np.array(encoded)
