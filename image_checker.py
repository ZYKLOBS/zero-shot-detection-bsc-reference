from PIL import Image
import numpy as np

# Load image
image = Image.open("paper_data/real/NLL_highres.jpeg")
img_array = np.array(image)

print("Image shape:", img_array.shape)  # (H, W, 3)

# Function to extract and print a 2x2 block
def print_block(block_index):
    row = 0
    col = block_index * 2
    block = img_array[row:row+2, col:col+2]  # shape (2, 2, 3)
    print(f"\nBlock {block_index + 1} (rows {row}-{row+1}, cols {col}-{col+1}):")
    print(block)

# Print the first 3 2×2 blocks
for i in range(3):
    print_block(i)
