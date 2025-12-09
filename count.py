import os

# Path to your dataset
dataset_path = "final_dataset_v3"

# Initialize dictionary to store counts
counts = {}

# Iterate over train, test, val
for split in ["train", "test", "val"]:
    split_path = os.path.join(dataset_path, split)
    counts[split] = {}
    
    # Iterate over classes real and fake
    for cls in ["real", "fake"]:
        class_path = os.path.join(split_path, cls)
        if os.path.exists(class_path):
            # Count files (images) in the folder
            counts[split][cls] = len([
                f for f in os.listdir(class_path)
                if os.path.isfile(os.path.join(class_path, f))
            ])
        else:
            counts[split][cls] = 0

# Print counts
for split, cls_counts in counts.items():
    print(f"{split}:")
    for cls, count in cls_counts.items():
        print(f"  {cls}: {count}")
