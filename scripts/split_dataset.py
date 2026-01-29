import os
import shutil
import random
import glob

# --- Configuration ---
# Project Root (One level up from scripts/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# List of all data sources to merge (Relative to Project Root)
# Add "data" if you have labeled data there too
SOURCE_DIRS = [
    "output" 
]

CLASSES_FILE = os.path.join(PROJECT_ROOT, "classes.txt")

# Destination Directory for the final split dataset
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "dataset")

SPLIT_RATIO = 0.8  # 80% Train, 20% Validation

def load_classes():
    """Load class names from the classes.txt file."""
    if os.path.exists(CLASSES_FILE):
        with open(CLASSES_FILE, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        return lines
    return ["object"]

def main():
    print(f"Starting Dataset Preparation...")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Sources: {SOURCE_DIRS}")
    print(f"Destination: {OUTPUT_DIR}")
    
    # 1. Prepare Output Directories
    if os.path.exists(OUTPUT_DIR):
        print(f"Cleaning existing output directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
        
    sub_dirs = [
        "images/train", "images/val",
        "labels/train", "labels/val"
    ]
    for d in sub_dirs:
        os.makedirs(os.path.join(OUTPUT_DIR, d), exist_ok=True)

    # 2. Pairs Discovery from ALL sources
    valid_pairs = [] # List of (img_path, lbl_path, new_filename_prefix)
    
    total_found_images = 0
    
    for source_name in SOURCE_DIRS:
        source_path = os.path.join(PROJECT_ROOT, source_name)
        img_dir = os.path.join(source_path, "images")
        lbl_dir = os.path.join(source_path, "labels")
        
        if not os.path.exists(img_dir):
            print(f"Skipping {source_name} (Not found: {img_dir})")
            continue
            
        print(f"Scanning source: {source_name} ...")
        
        image_exts = ['.jpg', '.jpeg', '.png', '.bmp']
        source_images = []
        for ext in image_exts:
            source_images.extend(glob.glob(os.path.join(img_dir, f"*{ext}")))
            source_images.extend(glob.glob(os.path.join(img_dir, f"*{ext.upper()}")))
            
        print(f"  Found {len(source_images)} images in {source_name}")
        total_found_images += len(source_images)
        
        for img_path in source_images:
            filename = os.path.basename(img_path)
            base_name = os.path.splitext(filename)[0]
            
            # Check for corresponding label
            label_path = os.path.join(lbl_dir, base_name + ".txt")
            if os.path.exists(label_path):
                # We add the source_name as prefix to avoid filename collisions
                # e.g. "data_main_image001.jpg"
                valid_pairs.append({
                    "img_src": img_path,
                    "lbl_src": label_path,
                    "prefix": source_name
                })
            else:
                pass # Silent skip
                
    print(f"Total valid image-label pairs found: {len(valid_pairs)}")

    if not valid_pairs:
        print("Error: No data found to split.")
        return

    # 3. Shuffle and Split
    random.shuffle(valid_pairs)
    split_index = int(len(valid_pairs) * SPLIT_RATIO)
    
    train_set = valid_pairs[:split_index]
    val_set = valid_pairs[split_index:]
    
    print(f"Splitting: {len(train_set)} Train, {len(val_set)} Validation")

    # 4. Copy Files
    def copy_set(dataset, split_name):
        print(f"Copying {split_name} data...")
        count = 0
        for item in dataset:
            img_src = item["img_src"]
            lbl_src = item["lbl_src"]
            prefix = item["prefix"]
            
            original_filename = os.path.basename(img_src)
            original_lbl_filename = os.path.basename(lbl_src)
            
            # Construct new unique filename
            new_filename = f"{prefix}_{original_filename}"
            new_lbl_filename = f"{prefix}_{original_lbl_filename}"
            
            # Copy Image
            shutil.copy2(img_src, os.path.join(OUTPUT_DIR, "images", split_name, new_filename))
            # Copy Label
            shutil.copy2(lbl_src, os.path.join(OUTPUT_DIR, "labels", split_name, new_lbl_filename))
            count += 1
        return count

    copy_set(train_set, "train")
    copy_set(val_set, "val")

    # 5. Generate data.yaml
    class_names = load_classes()
    
    # Absolute path for safety in training
    train_path = os.path.join(OUTPUT_DIR, "images", "train")
    val_path = os.path.join(OUTPUT_DIR, "images", "val")
    
    yaml_content = []
    yaml_content.append(f"path: {OUTPUT_DIR}")
    yaml_content.append(f"train: images/train")
    yaml_content.append(f"val: images/val")
    yaml_content.append("")
    yaml_content.append("names:")
    for idx, name in enumerate(class_names):
        yaml_content.append(f"  {idx}: {name}")
    
    yaml_out_path = os.path.join(OUTPUT_DIR, "data.yaml")
    with open(yaml_out_path, "w") as f:
        f.write("\n".join(yaml_content))

    print("------------------------------------------------")
    print("Dataset Preparation Complete!")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Configuration File: {yaml_out_path}")
    print("You can now use this data.yaml for YOLO-Seg training.")

if __name__ == "__main__":
    main()
