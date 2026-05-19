import os
import glob
import shutil

def quarantine_empty_labels(base_dir):
    labels_dir = os.path.join(base_dir, "labels")
    images_dir = os.path.join(base_dir, "images")
    quarantine_dir = os.path.join(base_dir, "background_quarantine")
    
    os.makedirs(quarantine_dir, exist_ok=True)
    q_images_dir = os.path.join(quarantine_dir, "images")
    q_labels_dir = os.path.join(quarantine_dir, "labels")
    os.makedirs(q_images_dir, exist_ok=True)
    os.makedirs(q_labels_dir, exist_ok=True)

    txt_files = glob.glob(os.path.join(labels_dir, '**', '*.txt'), recursive=True)
    moved_count = 0
    
    for txt_path in txt_files:
        # Check if file is completely empty
        if os.path.getsize(txt_path) == 0:
            # Figure out the relative path to maintain train/val structure
            rel_path = os.path.relpath(txt_path, labels_dir)
            
            # Label paths
            dest_label_path = os.path.join(q_labels_dir, rel_path)
            os.makedirs(os.path.dirname(dest_label_path), exist_ok=True)
            
            # Corresponding image paths (.jpg, .png, etc.)
            # Assuming images have same base name
            img_rel_path_base = os.path.splitext(rel_path)[0]
            img_dir = os.path.dirname(os.path.join(images_dir, rel_path))
            base_name = os.path.basename(img_rel_path_base)
            
            # Find the actual image file (could be jpg, png, jpeg)
            possible_images = glob.glob(os.path.join(img_dir, base_name + ".*"))
            
            if possible_images:
                img_path = possible_images[0]
                img_dest_path = os.path.join(q_images_dir, os.path.relpath(img_path, images_dir))
                os.makedirs(os.path.dirname(img_dest_path), exist_ok=True)
                
                # Move them
                shutil.move(img_path, img_dest_path)
                shutil.move(txt_path, dest_label_path)
                moved_count += 1
            else:
                # If no image found, just move the txt file
                shutil.move(txt_path, dest_label_path)
                moved_count += 1
                
    # Clean cache
    for cache_file in glob.glob(os.path.join(labels_dir, '**', '*.cache'), recursive=True):
        os.remove(cache_file)
        
    return moved_count

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    moved = quarantine_empty_labels(base_dir)
    print(f"Quarantined {moved} empty background images and labels.")
