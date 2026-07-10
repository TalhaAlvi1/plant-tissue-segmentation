"""
Visualize segmentation results - compare input and output side by side
"""

import cv2
import numpy as np
from pathlib import Path
import os


def create_comparison(input_path, output_path, comparison_path):
    """
    Create side-by-side comparison of input and segmented output
    """
    # Load images
    input_img = cv2.imread(input_path)
    output_img = cv2.imread(output_path)
    
    if input_img is None or output_img is None:
        print(f"Error loading images")
        return
    
    # Resize if needed to match heights
    h1, w1 = input_img.shape[:2]
    h2, w2 = output_img.shape[:2]
    
    if h1 != h2:
        # Resize to match the smaller height
        target_height = min(h1, h2)
        input_img = cv2.resize(input_img, (int(w1 * target_height / h1), target_height))
        output_img = cv2.resize(output_img, (int(w2 * target_height / h2), target_height))
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    font_thickness = 3
    color = (255, 255, 255)
    
    # Add "Input" label
    cv2.putText(input_img, "Input", (20, 50), font, font_scale, color, font_thickness)
    
    # Add "Segmented" label
    cv2.putText(output_img, "Segmented", (20, 50), font, font_scale, (0, 255, 0), font_thickness)
    
    # Concatenate horizontally
    comparison = np.hstack([input_img, output_img])
    
    # Save comparison
    cv2.imwrite(comparison_path, comparison)
    print(f"✓ Created comparison: {comparison_path}")
    
    return comparison


def main():
    """Create comparisons for all processed images"""
    input_folder = Path("input")
    output_folder = Path("output")
    comparison_folder = Path("comparisons")
    comparison_folder.mkdir(exist_ok=True)
    
    # Find all segmented images
    segmented_files = list(output_folder.glob("segmented_*.jpeg")) + \
                     list(output_folder.glob("segmented_*.jpg")) + \
                     list(output_folder.glob("segmented_*.png"))
    
    if not segmented_files:
        print("No segmented images found in output folder")
        return
    
    print(f"\n{'='*70}")
    print(f"Creating comparison images...")
    print(f"{'='*70}\n")
    
    for seg_file in segmented_files:
        # Get original filename
        original_name = seg_file.name.replace("segmented_", "")
        input_file = input_folder / original_name
        
        if not input_file.exists():
            print(f"⚠ Original file not found: {original_name}")
            continue
        
        # Create comparison
        comparison_file = comparison_folder / f"comparison_{original_name}"
        create_comparison(str(input_file), str(seg_file), str(comparison_file))
    
    print(f"\n{'='*70}")
    print(f"All comparisons saved in '{comparison_folder}' folder")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
