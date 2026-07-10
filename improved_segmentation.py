"""
Improved Plant Segmentation - Clean extraction of only green plant parts
Matches the expected output: only leaves and stems, no glass, no roots
"""

import cv2
import numpy as np
from pathlib import Path
import os


class ImprovedPlantSegmenter:
    def __init__(self):
        """Initialize improved segmentation pipeline"""
        self.debug = True
        
    def detect_tube_regions(self, image):
        """
        Detect individual test tube regions
        """
        height, width = image.shape[:2]
        
        # Simple approach: divide into 8 equal segments
        segment_width = width // 8
        tube_regions = []
        
        for i in range(8):
            x = i * segment_width
            tube_regions.append((x, 0, segment_width, height))
        
        return tube_regions
    
    def extract_green_plant_only(self, tube_image):
        """
        Extract ONLY green plant parts using aggressive color filtering
        """
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(tube_image, cv2.COLOR_BGR2HSV)
        
        # Create multiple masks for different shades of green
        masks = []
        
        # Dark green (main leaves)
        lower1 = np.array([35, 40, 30])
        upper1 = np.array([85, 255, 255])
        mask1 = cv2.inRange(hsv, lower1, upper1)
        masks.append(mask1)
        
        # Medium green
        lower2 = np.array([30, 30, 25])
        upper2 = np.array([90, 255, 255])
        mask2 = cv2.inRange(hsv, lower2, upper2)
        masks.append(mask2)
        
        # Light/yellowish green
        lower3 = np.array([25, 25, 20])
        upper3 = np.array([95, 255, 255])
        mask3 = cv2.inRange(hsv, lower3, upper3)
        masks.append(mask3)
        
        # Combine all green masks
        green_mask = masks[0]
        for mask in masks[1:]:
            green_mask = cv2.bitwise_or(green_mask, mask)
        
        # Remove non-green colors more aggressively
        # Convert to LAB color space for better separation
        lab = cv2.cvtColor(tube_image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # In LAB, green has negative 'a' values
        # Filter out non-green colors
        green_lab_mask = cv2.inRange(a, 0, 135)  # Green has lower 'a' values
        
        # Combine HSV and LAB masks
        combined_mask = cv2.bitwise_and(green_mask, green_lab_mask)
        
        # Remove very bright pixels (glass reflections)
        gray = cv2.cvtColor(tube_image, cv2.COLOR_BGR2GRAY)
        _, no_bright = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        combined_mask = cv2.bitwise_and(combined_mask, no_bright)
        
        # Remove very dark pixels
        _, no_dark = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        combined_mask = cv2.bitwise_and(combined_mask, no_dark)
        
        # Morphological operations to clean up
        kernel_small = np.ones((3, 3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_small, iterations=2)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        
        # Remove small noise
        kernel_medium = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_medium, iterations=1)
        
        return combined_mask
    
    def remove_roots_aggressive(self, mask, height):
        """
        Aggressively remove roots - keep only top 60% of the tube
        """
        cutoff = int(height * 0.60)
        mask[cutoff:, :] = 0
        return mask
    
    def remove_glass_and_artifacts(self, tube_image, mask):
        """
        Remove all glass tube artifacts and non-plant elements
        """
        # Create a copy of the mask
        clean_mask = mask.copy()
        
        # Find contours in the mask
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area - remove very small ones (noise)
        min_area = 50  # Minimum area for plant parts
        filtered_mask = np.zeros_like(clean_mask)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                cv2.drawContours(filtered_mask, [contour], -1, 255, -1)
        
        # Additional filtering: check if pixels are actually green in original image
        hsv = cv2.cvtColor(tube_image, cv2.COLOR_BGR2HSV)
        
        # Very strict green filter
        lower_strict = np.array([30, 35, 25])
        upper_strict = np.array([90, 255, 255])
        strict_green = cv2.inRange(hsv, lower_strict, upper_strict)
        
        # Combine with filtered mask
        final_mask = cv2.bitwise_and(filtered_mask, strict_green)
        
        # One more morphological cleanup
        kernel = np.ones((3, 3), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        return final_mask
    
    def process_single_tube(self, tube_image):
        """
        Process a single test tube to extract only plant parts
        """
        height, width = tube_image.shape[:2]
        
        # Step 1: Extract green plant parts
        green_mask = self.extract_green_plant_only(tube_image)
        
        # Step 2: Remove roots (bottom portion)
        green_mask = self.remove_roots_aggressive(green_mask, height)
        
        # Step 3: Remove glass and artifacts
        clean_mask = self.remove_glass_and_artifacts(tube_image, green_mask)
        
        # Step 4: Apply mask to extract plant on black background
        result = np.zeros_like(tube_image)
        result[clean_mask > 0] = tube_image[clean_mask > 0]
        
        return result
    
    def process_image(self, input_path, output_path):
        """
        Main processing pipeline
        """
        print(f"\nProcessing: {input_path}")
        
        # Load image
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"Could not load image: {input_path}")
        
        height, width = image.shape[:2]
        print(f"Image size: {width}x{height}")
        
        # Create black background
        output_image = np.zeros_like(image)
        
        # Detect test tube regions (8 equal segments)
        tube_regions = self.detect_tube_regions(image)
        print(f"Processing {len(tube_regions)} tubes...")
        
        # Process each tube
        for idx, (x, y, w, h) in enumerate(tube_regions):
            print(f"  Tube {idx + 1}/8...", end=" ")
            
            # Extract tube region
            tube_roi = image[y:y+h, x:x+w]
            
            # Segment plant only
            segmented_tube = self.process_single_tube(tube_roi)
            
            # Place on output image
            output_image[y:y+h, x:x+w] = segmented_tube
            
            print("✓")
        
        # Save result
        cv2.imwrite(output_path, output_image)
        print(f"✓ Saved: {output_path}\n")
        
        return output_image
    
    def process_folder(self, input_folder, output_folder):
        """
        Process all images in input folder
        """
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True)
        
        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = [
            f for f in input_path.iterdir() 
            if f.suffix.lower() in image_extensions
        ]
        
        if not image_files:
            print(f"No images found in '{input_folder}'")
            return
        
        print(f"\n{'='*70}")
        print(f"IMPROVED PLANT SEGMENTATION")
        print(f"Found {len(image_files)} images to process")
        print(f"{'='*70}")
        
        for img_file in image_files:
            output_file = output_path / f"improved_{img_file.name}"
            
            try:
                self.process_image(str(img_file), str(output_file))
            except Exception as e:
                print(f"✗ Error processing {img_file.name}: {e}\n")
        
        print(f"{'='*70}")
        print(f"Processing complete! Results saved in '{output_folder}'")
        print(f"{'='*70}\n")


def main():
    """Main execution"""
    segmenter = ImprovedPlantSegmenter()
    
    input_folder = "input"
    output_folder = "output"
    
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' not found!")
        return
    
    segmenter.process_folder(input_folder, output_folder)


if __name__ == "__main__":
    main()
