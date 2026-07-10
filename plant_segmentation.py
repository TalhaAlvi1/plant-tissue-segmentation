"""
Plant Segmentation Algorithm for Test Tube Images
Segments plant leaves and stems, excluding roots and glass tubes
"""

import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path


class PlantSegmenter:
    def __init__(self):
        """Initialize the plant segmentation pipeline"""
        self.debug = True
        
    def detect_test_tubes(self, image):
        """
        Detect individual test tube regions in the image
        Returns list of bounding boxes for each tube
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding to find tube boundaries
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 21, 5
        )
        
        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter and sort contours by x-coordinate (left to right)
        height, width = image.shape[:2]
        min_area = (height * width) * 0.01  # At least 1% of image
        
        tube_boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                # Filter by aspect ratio (tubes are tall and narrow)
                aspect_ratio = h / w if w > 0 else 0
                if aspect_ratio > 1.5:  # Tubes should be taller than wide
                    tube_boxes.append((x, y, w, h))
        
        # Sort by x-coordinate (left to right)
        tube_boxes.sort(key=lambda box: box[0])
        
        # If detection fails, divide image into equal segments
        if len(tube_boxes) < 6:
            tube_boxes = self._divide_into_segments(image, num_segments=8)
        
        return tube_boxes
    
    def _divide_into_segments(self, image, num_segments=8):
        """Fallback: divide image into equal vertical segments"""
        height, width = image.shape[:2]
        segment_width = width // num_segments
        
        boxes = []
        for i in range(num_segments):
            x = i * segment_width
            boxes.append((x, 0, segment_width, height))
        
        return boxes
    
    def segment_plant_from_tube(self, tube_image):
        """
        Segment plant (leaves + stem) from a single test tube image
        Excludes roots, glass, and background
        """
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(tube_image, cv2.COLOR_BGR2HSV)
        
        # Define range for green color (leaves and stem)
        # Adjusted for various shades of green
        lower_green1 = np.array([25, 20, 20])
        upper_green1 = np.array([90, 255, 255])
        
        # Create mask for green regions
        green_mask = cv2.inRange(hsv, lower_green1, upper_green1)
        
        # Also detect yellowish-green and light green
        lower_green2 = np.array([20, 15, 20])
        upper_green2 = np.array([95, 255, 255])
        green_mask2 = cv2.inRange(hsv, lower_green2, upper_green2)
        
        # Combine masks
        plant_mask = cv2.bitwise_or(green_mask, green_mask2)
        
        # Remove noise with morphological operations
        kernel = np.ones((3, 3), np.uint8)
        plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Remove roots (bottom portion of the tube)
        height = tube_image.shape[0]
        root_cutoff = int(height * 0.65)  # Keep only top 65% (adjust as needed)
        plant_mask[root_cutoff:, :] = 0
        
        # Apply mask to extract plant
        segmented_plant = cv2.bitwise_and(tube_image, tube_image, mask=plant_mask)
        
        return segmented_plant, plant_mask
    
    def remove_glass_artifacts(self, segmented_plant, mask):
        """
        Remove glass tube reflections and artifacts
        """
        # Convert to grayscale
        gray = cv2.cvtColor(segmented_plant, cv2.COLOR_BGR2GRAY)
        
        # Remove very bright pixels (glass reflections)
        _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Combine with existing mask
        refined_mask = cv2.bitwise_and(mask, bright_mask)
        
        # Apply refined mask
        result = cv2.bitwise_and(segmented_plant, segmented_plant, mask=refined_mask)
        
        return result, refined_mask
    
    def process_image(self, input_path, output_path):
        """
        Main processing pipeline:
        1. Load image
        2. Detect test tubes
        3. Segment plants from each tube
        4. Combine results on black background
        """
        # Load image
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"Could not load image: {input_path}")
        
        height, width = image.shape[:2]
        
        # Create black background for output
        output_image = np.zeros_like(image)
        
        # Detect test tubes
        tube_boxes = self.detect_test_tubes(image)
        
        print(f"Detected {len(tube_boxes)} test tube regions")
        
        # Process each test tube
        for idx, (x, y, w, h) in enumerate(tube_boxes):
            print(f"Processing tube {idx + 1}/{len(tube_boxes)}...")
            
            # Extract tube region
            tube_roi = image[y:y+h, x:x+w]
            
            # Segment plant from tube
            segmented_plant, mask = self.segment_plant_from_tube(tube_roi)
            
            # Remove glass artifacts
            clean_plant, refined_mask = self.remove_glass_artifacts(segmented_plant, mask)
            
            # Place segmented plant on black background at original position
            output_image[y:y+h, x:x+w] = clean_plant
        
        # Save result
        cv2.imwrite(output_path, output_image)
        print(f"Segmentation complete! Saved to: {output_path}")
        
        return output_image
    
    def process_folder(self, input_folder, output_folder):
        """Process all images in a folder"""
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True)
        
        # Supported image formats
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        image_files = [
            f for f in input_path.iterdir() 
            if f.suffix.lower() in image_extensions
        ]
        
        print(f"Found {len(image_files)} images to process")
        
        for img_file in image_files:
            print(f"\n{'='*60}")
            print(f"Processing: {img_file.name}")
            print(f"{'='*60}")
            
            output_file = output_path / f"segmented_{img_file.name}"
            
            try:
                self.process_image(str(img_file), str(output_file))
            except Exception as e:
                print(f"Error processing {img_file.name}: {e}")


def main():
    """Main execution function"""
    # Initialize segmenter
    segmenter = PlantSegmenter()
    
    # Process all images in input folder
    input_folder = "input"
    output_folder = "output"
    
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' not found!")
        return
    
    segmenter.process_folder(input_folder, output_folder)
    
    print("\n" + "="*60)
    print("All images processed successfully!")
    print(f"Results saved in '{output_folder}' folder")
    print("="*60)


if __name__ == "__main__":
    main()
