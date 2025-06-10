import cv2
import numpy as np
import math
import random

def create_string_art(image_path, nails=200, max_lines=10000, output_size=800):
    # 1. Load and prepare image
    img = cv2.imread(image_path)
    if img is None:
        print("Error loading image")
        return None
    
    # Resize image to square while maintaining aspect ratio
    h, w = img.shape[:2]
    size = min(h, w)
    img = img[(h-size)//2:(h-size)//2+size, (w-size)//2:(w-size)//2+size]
    img = cv2.resize(img, (output_size, output_size))
    
    # 2. Convert to grayscale and enhance contrast
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    h, w = gray.shape
    
    # 3. Create circular mask
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w//2, h//2)
    radius = int(min(h, w) * 0.45)
    cv2.circle(mask, center, radius, 255, -1)
    
    # 4. Apply mask and invert brightness
    masked = cv2.bitwise_and(gray, gray, mask=mask)
    inverted = 255 - masked
    
    # 5. Create nail positions on circumference
    nails_pos = []
    for i in range(nails):
        angle = 2 * math.pi * i / nails
        x = int(center[0] + radius * math.cos(angle))
        y = int(center[1] + radius * math.sin(angle))
        nails_pos.append((x, y))
    
    # 6. Main algorithm
    result = np.ones((h, w, 3), dtype=np.uint8) * 255
    density_map = np.zeros((h, w), dtype=np.float32)
    
    # Precompute all possible line segments
    line_segments = [(i, j) for i in range(nails) for j in range(i+1, nails)]
    random.shuffle(line_segments)
    
    lines_drawn = 0
    for i, j in line_segments[:max_lines]:
        pt1, pt2 = nails_pos[i], nails_pos[j]
        
        # Create line mask
        line_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.line(line_mask, pt1, pt2, 255, 1)
        
        # Calculate line importance
        brightness = cv2.mean(inverted, mask=line_mask)[0]
        line_length = math.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)
        
        # Adjust brightness by line length (longer lines are less important)
        adjusted_brightness = brightness * (1 - 0.5*line_length/radius)
        
        if adjusted_brightness > 30:  # Adjustable threshold
            # Vary thickness based on brightness
            thickness = max(1, int(adjusted_brightness / 40))
            cv2.line(result, pt1, pt2, (0, 0, 0), thickness, lineType=cv2.LINE_AA)
            cv2.line(density_map, pt1, pt2, adjusted_brightness, 1, lineType=cv2.LINE_AA)
            lines_drawn += 1
    
    print(f"Drew {lines_drawn} lines out of {max_lines} possible")
    
    # 7. Enhance contrast
    density_map = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX)
    density_map = density_map.astype(np.uint8)
    
    # 8. Final blending
    result = cv2.addWeighted(result, 0.9, 
                           cv2.cvtColor(density_map, cv2.COLOR_GRAY2BGR), 
                           0.1, 0)
    
    # 9. Draw nails and circle
    for pt in nails_pos:
        cv2.circle(result, pt, 2, (0, 0, 255), -1)  # Red nails for visibility
    cv2.circle(result, center, radius, (0, 0, 0), 1)
    
    return result

# Usage example
input_img = "face1.jpg"  # Replace with your image path
output_img = "string_art_output.jpg"

art = create_string_art(input_img, nails=300, max_lines=5000)
if art is not None:
    cv2.imwrite(output_img, art)
    cv2.imshow("String Art", art)
    cv2.waitKey(0)
    cv2.destroyAllWindows()