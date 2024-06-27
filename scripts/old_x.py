"""
To run comparing 2 images, you can use the following command:
python3 scripts/old_x.py \
    --target_img_path=media/target_image.jpg \
    --scene_img_path=media/target_image.jpg \
    --debug_print_results_to_console \
    --save_matching_results_to_image
"""


import sys
import os

# Get the path to the package directory
package_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, package_dir)

# Add the path to the accelerated_features directory
accelerated_features_dir = os.path.join(package_dir, 'libs', 'accelerated_features')
sys.path.insert(0, accelerated_features_dir)

# Add the modules directory to the path
modules_dir = os.path.join(accelerated_features_dir, 'modules')
sys.path.insert(0, modules_dir)

# Now you can import XFeat
from modules.xfeat import XFeat
import numpy as np
import torch
import cv2
import argparse
import time
import queue
import threading

xfeat = XFeat()

# Global queue for image saving
image_save_queue = queue.Queue()
stop_worker = False

# Compare two images and return a similarity score
def compare_images(target_image, scene_image, debug_print=False, save_results=False):
    if debug_print:
        start_time = time.time()
    # Convert numpy arrays to torch tensors
    target_tensor = torch.from_numpy(target_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    scene_tensor = torch.from_numpy(scene_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    
    # Detect and compute features
    max_features = 4096
    target_output = xfeat.detectAndCompute(target_tensor, top_k=max_features)[0]
    scene_output = xfeat.detectAndCompute(scene_tensor, top_k=max_features)[0]
    
    # Match the two images
    mkpts_0, mkpts_1 = xfeat.match_xfeat(target_tensor, scene_tensor)
    
    # Convert keypoints to numpy arrays for OpenCV operations
    kpts_0 = target_output['keypoints'].cpu().numpy()
    kpts_1 = scene_output['keypoints'].cpu().numpy()
    

    
    # Ensure we have matching points before trying to estimate homography
    if len(mkpts_0) < 4 or len(mkpts_1) < 4:
        print("Not enough matching points to estimate homography")
        return 0.0  # Return lowest similarity score
    
    # Estimate homography
    H, mask = cv2.findHomography(mkpts_0, mkpts_1, cv2.RANSAC, 5.0)
    
    # Calculate metrics
    """
        - Number of matches: This is the number of feature points that were successfully matched between the two images.
        - Number of inliers: These are the matched points that are consistent with the estimated homography (transformation between the images). For identical images, this should be equal to the number of matches.
        - Inlier ratio: This is the ratio of inliers to total matches. A value of 1.0 means all matches are consistent with the homography, which is expected for identical images.
        - Feature ratio: This is the ratio of the number of features in the image with fewer features to the image with more features. For identical images, this should be 1.0.
        - Number of keypoints in target image: This is the number of feature points detected in the target image.
    """
    # Calculate metrics
    num_matches = len(mkpts_0)

    num_inliers = np.sum(mask) if mask is not None else 0
    inlier_ratio = num_inliers / num_matches if num_matches > 0 else 0
    feature_ratio = min(len(kpts_0), len(kpts_1)) / max(len(kpts_0), len(kpts_1))
    
    match_ratio = num_matches / max_features

    # Debug output
    if debug_print:
        execution_time = time.time() - start_time
        print(f"Number of matches: {num_matches}")
        print(f"Number of inliers: {num_inliers}")
        print(f"Inlier ratio: {inlier_ratio}")
        print(f"Feature ratio: {feature_ratio}")
        print(f"Match ratio: {match_ratio}")
        print(f"Number of keypoints in target image: {len(kpts_0)}")
        print(f"Number of keypoints in scene image: {len(kpts_1)}")
        print(f"Execution time: {execution_time:.4f} seconds")


    # Check for identical images
    if (inlier_ratio == 1.0 and 
        feature_ratio == 1.0 and 
        len(kpts_0) == len(kpts_1) == num_matches):
        similarity = 1.0
    else:
        # Calculate similarity for non-identical images
        # Bound the similarity score between 0 and 1
        weight_inlier = 0.4
        weight_feature = 0.3
        weight_match = 0.3
        similarity = min(1.0, (weight_inlier * inlier_ratio
                                + weight_feature * feature_ratio 
                                + weight_feature * match_ratio
                                ) 
                                * (1 + match_ratio)
                                )

    if save_results:
        image_save_queue.put((target_image, scene_image, mkpts_0, mkpts_1, similarity))

    return similarity

# Run the standard mode of the script
def run_standard_mode(debug_print=False, save_results=False):
    while True:
        # Read image dimensions
        line = sys.stdin.buffer.readline().strip()
        if not line:
            break
        h1, w1, c1, h2, w2, c2 = map(int, line.split())
        
        # Read image data
        target_data = sys.stdin.buffer.read(h1 * w1 * c1)
        scene_data = sys.stdin.buffer.read(h2 * w2 * c2)
        
        if not target_data or not scene_data:
            break
        
        # Convert to numpy arrays and reshape
        target_image = np.frombuffer(target_data, dtype=np.uint8).reshape((h1, w1, c1))
        scene_image = np.frombuffer(scene_data, dtype=np.uint8).reshape((h2, w2, c2))
        
        # Compare images and get similarity
        similarity = compare_images(target_image, scene_image, debug_print, save_results)
        
        # Write result to stdout
        sys.stdout.buffer.write(f"{similarity:.6f}\n".encode('utf-8'))
        sys.stdout.buffer.flush()

# Test the image comparison function with local files
def local_file_test(target_img_path, scene_img_path, debug_print=False, save_results=False):
    target_image = cv2.imread(target_img_path)
    scene_image = cv2.imread(scene_img_path)

    if target_image is None:
        print(f"ERROR: Could not read target image from {target_img_path}")
        return
    if scene_image is None:
        print(f"ERROR: Could not read scene image from {scene_img_path}")
        return

    similarity = compare_images(target_image, scene_image, debug_print, save_results)
    print(f"Similarity score: {similarity:.6f}")

# Worker thread to save matching results to an image
def save_matching_results_worker():
    global stop_worker
    last_save_time = 0
    while not stop_worker:
        current_time = time.time()
        if current_time - last_save_time >= 4:  # Save every 4 seconds
            try:
                target_image, scene_image, mkpts_0, mkpts_1, similarity = image_save_queue.get(timeout=1)
                save_image(target_image, scene_image, mkpts_0, mkpts_1, similarity)
                last_save_time = current_time
            except queue.Empty:
                pass  # No new image to save, continue waiting
        else:
            time.sleep(0.1)  # Sleep for a short time to avoid busy waiting
    
    # Process any remaining items in the queue
    while not image_save_queue.empty():
        target_image, scene_image, mkpts_0, mkpts_1, similarity = image_save_queue.get()
        save_image(target_image, scene_image, mkpts_0, mkpts_1, similarity)

# Save the image with matching keypoints
def save_image(target_image, scene_image, mkpts_0, mkpts_1, similarity_score):
    # Create a new image with both target and scene images side by side
    h1, w1 = target_image.shape[:2]
    h2, w2 = scene_image.shape[:2]
    new_h = max(h1, h2) + 40  # Extra space for labels
    new_img = np.zeros((new_h, w1 + w2, 3), dtype=np.uint8)
    
    # Add white background
    new_img.fill(255)
    
    # Place images
    new_img[40:40+h1, :w1] = target_image
    new_img[40:40+h2, w1:w1+w2] = scene_image
    
    # Draw a border between images
    cv2.line(new_img, (w1, 0), (w1, new_h), (0, 0, 0), 2)
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(new_img, "Target Image", (10, 30), font, 0.7, (0, 0, 0), 2)
    cv2.putText(new_img, "Scene Image", (w1 + 10, 30), font, 0.7, (0, 0, 0), 2)
    
    # Add similarity score
    score_text = f"Similarity Score: {similarity_score:.4f}"
    cv2.putText(new_img, score_text, (10, new_h - 10), font, 0.7, (0, 0, 0), 2)

    # Draw matching keypoints
    for pt1, pt2 in zip(mkpts_0, mkpts_1):
        pt1 = tuple(map(int, [pt1[0], pt1[1] + 40]))  # Adjust y-coordinate for label space
        pt2 = tuple(map(int, [pt2[0] + w1, pt2[1] + 40]))  # Adjust x-coordinate for scene image and y for label
        color = (0, 255, 0)  # Green color
        cv2.circle(new_img, pt1, 3, color, -1)
        cv2.circle(new_img, pt2, 3, color, -1)
        cv2.line(new_img, pt1, pt2, color, 1)

    # Save the image
    cv2.imwrite(f'matching_results_{time.time():.0f}.jpg', new_img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='XFeat Image Comparison')
    parser.add_argument('--target_img_path', type=str, help='Path to the target image')
    parser.add_argument('--scene_img_path', type=str, help='Path to the scene image')
    parser.add_argument('--debug_print_results_to_console', action='store_true', help='Enable debug prints')
    parser.add_argument('--save_matching_results_to_image', action='store_true', help='Save matching results to an image')
    args = parser.parse_args()

    save_thread = None
    try:
        # Start the image saving worker thread
        if args.save_matching_results_to_image:
            save_thread = threading.Thread(target=save_matching_results_worker, daemon=True)
            save_thread.start()

        if args.target_img_path and args.scene_img_path:
            local_file_test(args.target_img_path, args.scene_img_path, 
                           args.debug_print_results_to_console, 
                           args.save_matching_results_to_image)
        else:
            run_standard_mode(args.debug_print_results_to_console, 
                              args.save_matching_results_to_image)
    finally:
        if save_thread:
            # Signal the worker thread to stop
            stop_worker = True
            # Wait for the worker thread to finish (with a timeout)
            save_thread.join(timeout=10)  # Wait up to 10 seconds
            
            # If there are still items in the queue, process them
            while not image_save_queue.empty():
                target_image, scene_image, mkpts_0, mkpts_1 = image_save_queue.get()
                save_image(target_image, scene_image, mkpts_0, mkpts_1)