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
from PIL import Image
import torch
import cv2
import argparse
import time
import queue
import threading
from abc import ABC, abstractmethod

class SimilarityScorer(ABC):

    xfeat = XFeat()
    def __init__(self, **kwargs):
        #Gives all implementations access to the XFeat class

        # Matching weights and similarity score
        self.weight_inlier = 0.4
        self.weight_feature = 0.3
        self.weight_match = 0.3
        self.similarity = 0.0

    # Calculate matching scores based on the target and scene images
    # We calculate the number of matches, number of inliers, inlier ratio, feature ratio, and match ratio
    # uses the XFeat class to detect and compute features, match features
    # Homography is calculated using OpenCV
    def calculate_matching_scores(self):
        target_tensor = torch.from_numpy(self.target_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        scene_tensor = torch.from_numpy(self.scene_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        max_features = 4096
        target_output = self.xfeat.detectAndCompute(target_tensor, top_k=max_features)[0]
        scene_output = self.xfeat.detectAndCompute(scene_tensor, top_k=max_features)[0]
        
        mkpts_0, mkpts_1 = self.xfeat.match_xfeat(target_tensor, scene_tensor)
        
        if len(mkpts_0) < 4 or len(mkpts_1) < 4:
            return None, None, 0, 0, 0, 0, 0, [], []

        H, mask = cv2.findHomography(mkpts_0, mkpts_1, cv2.RANSAC, 5.0)
        
        num_matches = len(mkpts_0)
        num_inliers = np.sum(mask) if mask is not None else 0
        inlier_ratio = num_inliers / num_matches if num_matches > 0 else 0
        feature_ratio = min(len(target_output['keypoints']), len(scene_output['keypoints'])) / max(len(target_output['keypoints']), len(scene_output['keypoints']))
        match_ratio = num_matches / max_features

        return target_output, scene_output, num_matches, num_inliers, inlier_ratio, feature_ratio, match_ratio, mkpts_0, mkpts_1


    # Set the similarity score based on the inlier ratio, feature ratio, and match ratio
    # If the inlier ratio is 1.0, feature ratio is 1.0, and the number of matches is the same for both images, the similarity score is 1.0
    """
        - Number of matches: This is the number of feature points that were successfully matched between the two images.
        - Number of inliers: These are the matched points that are consistent with the estimated homography (transformation between the images). For identical images, this should be equal to the number of matches.
        - Inlier ratio: This is the ratio of inliers to total matches. A value of 1.0 means all matches are consistent with the homography, which is expected for identical images.
        - Feature ratio: This is the ratio of the number of features in the image with fewer features to the image with more features. For identical images, this should be 1.0.
        - Number of keypoints in target image: This is the number of feature points detected in the target image.
    """
    def set_similarity_score(self, inlier_ratio, feature_ratio, match_ratio):
        try:
            if (inlier_ratio == 1.0 and 
                feature_ratio == 1.0):
                    
                self.similarity = 1.0

            else:
                self.similarity = min(1.0, (self.weight_inlier * inlier_ratio
                                        + self.weight_feature * feature_ratio
                                        + self.weight_match * match_ratio
                                        )
                                        * (1 + match_ratio)
                                        )
        except Exception as e:
            print(f"Error setting similarity score: {e}")
            self.similarity = 0.0
    

    def save_debug_image(self, target_image, scene_image, mkpts_0, mkpts_1, similarity_score):
        try:
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
                pt1 = tuple(map(int, [pt1[0], pt1[1] + 40]))
                pt2 = tuple(map(int, [pt2[0] + w1, pt2[1] + 40]))
                color = (0, 255, 0)  # Green color
                cv2.circle(new_img, pt1, 3, color, -1)
                cv2.circle(new_img, pt2, 3, color, -1)
                cv2.line(new_img, pt1, pt2, color, 1)

            # Save the image
            cv2.imwrite(f'{self.match_result_image_output_path}/matching_results_{time.time():.0f}.jpg', new_img)

        except Exception as e:
            print(f"Error saving debug image: {e}")


class SimilarityScorer_realtime(SimilarityScorer):
    def __init__(self, debug_print=False, match_result_image_output_path=None, match_result_image_save_hz=0.25):
        try:
            super().__init__()

            self.debug_print = debug_print
            self.match_result_image_output_path = match_result_image_output_path
            self.match_result_image_save_hz = match_result_image_save_hz
            
            self.target_image = None
            self.scene_image = None 


            self.image_save_queue = queue.Queue()
            self.stop_worker = False
            self.last_save_time = 0

            if self.match_result_image_output_path:
                self.save_thread = threading.Thread(target=self.save_matching_results_worker, daemon=True)
                self.save_thread.start()

            if self.debug_print or self.match_result_image_output_path:
                # Check if the output path exists
                if self.match_result_image_output_path:
                    raise Exception("Output path does not exist")

                self.run_debug_loop()
            else:
                self.run_realtime_loop()

        except Exception as e:
            raise e

    def run_realtime_loop(self):
        while True:
            if not self.get_images_from_memory_buffer():
                break
            if self.target_image is not None and self.scene_image is not None:
                self.compare_images_realtime()
                sys.stdout.buffer.write(f"{self.similarity:.6f}\n".encode('utf-8'))
                sys.stdout.buffer.flush()

    def run_debug_loop(self):
        while True:
            if not self.get_images_from_memory_buffer():
                break
            if self.target_image is not None and self.scene_image is not None:
                self.similarity, debug_info = self.compare_images_debug()
                sys.stdout.buffer.write(f"{self.similarity:.6f}\n".encode('utf-8'))
                sys.stdout.buffer.flush()
                
                if self.debug_print:
                    print(debug_info)
                
                if self.match_result_image_output_path:
                    current_time = time.time()
                    if current_time - self.last_save_time >= 1 / self.match_result_image_save_hz:
                        self.image_save_queue.put((self.target_image, self.scene_image, debug_info['mkpts_0'], debug_info['mkpts_1'], self.similarity))
                        self.last_save_time = current_time

    def get_images_from_memory_buffer(self):
        # Read image dimensions
        line = sys.stdin.buffer.readline().strip()
        if not line:
            return False  # End of input

        h1, w1, c1, h2, w2, c2 = map(int, line.split())
        
        # Read image data
        target_data = sys.stdin.buffer.read(h1 * w1 * c1)
        scene_data = sys.stdin.buffer.read(h2 * w2 * c2)

        if not target_data or not scene_data:
            return False  # End of input
        
        # Convert to numpy arrays and reshape
        self.target_image = np.frombuffer(target_data, dtype=np.uint8).reshape((h1, w1, c1))
        self.scene_image = np.frombuffer(scene_data, dtype=np.uint8).reshape((h2, w2, c2))
        return True


    def compare_images_realtime(self):
        target_output, scene_output, num_matches, num_inliers, inlier_ratio, feature_ratio, match_ratio, mkpts_0, mkpts_1 = self.calculate_matching_scores()
        self.set_similarity_score(inlier_ratio, feature_ratio, match_ratio)

    def compare_images_debug(self):
        if self.debug_print:
            start_time = time.time()

        target_output, scene_output, num_matches, num_inliers, inlier_ratio, feature_ratio, match_ratio, mkpts_0, mkpts_1 = self.calculate_matching_scores()

        self.set_similarity_score(inlier_ratio, feature_ratio, match_ratio)

        if self.debug_print:
            execution_time = time.time() - start_time
            debug_info = {
                "num_matches": num_matches,
                "num_inliers": num_inliers,
                "inlier_ratio": inlier_ratio,
                "feature_ratio": feature_ratio,
                "match_ratio": match_ratio,
                "num_keypoints_target": len(target_output['keypoints']),
                "num_keypoints_scene": len(scene_output['keypoints']),
                "execution_time": execution_time,
                "mkpts_0": mkpts_0,
                "mkpts_1": mkpts_1
            }
            print(execution_time)
            print(debug_info)

        if self.match_result_image_output_path:
            current_time = time.time()
            if current_time - self.last_save_time >= 1 / self.match_result_image_save_hz:
                self.image_save_queue.put((self.target_image, self.scene_image, mkpts_0, mkpts_1, self.similarity))
                self.last_save_time = current_time

    def save_matching_results_worker(self):
        try:
            while not self.stop_worker:
                try:
                    target_image, scene_image, mkpts_0, mkpts_1, similarity = self.image_save_queue.get(timeout=1)
                    self.save_debug_image(target_image, scene_image, mkpts_0, mkpts_1, similarity)
                except queue.Empty:
                    pass
        except Exception as e:
            raise e

    def __del__(self):
        self.stop_worker = True
        if hasattr(self, 'save_thread'):
            self.save_thread.join(timeout=10)  # Wait up to 10 seconds
        
        # If there are still items in the queue, process them
        while not self.image_save_queue.empty():
            target_image, scene_image, mkpts_0, mkpts_1, similarity = self.image_save_queue.get()
            self.save_debug_image(target_image, scene_image, mkpts_0, mkpts_1, similarity)

class SimilarityScorer_image_comparison(SimilarityScorer):
    def __init__(self, target_img_path, scene_img_path, debug_print=False, match_result_image_output_path=None):
        try:
            super().__init__()
            self.target_img_path = target_img_path
            self.scene_img_path = scene_img_path
            self.debug_print = debug_print
            self.match_result_image_output_path = match_result_image_output_path
            self.target_image = None
            self.scene_image = None
        
            self.run_static_image_comparison()

        except Exception as e:
            raise e

    def run_static_image_comparison(self):
        self.target_image = np.array(Image.open(self.target_img_path))
        self.scene_image = np.array(Image.open(self.scene_img_path))

        if self.target_image is None or self.scene_image is None:
            raise Exception("Error loading images")

        self.compare_images_static()
        print(f"Similarity Score: {self.similarity:.4f}")

    def compare_images_static(self):
        if self.debug_print:
            start_time = time.time()

        target_output, scene_output, num_matches, num_inliers, inlier_ratio, feature_ratio, match_ratio, mkpts_0, mkpts_1 = self.calculate_matching_scores()
        self.set_similarity_score(inlier_ratio, feature_ratio, match_ratio)

        if self.debug_print:
            execution_time = time.time() - start_time
            debug_info = {
                "num_matches": num_matches,
                "num_inliers": num_inliers,
                "inlier_ratio": inlier_ratio,
                "feature_ratio": feature_ratio,
                "match_ratio": match_ratio,
                "num_keypoints_target": len(target_output['keypoints']),
                "num_keypoints_scene": len(scene_output['keypoints']),
                "mkpts_0": mkpts_0,
                "mkpts_1": mkpts_1
            }

            print(debug_info)
            print(execution_time)

        if self.match_result_image_output_path:
            self.save_debug_image(self.target_image, self.scene_image, mkpts_0, mkpts_1, self.similarity)

class SimilarityScorer_video_analysis(SimilarityScorer):
    def __init__(self, video_input_path, video_output_path):
        super().__init__()

    def compare_images(self, target_image, scene_image):
        pass   
