import sys
import os

# Get the path to the package directory
package_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, package_dir)

# Add the path to the accelerated_features directory
accelerated_features_dir = os.path.join(package_dir, 'libs', 'accelerated_features')
sys.path.insert(0, accelerated_features_dir)

# Add the modules directory to the path
modules_dir = os.path.join(accelerated_features_dir, 'modules')
sys.path.insert(0, modules_dir)

# Now you can import XFeat
from modules.xfeat import XFeat

from abc import ABC, abstractmethod

import torch
import cv2
import numpy as np

import time


# Base class for similarity scorers
class SimilarityScorer(ABC):

    xfeat = XFeat()
    def __init__(self, **kwargs):
        #Gives all implementations access to the XFeat class
        # Matching weights and similarity score
        self.weight_inlier = 0.4
        self.weight_feature = 0.3
        self.weight_match = 0.3
        self.similarity = 0.0

    def calculate_matching_scores(self, target_image=None, scene_image=None):
        # Calculate matching scores based on the target and scene images
        # We calculate the number of matches, number of inliers, inlier ratio, feature ratio, and match ratio
        # uses the XFeat class to detect and compute features, match features
        # Homography is calculated using OpenCV
        target_tensor = torch.from_numpy(target_image.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        scene_tensor = torch.from_numpy(scene_image.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
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


    def set_similarity_score(self, inlier_ratio, feature_ratio, match_ratio):
        # Set the similarity score based on the inlier ratio, feature ratio, and match ratio
        # If the inlier ratio is 1.0, feature ratio is 1.0, and the number of matches is the same for both images, the similarity score is 1.0
        """
            - Number of matches: This is the number of feature points that were successfully matched between the two images.
            - Number of inliers: These are the matched points that are consistent with the estimated homography (transformation between the images). For identical images, this should be equal to the number of matches.
            - Inlier ratio: This is the ratio of inliers to total matches. A value of 1.0 means all matches are consistent with the homography, which is expected for identical images.
            - Feature ratio: This is the ratio of the number of features in the image with fewer features to the image with more features. For identical images, this should be 1.0.
            - Number of keypoints in target image: This is the number of feature points detected in the target image.
        """
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
            save_path = os.path.join("media", self.match_result_image_output_path, f'matching_results_{time.time():.0f}.jpg')
            cv2.imwrite(save_path, new_img)

        except Exception as e:
            print(f"Error saving debug image: {e}")
            raise e