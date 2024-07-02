from similarity_scorer import SimilarityScorer

import numpy as np
from PIL import Image
import time


class SimilarityScorer_static_image_comparison(SimilarityScorer):
    def __init__(self, target_img_path, scene_img_path, debug_print=False, match_result_image_output_path=None):
        try:
            super().__init__()
            self.target_img_path = target_img_path
            self.scene_img_path = scene_img_path
            self.debug_print = debug_print
            self.match_result_image_output_path = match_result_image_output_path

            self.target_image = None
            self.scene_image = None
        
            self.__run_static_image_comparison()

        except Exception as e:
            raise e

    def __run_static_image_comparison(self):
        self.target_image = np.array(Image.open(self.target_img_path))
        self.scene_image = np.array(Image.open(self.scene_img_path))

        if self.target_image is None or self.scene_image is None:
            raise Exception("Error loading images")

        self.__compare_images_static()
        print(f"Similarity Score: {self.similarity:.4f}")

    def __compare_images_static(self):
        if self.debug_print:
            start_time = time.time()

        target_output, scene_output, num_matches, num_inliers, inlier_ratio, feature_ratio, match_ratio, mkpts_0, mkpts_1 = self.calculate_matching_scores(self.target_image, self.scene_image)
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
            print(f"Execution time: {execution_time:.4f} seconds")

        if self.match_result_image_output_path:
            self.save_debug_image(self.target_image, self.scene_image, mkpts_0, mkpts_1, self.similarity)
