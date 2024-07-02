import sys
import numpy as np
from similarity_scorer import SimilarityScorer
import struct
import traceback

class SimilarityScorer_realtime(SimilarityScorer):
    def __init__(self):
        print("Initializing SimilarityScorer_realtime", file=sys.stderr)
        super().__init__()
        self.target_image = None
        self.scene_image = None
        self.__run_realtime_loop()

    def __run_realtime_loop(self):
        print("Starting realtime loop", file=sys.stderr)
        while True:
            try:
                if not self.__get_images():
                    continue
                if self.target_image is not None and self.scene_image is not None:
                    self.__compare_images_realtime()
                    self.__output_latest_data()
                    self.target_image = None  # Reset both images after comparison
                    self.scene_image = None
            

            except Exception as e:
                print(f"Error in run_realtime_loop: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)

    def __get_images(self):
        try:
            for _ in range(2):  # Get two images (target and scene)
                header = sys.stdin.buffer.read(12)  # 3 integers, 4 bytes each
                if not header:
                    return False
                h, w, c = struct.unpack('3i', header)
                
                image_size = h * w * c
                image_data = sys.stdin.buffer.read(image_size)
                
                if len(image_data) != image_size:
                    return False
                
                image = np.frombuffer(image_data, dtype=np.uint8).reshape((h, w, c))
                
                if self.target_image is None:
                    self.target_image = image
                else:
                    self.scene_image = image
            
            return True

        except Exception as e:
            print(f"Error in __get_images: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return False

    def __compare_images_realtime(self):
        try:
            _, _, num_matches, num_inliers, inlier_ratio, feature_ratio, match_ratio, self.target_keypoints, self.scene_keypoints = self.calculate_matching_scores(self.target_image, self.scene_image)
            self.set_similarity_score(inlier_ratio, feature_ratio, match_ratio)
        except Exception as e:
            print(f"Error in __compare_images_realtime: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    def __output_latest_data(self):
        try:
            sys.stdout.buffer.write(struct.pack('f', self.similarity))
            sys.stdout.buffer.write(struct.pack('I', len(self.target_keypoints)))
            for kp in self.target_keypoints:
                sys.stdout.buffer.write(struct.pack('ff', float(kp[0]), float(kp[1])))
            sys.stdout.buffer.write(struct.pack('I', len(self.scene_keypoints)))
            for kp in self.scene_keypoints:
                sys.stdout.buffer.write(struct.pack('ff', float(kp[0]), float(kp[1])))
            sys.stdout.buffer.flush()
        except Exception as e:
            print(f"Error in __output_latest_data: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
if __name__ == "__main__":
    SimilarityScorer_realtime()