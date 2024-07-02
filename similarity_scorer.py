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


from scipy.ndimage import gaussian_filter
from PIL import Image

import json
import queue
import time

import traceback
import threading
import signal


from tqdm import tqdm

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
        target_tensor = torch.from_numpy(target_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        scene_tensor = torch.from_numpy(scene_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
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

# Class for matching live video stream
class SimilarityScorer_realtime(SimilarityScorer):
    def __init__(self, debug_print=False, match_result_image_output_path=None, match_result_image_save_hz=0.25):
        try:
            super().__init__()

            
            self.target_image = None
            self.scene_image = None 


            self.image_save_queue = queue.Queue()
            self.stop_worker = False
            self.last_save_time = 0


            # Check for debugging args
            if self.debug_print or self.match_result_image_output_path:

                self.debug_print = debug_print

                # Start thread for saving debug images
                if self.match_result_image_output_path:
                    self.save_thread = threading.Thread(target=self.save_matching_results_worker, daemon=True)
                    self.save_thread.start()

                    self.match_result_image_output_path = match_result_image_output_path
                    self.match_result_image_save_hz = match_result_image_save_hz

                self.__run_debug_loop()

            # Normal loop
            else:

                self.__run_realtime_loop()

        except Exception as e:
            raise e

    def __run_realtime_loop(self):
        while True:
            if not self.__get_images_from_memory_buffer():
                break
            if self.target_image is not None and self.scene_image is not None:
                self.__compare_images_realtime()
                sys.stdout.buffer.write(f"{self.similarity:.6f}\n".encode('utf-8'))
                sys.stdout.buffer.flush()

    def __run_debug_loop(self):
        while True:
            if not self.__get_images_from_memory_buffer():
                break
            if self.target_image is not None and self.scene_image is not None:
                self.similarity, debug_info = self.__compare_images_debug()
                sys.stdout.buffer.write(f"{self.similarity:.6f}\n".encode('utf-8'))
                sys.stdout.buffer.flush()
                
                if self.debug_print:
                    print(debug_info)
                
                if self.match_result_image_output_path:
                    current_time = time.time()
                    if current_time - self.last_save_time >= 1 / self.match_result_image_save_hz:
                        self.image_save_queue.put((self.target_image, self.scene_image, debug_info['mkpts_0'], debug_info['mkpts_1'], self.similarity))
                        self.last_save_time = current_time

    def __get_images_from_memory_buffer(self):
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


    def __compare_images_realtime(self):
        # Method to get the similarity score of two images. Uses inherited methods.
        target_output, scene_output, num_matches, num_inliers, inlier_ratio, feature_ratio, match_ratio, mkpts_0, mkpts_1 = self.calculate_matching_scores(self.target_image, self.scene_image)
        self.set_similarity_score(inlier_ratio, feature_ratio, match_ratio)

    def __compare_images_debug(self):
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
                "execution_time": execution_time,
                "mkpts_0": mkpts_0,
                "mkpts_1": mkpts_1
            }
            
            print(f"Execution time: {execution_time:.4f} seconds")

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

class SimilarityScorer_video_analysis(SimilarityScorer):
    def __init__(self, 
                 mode=None,
                 video_input_path= None, 
                 video_output_path=None,
                 metadata_output_path=None, 
                 target_img_path=None, 
                 debug_print=False, 
                 frame_skip=5, 
                 metadata_input_path=None
                 ):
        super().__init__()

        # Input args
        self.video_input_path = video_input_path
        self.target_img_path = target_img_path
        self.target_image = np.array(Image.open(self.target_img_path))
        self.metadata_input_path = metadata_input_path

        self.video_output_path = video_output_path
        self.metadata_output_path = metadata_output_path

        self.frame_skip = frame_skip
        self.debug_print = debug_print

        # Threading and queue setup
        self.stop_event = threading.Event()
        self.threads = []

        # 5 min * 60 sec/min * 30 frames/sec = 9000 frames
        max_queue_size = 10000
        self.frame_queue = queue.Queue(maxsize=max_queue_size)
        self.metadata_queue = queue.Queue(maxsize=max_queue_size)

        # Set up signal handling
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        try:
            if mode == "video_analysis":
                # Verify inputs
                if not os.path.exists(self.video_input_path):
                    raise FileNotFoundError(f"Video input path does not exist: {self.video_input_path}")

                if not os.path.exists(self.target_img_path):
                    raise FileNotFoundError(f"Target image path does not exist: {self.target_img_path}")

                if self.target_image is None or self.target_image.size == 0:
                    raise ValueError("Error loading target image")
                
                # Verify args for video analysis mode
                if not os.path.exists(self.video_output_path):
                    raise FileNotFoundError(f"Video output path does not exist: {self.video_output_path}")
                if not os.path.exists(self.metadata_output_path):
                    raise FileNotFoundError(f"Metadata output path does not exist: {self.metadata_output_path}")

                self.__run_video_analysis()

            elif mode == "render_analyzed_video":

                # Verify args for render analyzed video mode
                if not os.path.exists(self.metadata_input_path):
                    raise FileNotFoundError(f"Metadata input path does not exist: {self.metadata_input_path}")
                
                if not os.path.exists(self.video_output_path):
                    raise FileNotFoundError(f"Video output path does not exist: {self.video_output_path}")

                self.__render_analyzed_comparison()

            else:
                raise ValueError("Invalid mode selected")

        except Exception as e:
            error_message = f"Error in SimilarityScorer_video_analysis: {str(e)}\n"
            error_message += "Traceback:\n"
            error_message += traceback.format_exc()
            print(error_message)
            self.cleanup()
            raise


    def __run_video_analysis(self):
        print("Starting video analysis")
        self.threads = [
            threading.Thread(target=self.__slice_video_into_frames),
            threading.Thread(target=self.__process_frames),
            threading.Thread(target=self.__create_metadata_file)
        ]

        for thread in self.threads:
            thread.start()

        for thread in self.threads:
            thread.join()

        print("Video analysis completed")

    def __slice_video_into_frames(self):
        print("Slicing video into frames")
        try:
            cap = cv2.VideoCapture(self.video_input_path)
            frame_count = 0
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count % self.frame_skip == 0:
                    self.frame_queue.put((frame_count, frame))
                frame_count += 1
            cap.release()
        except Exception as e:
            print(f"Error in slicing video: {str(e)}")
        finally:
            self.frame_queue.put(None)  # Signal end of frames

    def __process_frames(self):
        print("Processing frames")
        if self.debug_print:
            start_time = time.time()
        try:
            while not self.stop_event.is_set():
                try:
                    item = self.frame_queue.get(timeout=1)
                    if item is None:
                        break
                    frame_count, frame = item
                    
                    # Perform similarity calculation
                    target_output, scene_output, num_matches, num_inliers, inlier_ratio, feature_ratio, match_ratio, mkpts_0, mkpts_1 = self.calculate_matching_scores(self.target_image, frame)
                    self.set_similarity_score(inlier_ratio, feature_ratio, match_ratio)

                    metadata = {
                        "frame_number": int(frame_count),
                        "similarity_score": float(self.similarity),
                        "num_matches": int(num_matches),
                        "num_inliers": int(num_inliers),
                        "inlier_ratio": float(inlier_ratio),
                        "feature_ratio": float(feature_ratio),
                        "match_ratio": float(match_ratio),
                        "keypoints_target": [[float(x) for x in point] for point in mkpts_0.tolist()] if mkpts_0 is not None else [],
                        "keypoints_frame": [[float(x) for x in point] for point in mkpts_1.tolist()] if mkpts_1 is not None else []
                    }

                    self.metadata_queue.put(metadata)

                    if self.debug_print:
                        print(f"Processed frame {frame_count}, similarity: {self.similarity:.4f}")
                except queue.Empty:
                    continue
        except Exception as e:
            print(f"Error in processing frames: {str(e)}")
        finally:
            if self.debug_print:
                execution_time = time.time() - start_time
                print(f"Frame processing time: {execution_time:.4f} seconds")

            self.metadata_queue.put(None)  # Signal end of processing
        
        print("Frame processing completed")

    def __create_metadata_file(self):
        print("Creating metadata file")
        all_metadata = []
        try:
            while True:
                try:
                    metadata = self.metadata_queue.get(timeout=1)
                    if metadata is None:
                        break
                    all_metadata.append(metadata)
                except queue.Empty:
                    if self.stop_event.is_set():
                        break
                    continue

            metadata_path = os.path.join(self.metadata_output_path, "metadata.json")
            
            # Ensure the output directory exists
            os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
            
            # Create the final list with video_input_path as the first item
            final_metadata = [
                {"video_input_path": self.video_input_path},
                {"target_image_path": self.target_img_path},
                {"frame_skip_rate": self.frame_skip}
            ] + all_metadata
            
            with open(metadata_path, 'w') as f:
                json.dump(final_metadata, f, indent=2)
            print(f"Metadata saved to {metadata_path}")
            print(f"Number of frames processed: {len(all_metadata)}")

        except Exception as e:
            print(f"Error in creating metadata file: {str(e)}")
            print(f"Metadata path attempted: {metadata_path}")
            print(f"Number of metadata entries: {len(all_metadata)}")
        
    
    def __read_metadata_file(self):
        # Read the metadata file and return the metadata dictionary
        try:
            with open(self.metadata_input_path, 'r') as f:
                metadata = json.load(f)
            return metadata
        except Exception as e:
            print(f"Error reading metadata file: {str(e)}")
            return None


    def __render_analyzed_comparison(self):
        try:
            metadata = self.__read_metadata_file()
            if metadata is None:
                raise ValueError("Error reading metadata file")

            # Extract information from the first item in metadata
            video_input_path = metadata[0]["video_input_path"]
            target_img_path = metadata[0]["target_image_path"]
            frame_skip = metadata[0]["frame_skip_rate"]

            # Check if the paths exist
            if not os.path.exists(video_input_path):
                raise FileNotFoundError(f"Video input path does not exist: {video_input_path}")
            if not os.path.exists(target_img_path):
                raise FileNotFoundError(f"Target image path does not exist: {target_img_path}")
            
            target_image = cv2.imread(target_img_path)
            if target_image is None:
                raise ValueError(f"Could not read target image from {target_img_path}")

            cap = cv2.VideoCapture(video_input_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            output_path = os.path.join(self.video_output_path, "analyzed_video.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            frame_queue = queue.Queue()

            for frame_data in tqdm(metadata[3:], desc="Rendering frames"):
                frame_number = frame_data["frame_number"]
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                if not ret:
                    break

                # Create heatmaps for both target and frame
                target_heatmap = self.__create_heatmap(target_image.shape[:2], frame_data["keypoints_target"])
                frame_heatmap = self.__create_heatmap(frame.shape[:2], frame_data["keypoints_frame"])
                
                # Apply heatmaps
                frame_with_heatmap = self.__apply_heatmap(frame, frame_heatmap)
                target_with_heatmap = self.__apply_heatmap(target_image, target_heatmap)

                # Resize target image for overlay
                target_small = cv2.resize(target_with_heatmap, (width // 4, height // 4))

                # Create final frame
                final_frame = frame_with_heatmap.copy()
                final_frame[10:10+target_small.shape[0], 10:10+target_small.shape[1]] = target_small

                # Add similarity score
                similarity_score = frame_data["similarity_score"]
                cv2.putText(final_frame, f"Similarity: {similarity_score:.2f}", (width - 200, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                frame_queue.put(final_frame)

            cap.release()

            # Write frames to video
            while not frame_queue.empty():
                out.write(frame_queue.get())

            out.release()
            print(f"Analyzed video saved to {output_path}")

        except Exception as e:
            raise

    def __create_heatmap(self, shape, keypoints):
        # Create a heatmap from the keypoints
        heatmap = np.zeros(shape, dtype=np.float32)
        for kp in keypoints:
            # Ensure the keypoint is within the image bounds
            x, y = int(kp[0]), int(kp[1])
            if 0 <= x < shape[1] and 0 <= y < shape[0]:
                heatmap[y, x] += 1
        # Apply Gaussian filter and normalize. A gaussian filter is applied to smooth the heatmap.
        heatmap = gaussian_filter(heatmap, sigma=10)

        # Normalize the heatmap
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

        return heatmap

    def __apply_heatmap(self, image, heatmap):
        # Apply the heatmap to the image
        heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)

        return cv2.addWeighted(image, 0.7, heatmap_color, 0.3, 0)
    
    def signal_handler(self, signum, frame):
        print("\nInterrupt received. Stopping video analysis...")
        self.cleanup()
        sys.exit(0)

    def cleanup(self):
        self.stop_event.set()
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=1)
        print("Cleanup complete")

    def __del__(self):
        self.cleanup()
