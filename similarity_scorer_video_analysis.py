from similarity_scorer import SimilarityScorer

import os
import time
import sys


import threading
import queue
import signal
import traceback

import cv2
import numpy as np
import json

from PIL import Image
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

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
        try:
            total_frames = self.frame_queue.qsize()
            with tqdm(total=total_frames, desc="Processing frames", unit="frame") as pbar:
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

                        pbar.update(1)
                        pbar.set_postfix({"similarity": f"{self.similarity:.4f}"})

                    except queue.Empty:
                        continue
        except Exception as e:
            print(f"Error in processing frames: {str(e)}")
        finally:
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
            target_img_path = metadata[1]["target_image_path"]
            frame_skip = metadata[2]["frame_skip_rate"]

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