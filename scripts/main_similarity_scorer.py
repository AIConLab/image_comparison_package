"""
Entry point for the image similarity scorer. This script takes in arguments to run the similarity scorer in different modes.
Use cases:
    - Real time mode: Take in a target image and compares it to images fed into a shared memory. Currently implemented for ROS system. Outputs similarity score.
        - (Optional) arg:print_debug: Print to console
        - (Optional) arg: match_result_image_output_path: Path to save the matching result image
            - arg: match_result_image_save_hz: Matching result image save frequency in Hz
    - Video Analysis mode: Take in a video file and compare the target image to each frame in the video, outputs metadata and a video with the matching results.
        - arg: video_input_path: Path to the video file
        - arg: video_output_path: Path to save the output video file.
    - Single image comparison mode: Take in a target image and a scene image, compare the two images and output metadata and a matching result image. Outputs similarity score.
        - arg: target_img_path: Path to the target image
        - arg: scene_img_path: Path to the scene image
        - (Optional) arg:print_debug: Print to console
        - (Optional) arg: match_result_image_output_path: Path to save the matching result image
"""

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image similarity scorer')
    
    parser.add_argument("--use_case", type=str, default=None, help="Use case to run (realtime, video_analysis, image_comparison)")

    # Shared arguments
    parser.add_argument('--print_debug', action='store_true', help='Print debug information to console')
    parser.add_argument('--match_result_image_output_path', type=str, default=None, help='Path to save the matching result image')

    # Realtime mode arguments
    parser.add_argument('--match_result_image_save_hz', type=float, default=0.25, help='Matching result image save frequency in Hz')

    # Video analysis mode arguments
    parser.add_argument('--video_input_path', type=str, default= None, help='Path to the video file')
    parser.add_argument('--video_output_path', type=str, default=None, help='Path to save the output video file.')

    # Image comparison mode arguments
    parser.add_argument('--target_img_path', type=str, default=None, help='Path to the target image')
    parser.add_argument('--scene_img_path', type=str, default=None, help='Path to the scene image')

    args = parser.parse_args()

    from similarity_scorer import SimilarityScorer_realtime, SimilarityScorer_video_analysis, SimilarityScorer_image_comparison

    try:
        if args.use_case == 'realtime':
            # Check for hz if output path is provided
            if args.match_result_image_output_path and not args.match_result_image_save_hz:
                raise Exception("Please provide a matching result image save frequency in Hz")
            else:
                similarity_scorer = SimilarityScorer_realtime(debug_print=args.print_debug, 
                                                              match_result_image_output_path=args.match_result_image_output_path, 
                                                              match_result_image_save_hz=args.match_result_image_save_hz)
        elif args.use_case == 'video_analysis':
            # Check for empty or args
            if not args.video_input_path or not args.video_output_path:
                raise Exception("Please provide a video input path and a video output path")
            else:
                similarity_scorer = SimilarityScorer_video_analysis(video_input_path=args.video_input_path, 
                                                                   video_output_path=args.video_output_path)

        elif args.use_case == 'image_comparison':
            # Check for empty or args
            if not args.target_img_path or not args.scene_img_path:
                raise Exception("Please provide a target image path and a scene image path")
            else:
                similarity_scorer = SimilarityScorer_image_comparison(target_img_path=args.target_img_path, 
                                                                     scene_img_path=args.scene_img_path,
                                                                     debug_print=args.print_debug, 
                                                                     match_result_image_output_path=args.match_result_image_output_path)
        else:
            raise Exception("Invalid use case, please choose from realtime, video_analysis, image_comparison")

    except Exception as e:
        print(f"Error: {e}")
        exit(1)