import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image similarity scorer')
    
    parser.add_argument("--use_case", type=str, default=None, help="Use case to run (realtime, video_analysis, image_comparison)")

    # Shared arguments
    parser.add_argument('--print_debug', action='store_true', help='Print debug information to console')
    parser.add_argument('--match_result_image_output_path', type=str, default=None, help='Path to save the matching result image')
    parser.add_argument('--target_img_path', type=str, default=None, help='Path to the target image')

    # Realtime mode arguments
    parser.add_argument('--match_result_image_save_hz', type=float, default=0.25, help='Matching result image save frequency in Hz')

    # Video analysis mode arguments
    parser.add_argument('--video_analysis_mode', type=str, default=None, help='Mode to run the video analysis in (video_analysis, render_analyzed_video)')
    parser.add_argument('--video_input_path', type=str, default= None, help='Path to the video file')
    parser.add_argument('--video_output_path', type=str, default=None, help='Path to save the output video file.')
    parser.add_argument('--frame_skip', type=int, default=5, help='Number of frames to skip in the video')
    parser.add_argument('--metadata_output_path', type=str, default=None, help='Path to save the metadata output')
    parser.add_argument('--metadata_input_path', type=str, default=None, help='Path to the metadata input')



    # Image comparison mode arguments
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
                                                              match_result_image_save_hz=args.match_result_image_save_hz
                )
        elif args.use_case == 'video_analysis':
            if args.video_analysis_mode not in ['video_analysis', 'render_analyzed_video']:
                raise Exception("Please provide a valid video analysis mode (video_analysis, render_analyzed_video)")
            # Check for empty or args

            if not args.video_input_path or not args.video_output_path or not args.target_img_path:
                raise Exception("Please provide a video input path and a video output path")

            # Video anlysis mode must have video input and output paths, target image path, metadata output path, 
            if args.video_analysis_mode == 'video_analysis':
                if not args.metadata_output_path:
                    raise Exception("Please provide a metadata output path")
                
                else:
                    similarity_scorer = SimilarityScorer_video_analysis(mode='video_analysis',
                                                                        video_input_path=args.video_input_path, 
                                                                        video_output_path=args.video_output_path, 
                                                                        target_img_path=args.target_img_path,
                                                                        metadata_output_path=args.metadata_output_path,
                                                                        frame_skip=args.frame_skip,
                                                                        debug_print=args.print_debug
                                                                        )
            # Render analyzed video mode must have video input and output paths, metadata input path, target image path
            elif args.video_analysis_mode == 'render_analyzed_video':
                if not args.metadata_input_path:
                    raise Exception("Please provide a metadata input path")

                else:
                    similarity_scorer = SimilarityScorer_video_analysis(mode='render_analyzed_video',
                                                                        video_input_path=args.video_input_path, 
                                                                        video_output_path=args.video_output_path, 
                                                                        target_img_path=args.target_img_path,
                                                                        metadata_input_path=args.metadata_input_path,
                                                                        frame_skip=args.frame_skip,
                                                                        debug_print=args.print_debug
                                                                        )

        elif args.use_case == 'image_comparison':
            # Bad args check

            if not args.target_img_path or not args.scene_img_path:
                raise Exception("Please provide a target image path and a scene image path")

            

            else:
                similarity_scorer = SimilarityScorer_image_comparison(target_img_path=args.target_img_path, 
                                                                     scene_img_path=args.scene_img_path,
                                                                     debug_print=args.print_debug, 
                                                                     match_result_image_output_path=args.match_result_image_output_path
                )
        else:
            raise Exception("Invalid use case, please choose from realtime, video_analysis, image_comparison")

    except Exception as e:
        print(f"Error: {e}")
        exit(1)