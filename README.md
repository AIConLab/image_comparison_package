# image_comparison_package
General scripts to compare image media


# Run
Below are all the arguments that can be passed to the script. 
```bash
python3 similarity_scorer.py \
    --use_case <use_case> \
    --print_debug \
    --match_result_image_output_path <path> \
    --match_result_image_save_hz <hz> \
    --video_input_path <path> \
    --video_output_path <path> \
    --target_img_path <path> \
    --scene_img_path <path>
```

For example for the static image comparison use case:
```bash
python3 similarity_scorer.py \
    --use_case image_comparison \
    --print_debug \
    --match_result_image_output_path <path> \
    --target_img_path <path> \
    --scene_img_path <path>
```