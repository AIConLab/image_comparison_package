# image_comparison_package
General scripts to compare image media

## Installation

Clone the repo and ensure you initialize the submodules:
```bash
git clone https://github.com/AIConLab/image_comparison_package.git --recursive
```

For XFeat you will need to install the dependencies. failure to do so, or using the wrong version of the dependencies could result in slower performance or errors. 
(The following is directly from the XFeat README)
```bash
#CPU only, for GPU check in pytorch website the most suitable version to your gpu.
pip install torch==1.10.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
# CPU only for MacOS
# pip install torch==1.10.1 -f https://download.pytorch.org/whl/cpu/torch_stable.html

#Install dependencies for the demo
pip install opencv-contrib-python tqdm
```

# Run
Below are all the arguments that can be passed to the script. 
```bash
python3 main_similarity_scorer.py \
    --use_case <use_case> \
    --print_debug \
    --match_result_image_output_path <path> \
    --match_result_image_save_hz <hz> \
    --video_input_path <path> \
    --video_output_path <path> \
    --target_img_path <path> \
    --scene_img_path <path>
```

For example for the static image comparison use case using the test images in the `media` directory:
```bash
python3 main_similarity_scorer.py \
    --use_case image_comparison \
    --print_debug \
    --match_result_image_output_path media \
    --target_img_path media/target_image.jpg \
    --scene_img_path media/target_image.jpg
```

Example for creating a metadata file for a video file:
```bash
python3 main_similarity_scorer.py \
    --use_case video_analysis \
    --video_analysis_mode video_analysis \
    --match_result_image_output_path /home/jc/Videos/video_data \
    --video_input_path /home/jc/Videos/video_data/DJI_20240618075856_0003_S.MP4 \
    --video_output_path /home/jc/Videos/video_data \
    --target_img_path /home/jc/Videos/video_data/target.png \
    --metadata_output_path /home/jc/Videos/video_data \
    --print_debug

```

Example for rendering the analyzed video:
```bash
python3 main_similarity_scorer.py \
    --use_case video_analysis \
    --video_analysis_mode render_analyzed_video \
    --target_img_path /home/jc/Videos/video_data/target.png \
    --video_input_path /home/jc/Videos/video_data/DJI_20240618075856_0003_S.MP4 \
    --metadata_input_path /home/jc/Videos/video_data/metadata.json \
    --video_output_path /home/jc/Videos/video_data \
    --print_debug
```