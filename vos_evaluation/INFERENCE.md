## Semi-supervised VOS Inference

The `vos_inference.py` script can be used to generate predictions for semi-supervised video object segmentation (VOS) evaluation on datasets such as [DAVIS](https://davischallenge.org/index.html), [MOSE](https://henghuiding.github.io/MOSE/) , [SA-V](https://ai.meta.com/datasets/segment-anything-video/) and our SeCVOS dataset.

After installing dependencies, it can be used as follows ([DAVIS 2017 dataset](https://davischallenge.org/davis2017/code.html) as an example). This script saves the prediction PNG files to the `--output_mask_dir`.
```bash
python vos_evaluation/vos_inference.py \
    --model_path path/to/sec/model \
    --base_video_dir /path-to-davis-2017/JPEGImages/480p \
    --input_mask_dir /path-to-davis-2017/Annotations/480p \
    --video_list_file /path-to-davis-2017/ImageSets/2017/val.txt \
    --track_object_appearing_later_in_video \
    --output_mask_dir ./outputs/davis_2017_pred_pngs
```
(replace `/path-to-davis-2017` with the path to DAVIS 2017 dataset)

To evaluate on the SA-V and our SeCVOS dataset with per-object PNG files for the object masks, we need to **add the `--per_obj_png_file` flag** as follows (using SeCVOS as an example). This script will also save per-object PNG files for the output masks under the `--per_obj_png_file` flag.
```bash
python vos_evaluation/vos_inference.py \
    --model_path path/to/sec/model \
    --base_video_dir /path-to-secvos/JPEGImages \
    --input_mask_dir /path-to-secvos/Annotations \
    --video_list_file /path-to-secvos/video_id.txt \
    --per_obj_png_file \
    --track_object_appearing_later_in_video \
    --output_mask_dir ./outputs/secvos_pred_pngs 
```
(replace `/path-to-secvos` with the path to SeCVOS)

Then, we can use the evaluation tools or servers for each dataset to get the performance of the prediction PNG files above.

Note: by default, the `vos_inference.py` script above assumes that all objects to track already appear on frame 0 in each video (as is the case in DAVIS, MOSE or SA-V). **For VOS datasets that don't have all objects to track appearing in the first frame (such as LVOS or YouTube-VOS), please add the `--track_object_appearing_later_in_video` flag when using `vos_inference.py`**.

Also, you can set the `--num_chunks`, `--chunk_id`, `--num_nodes`, and `--node_id` arguments to run inference in parallel across multiple GPUs or nodes to speed up the inference process.

## Referring VOS Inference

Detailed object descriptions can be found in `meta_expressions.json`, which follows the same format as [ReasonVOS](https://github.com/showlab/VideoLISA/blob/main/BENCHMARK.md). Our SeC does not support Referring VOS Inference. You can refer to the inference implementations of other models such as [VideoLISA](https://github.com/showlab/VideoLISA/) and [SAMWISE](https://github.com/ClaudiaCuttano/SAMWISE) for guidance.