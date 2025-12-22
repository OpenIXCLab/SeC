import argparse
import os
from pathlib import Path
from multiprocessing import Pool

import cv2
import numpy as np
import tqdm

# from petrel_client.client import Client
# from io import BytesIO

# conf_path = '~/petreloss.conf'
# client = Client(conf_path)

def get_args_parser():
    parser = argparse.ArgumentParser(
        description="[SA-V Preprocessing] Extracting JPEG frames",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ------------
    # DATA
    # ------------
    data_parser = parser.add_argument_group(
        title="SA-V dataset data root",
        description="What data to load and how to process it.",
    )
    data_parser.add_argument(
        "--sav-vid-dir",
        type=str,
        required=True,
        help=("Where to find the SAV videos"),
    )
    data_parser.add_argument(
        "--sav-frame-sample-rate",
        type=int,
        default=4,
        help="Rate at which to sub-sample frames",
    )
    data_parser.add_argument(
        "--file-list-txt",
        type=str,
        default=None,
        help=("Path to a txt file containing list of video names"),
    )
    
    # ------------
    # OUTPUT
    # ------------
    output_parser = parser.add_argument_group(
        title="Setting for results output", description="Where and how to save results."
    )
    output_parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help=("Where to dump the extracted jpeg frames"),
    )

    # ------------
    # PARALLEL
    # ------------
    parallel_parser = parser.add_argument_group(
        title="Parallel processing settings",
        description="Control the number of processes and data splitting",
    )
    parallel_parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs to run",
    )
    parallel_parser.add_argument(
        "--chunk-index",
        type=int,
        default=0,
        help="Index of the data chunk to process",
    )
    parallel_parser.add_argument(
        "--n-chunks",
        type=int,
        default=1,
        help="Total number of data chunks",
    )

    return parser

def decode_video(video_path: str):
    assert os.path.exists(video_path)
    if video_path.startswith("s3://"):
        video = client.get(video_path)
        video = cv2.VideoCapture(BytesIO(video))
    else:
        video = cv2.VideoCapture(video_path)
    video_frames = []
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            video_frames.append(frame)
        else:
            break
    video.release()
    return video_frames

def extract_frames(video_path, sample_rate):
    frames = decode_video(video_path)
    return frames[::sample_rate]

def process_video(path_sample):
    path, sample_rate, save_root = path_sample
    frames = extract_frames(path, sample_rate)
    output_folder = os.path.join(save_root, Path(path).stem)
    # 扫描output文件夹中的文件数，如果相同则跳过
    if os.path.exists(output_folder):
        existing_files = len(os.listdir(output_folder))
        if existing_files == len(frames):
            # print(f"Skipping {path}, already processed.")
            return path
        else:
            print(f"补充{path}缺失的帧")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for fid, frame in enumerate(frames):
        frame_path = os.path.join(output_folder, f"{fid*sample_rate:05d}.jpg")
        cv2.imwrite(frame_path, frame)
    return path

def process_videos(video_paths, sample_rate, save_root, n_jobs):
    tasks = [(path, sample_rate, save_root) for path in video_paths]
    with Pool(n_jobs) as pool:
        for path in tqdm.tqdm(pool.imap_unordered(process_video, tasks), total=len(tasks)):
            # print(f"Processed: {path}")
            pass

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    sav_vid_dir = args.sav_vid_dir
    save_root = args.output_dir
    sample_rate = args.sav_frame_sample_rate
    n_jobs = args.n_jobs
    chunk_index = args.chunk_index
    n_chunks = args.n_chunks
    file_list_txt = args.file_list_txt
    # List all SA-V videos
    if file_list_txt is not None:
        with open(file_list_txt, "r") as f:
            mp4_files = f.readlines()
        mp4_files = [os.path.join(sav_vid_dir,"sav_" + f.strip()[4:7], f.strip() + ".mp4") for f in mp4_files]
    else:
        mp4_files = sorted([str(p) for p in Path(sav_vid_dir).glob("*.mp4")])
        # if args.random_sample:
        #     np.random.seed(0)
        #     mp4_files = np.random.choice(mp4_files, 15000, replace=False)
    print(f"Total videos found: {len(mp4_files)}")
    # Split the videos into chunks
    total_files = len(mp4_files)
    chunk_size = (total_files + n_chunks - 1) // n_chunks
    start_idx = chunk_index * chunk_size
    end_idx = min(start_idx + chunk_size, total_files)
    mp4_files_chunk = mp4_files[start_idx:end_idx]

    # print(f"Processing videos in: {sav_vid_dir}")
    print(f"Total files: {total_files}, Processing chunk {chunk_index + 1}/{n_chunks}, Files in chunk: {len(mp4_files_chunk)}")

    # Process videos in parallel
    process_videos(mp4_files_chunk, sample_rate, save_root, n_jobs)

    print(f"Saving outputs to {save_root}")
