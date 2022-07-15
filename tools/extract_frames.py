from pathlib import Path

import cv2
import os

from PIL import Image
from tqdm import trange


def extract_frames(video_path: Path):
    frames_save_root = video_path.parent / video_path.stem
    frames_save_root.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error opening video stream or file: {video_path}")
        return 0

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_bits = len(str(num_frames))

    for i in trange(num_frames):
        ret, frame = cap.read()
        if i % 10 != 0 or not ret:
            continue

        fp = frames_save_root / f'{video_path.stem}_{i:0>{num_bits}d}.jpg'
        # if not cv2.imwrite(str(fp), frame):
        # If cv2 cannot save images for Chinese character or other errors,
        # then use PIL to save images.
        im = Image.fromarray(frame[:, :, ::-1])  # bgr2rgb
        im.save(fp)


def extract_videos_frames(videos_root=Path(r'G:\Data\AD\video_20220512')):
    """
    Extract videos frames
    :param videos_root: video root
    :type videos_root: Path | str
    :return: None
    :rtype: None

    # video_20220512 provided by Chai Yanchong
    >>> extract_videos_frames()

    # videos from 192.168.2.18/Reolinkvideo_Out
    >>> extract_videos_frames(
            Path(r'G:/Data/AD/reolink/videos/Reolinkvideo_Out_UserVideos'))

    # videos from 192.168.2.18/ReolinkPR_Out_Keen
    >>> extract_videos_frames('G:/Data/AD/reolink/videos/ReolinkPR_Out_Keen')

    # test garbled
    >>> extract_videos_frames('G:/Data/AD/reolink/videos/garbled')
    """
    video_paths = [Path(root) / file
                   for root, dirs, files in os.walk(videos_root)
                   for file in files
                   if file[-4:].lower() in {'.avi', '.mov', '.mp4'}]
    for i, video_path in enumerate(video_paths):
        print(f'{i + 1} / {len(video_paths)} Extracting file: {video_path}')
        extract_frames(video_path)


def main():
    extract_videos_frames('G:/Data/AD/reolink/videos/ReolinkPR_Out_Keen')


if __name__ == '__main__':
    main()
