"""
visualize_detector_output.py will output images with bounding-boxes,
and this script will use these images to make videos to check conveniently
"""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm.asyncio import tqdm


def make_video_with_root():
    root = Path(r'G:\Data\AD\reolink\videos\ReolinkPR_Out_Keen')
    dst = root / 'vis'
    dst.mkdir(parents=True, exist_ok=True)

    dirs = sorted(list(root.glob('**/md_ge*')))
    for img_dir in tqdm(dirs):
        img_dir: Path
        img_paths = sorted(list(img_dir.glob('*.jpg')))
        if not img_paths:
            continue

        dst_dir = dst / img_dir.name
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_path = dst_dir / f"{img_paths[0].stem.rsplit('_', 1)[0]}.mp4"
        fps = 3
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        h, w, _ = np.array(Image.open(img_paths[0])).shape  # noqa
        vw = cv2.VideoWriter(str(dst_path), fourcc, fps, (w, h))
        for img_path in img_paths:
            vw.write(np.array(Image.open(img_path))[:, :, ::-1])  # noqa
        vw.release()


def main():
    make_video_with_root()


if __name__ == '__main__':
    main()
