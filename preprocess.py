import os
import glob
import cv2
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial

def extract_frames(video_item, source_dir, target_dir):
    rel_path = os.path.relpath(video_item, source_dir)
    video_name = os.path.splitext(rel_path)[0]
    output_path = os.path.join(target_dir, video_name)
    
    if os.path.exists(output_path):
        return

    os.makedirs(output_path, exist_ok=True)
    
    cap = cv2.VideoCapture(video_item)
    frame_idx = 1
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        save_path = os.path.join(output_path, f"img_{frame_idx:05d}.jpg")
        cv2.imwrite(save_path, frame)
        frame_idx += 1
    
    cap.release()
    print(f"Processed: {video_name} ({frame_idx-1} frames)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default='data/hmdb51/videos', help='Source video path')
    parser.add_argument('--dst', type=str, default='data/hmdb51/rawframes', help='Target rawframe path')
    parser.add_argument('--workers', type=int, default=4, help='Number of worker processes')
    args = parser.parse_args()

    if not os.path.exists(args.src):
        print(f"Error: Source directory {args.src} does not exist.")
        return

    video_files = glob.glob(os.path.join(args.src, '*', '*.avi'))
    print(f"Found {len(video_files)} videos. Starting extraction...")

    func = partial(extract_frames, source_dir=args.src, target_dir=args.dst)
    
    with Pool(args.workers) as pool:
        pool.map(func, video_files)
    
    print("All done!")

if __name__ == '__main__':
    main()