import os
import glob
import json
import numpy as np
import cv2
from multiprocessing import Pool

# Config
RAWFRAMES_DIR = 'data/hmdb51/rawframes'
ANNOTATION_DIR = 'data/hmdb51/annotations'
OUTPUT_DIR = 'data/hmdb51'
SPLIT = 1

def calc_motion_score(video_path):
    frames = sorted(glob.glob(os.path.join(video_path, '*.jpg')))
    if len(frames) < 2:
        return None
    
    diff_scores = []
    prev_img = cv2.imread(frames[0], cv2.IMREAD_GRAYSCALE)
    
    for i in range(1, len(frames)):
        curr_img = cv2.imread(frames[i], cv2.IMREAD_GRAYSCALE)
        if curr_img is None: break
        
        diff = cv2.absdiff(curr_img, prev_img)
        score = np.sum(diff) / (diff.shape[0] * diff.shape[1])
        diff_scores.append(float(score))
        prev_img = curr_img
        
    rel_path = os.path.relpath(video_path, RAWFRAMES_DIR).replace('\\', '/')
    return rel_path, diff_scores

def main():
    print("1. Generating file lists...")
    classes = sorted(os.listdir(RAWFRAMES_DIR))
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    
    train_list = []
    val_list = []
    
    split_files = glob.glob(os.path.join(ANNOTATION_DIR, f'*_test_split{SPLIT}.txt'))
    
    for txt_file in split_files:
        action = os.path.basename(txt_file).replace(f'_test_split{SPLIT}.txt', '')
        if action not in class_to_idx: continue
        
        with open(txt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2: continue
                
                vid_name = os.path.splitext(parts[0])[0]
                split_id = parts[1]
                
                full_path = os.path.join(RAWFRAMES_DIR, action, vid_name)
                if not os.path.exists(full_path): continue
                
                num_frames = len(glob.glob(os.path.join(full_path, '*.jpg')))
                if num_frames == 0: continue
                
                entry = f"{action}/{vid_name} {num_frames} {class_to_idx[action]}"
                
                if split_id == '1':
                    train_list.append(entry)
                elif split_id == '2':
                    val_list.append(entry)

    with open(os.path.join(OUTPUT_DIR, 'train.txt'), 'w') as f:
        f.write('\n'.join(train_list))
    with open(os.path.join(OUTPUT_DIR, 'val.txt'), 'w') as f:
        f.write('\n'.join(val_list))
        
    print(f"Lists generated. Train: {len(train_list)}, Val: {len(val_list)}")
    
    print("2. Calculating motion scores (MGSampler)...")
    video_paths = []
    for root, dirs, files in os.walk(RAWFRAMES_DIR):
        if len(files) > 0 and files[0].endswith('.jpg'):
            video_paths.append(root)
            
    motion_data = {}
    with Pool(6) as pool:
        for res in pool.imap_unordered(calc_motion_score, video_paths):
            if res:
                motion_data[res[0]] = res[1]
                
    with open(os.path.join(OUTPUT_DIR, 'motion_diff.json'), 'w') as f:
        json.dump(motion_data, f)
    print("Motion JSON generated.")

if __name__ == '__main__':
    main()