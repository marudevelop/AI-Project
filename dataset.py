import os
import json
import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import config

class MGSamplerDataset(Dataset):
    def __init__(self, txt_file, root_dir, motion_file=None, transform=None, test_mode=False):
        self.root_dir = root_dir
        self.transform = transform
        self.test_mode = test_mode
        self.num_segments = config.NUM_SEGMENTS
        
        self.video_list = []
        with open(txt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                path = parts[0]
                num_frames = int(parts[1])
                label = int(parts[2])
                self.video_list.append((path, num_frames, label))
        
        self.motion_data = {}
        if motion_file and os.path.exists(motion_file):
            with open(motion_file, 'r') as f:
                self.motion_data = json.load(f)

    def __len__(self):
        return len(self.video_list)

    def _get_motion_weights(self, video_name, num_frames):
        key = video_name.replace('\\', '/')
        if key in self.motion_data:
            scores = np.array(self.motion_data[key])
            if len(scores) == 0:
                return np.ones(num_frames)
            
            if len(scores) < num_frames:
                pad = np.zeros(num_frames - len(scores))
                scores = np.concatenate((scores, pad))
            elif len(scores) > num_frames:
                scores = scores[:num_frames]
            
            scores = scores + 1e-5
            return scores
        else:
            return np.ones(num_frames)

    def _sample_indices(self, num_frames, weights):
        indices = []
        seg_len = num_frames / self.num_segments
        
        for i in range(self.num_segments):
            start = int(i * seg_len)
            end = int((i + 1) * seg_len)
            if start >= end:
                start = max(0, end - 1)
            
            seg_weights = weights[start:end]
            seg_weights = seg_weights / seg_weights.sum()
            
            if self.test_mode:
                idx = start + int(len(seg_weights) / 2)
            else:
                try:
                    idx = np.random.choice(range(start, end), p=seg_weights)
                except ValueError:
                    idx = np.random.randint(start, end)
            
            indices.append(idx + 1)
            
        return indices

    def __getitem__(self, idx):
        path, num_frames, label = self.video_list[idx]
        full_path = os.path.join(self.root_dir, path)
        
        weights = self._get_motion_weights(path, num_frames)
        indices = self._sample_indices(num_frames, weights)
        
        images = []
        for i in indices:
            img_name = f"img_{i:05d}.jpg"
            img_path = os.path.join(full_path, img_name)
            
            try:
                img = Image.open(img_path).convert('RGB')
            except:
                img = Image.new('RGB', (224, 224))
            
            if self.transform:
                img = self.transform(img)
            images.append(img)

        data = torch.stack(images)
        data = data.permute(1, 0, 2, 3)
        return data, label

def get_transforms(mode='train'):
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])