from __future__ import print_function, absolute_import
import os
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
import random
import math
import cv2
import copy

def cross_modal_adain(content_image, style_image, alpha=0.5,prob=0.6):
    """
    MPB: transfer global channel statistics from style to content.

    Preserve the baseline grayscale, gamma and histogram postprocessing in
    both branches. This is statistical intervention, not RGB image synthesis.
    """
    if random.random() > prob:
        return content_image
    content_image = np.array(content_image)
    style_image = np.array(style_image)
    rgb = content_image.astype(np.float32) / 255.0
    ir = style_image.astype(np.float32) / 255.0

    if len(ir.shape) == 2:
        ir = np.stack([ir]*3, axis=-1)

    out = np.zeros_like(rgb)
    for c in range(3):
        mu_x, sigma_x = np.mean(rgb[:,:,c]), np.std(rgb[:,:,c])
        mu_y, sigma_y = np.mean(ir[:,:,c]), np.std(ir[:,:,c])
        out[:,:,c] = sigma_y * (rgb[:,:,c] - mu_x) / (sigma_x + 1e-6) + mu_y

    out = alpha * out + (1 - alpha) * rgb
    out = np.clip(out, 0, 1)

    gray = cv2.cvtColor((out * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
    gray = np.power(gray, 0.7)
    gray = cv2.equalizeHist((gray * 255).astype(np.uint8))
    ir_like = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    ir_like_pil = Image.fromarray(ir_like.astype(np.uint8))

    return ir_like_pil

def cross_modal_paste(base_img, aug_img, prob=0.5, min_area=0.1, max_area=0.3, alpha=0.5):
    """
    在 base_img (RGB) 上随机取一个区域，与 aug_img (IR) 对应区域相加融合
    - base_img, aug_img: PIL.Image
    - prob: 应用该增强的概率
    - min_area, max_area: 裁剪区域相对于图像的面积比例
    - alpha: 融合系数 (0=只保留RGB, 1=只保留IR)
    """
    if random.random() > prob:
        return base_img

    W, H = base_img.size
    aug_img = aug_img.resize((W, H), Image.BILINEAR)

    base_np = np.array(base_img).astype(np.float32)
    aug_np = np.array(aug_img).astype(np.float32)

    area = H * W
    target_area = random.uniform(min_area, max_area) * area
    aspect_ratio = random.uniform(0.5, 2.0)
    h = int(round((target_area * aspect_ratio) ** 0.5))
    w = int(round((target_area / aspect_ratio) ** 0.5))
    h = min(h, H)
    w = min(w, W)

    y1 = random.randint(0, H - h)
    x1 = random.randint(0, W - w)

    base_patch = base_np[y1:y1+h, x1:x1+w, :]
    aug_patch = aug_np[y1:y1+h, x1:x1+w, :]
    fused_patch = (1 - alpha) * base_patch + alpha * aug_patch

    base_np[y1:y1+h, x1:x1+w, :] = fused_patch

    base_img = Image.fromarray(np.uint8(np.clip(base_np, 0, 255)))
    return base_img

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

def read_text(text_path):
    with open(text_path, 'r') as file:
        content = file.read()
    return content

def tokenize(caption: str, tokenizer, text_length=77, truncate=True) -> torch.LongTensor:
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    tokens = [sot_token] + tokenizer.encode(caption) + [eot_token]

    result = torch.zeros(text_length, dtype=torch.long)
    if len(tokens) > text_length:
        if truncate:
            tokens = tokens[:text_length]
            tokens[-1] = eot_token
        else:
            raise RuntimeError(
                f"Input {caption} is too long for context length {text_length}"
            )
    result[:len(tokens)] = torch.tensor(tokens)
    return result

class VideoDataset_train_rgb(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset_rgb, seq_len=12, sample='evenly', transform1=None,transform2=None,transform3=None):

        self.dataset_rgb = dataset_rgb
        self.seq_len = seq_len
        self.sample = sample
        self.transform1 = transform1
        self.transform2 = transform2
        self.transform3 = transform3

        self.text_length = 77
        self.truncate = True

    def __len__(self):
        return len(self.dataset_rgb)

    def __getitem__(self, index):
        img_rgb_paths,pid_rgb,camid_rgb = self.dataset_rgb[index]
        num_rgb = len(img_rgb_paths)

        S = self.seq_len
        sample_clip_rgb = []
        frame_indices_rgb = list(range(num_rgb))
        if num_rgb < S:
            strip_rgb = list(range(num_rgb)) + [frame_indices_rgb[-1]] * (S - num_rgb)
            for s in range(S):
                pool_rgb = strip_rgb[s * 1:(s + 1) * 1]
                sample_clip_rgb.append(list(pool_rgb))
        else:
            inter_val_rgb = math.ceil(num_rgb / S)
            strip_rgb = list(range(num_rgb)) + [frame_indices_rgb[-1]] * (inter_val_rgb * S - num_rgb)
            for s in range(S):
                pool_rgb = strip_rgb[inter_val_rgb * s:inter_val_rgb * (s + 1)]
                sample_clip_rgb.append(list(pool_rgb))

        sample_clip_rgb = np.array(sample_clip_rgb)

        if self.sample == 'video_train':

            idx2 = np.random.choice(sample_clip_rgb.shape[1], sample_clip_rgb.shape[0])
            number_rgb = sample_clip_rgb[np.arange(len(sample_clip_rgb)), idx2]
            imgs_rgb = []

            idx = random.randint(0, 1)
            for index in number_rgb:
                index = int(index)
                img_path = img_rgb_paths[index]
                img = read_image(img_path)
                if self.transform1 is not None and self.transform2 is not None:
                    if self.transform1 is not None and self.transform2 is not None:
                        if idx==0:
                            img_trans = self.transform1(img)
                        elif idx==1:
                            img_trans = self.transform2(img)

                imgs_rgb.append(img_trans)
            imgs_rgb = torch.cat(imgs_rgb, dim=0)

            return imgs_rgb,pid_rgb

        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))

class VideoDataset_train_ir(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset_ir, seq_len=12, sample='evenly', transform1=None,transform2=None,transform3=None):
        self.dataset_ir = dataset_ir

        self.seq_len = seq_len
        self.sample = sample
        self.transform1 = transform1
        self.transform2 = transform2
        self.transform3 = transform3

        self.text_length = 77
        self.truncate = True

    def __len__(self):
        return len(self.dataset_ir)

    def __getitem__(self, index):

        img_ir_paths, pid_ir, camid_ir = self.dataset_ir[index]

        num_ir = len(img_ir_paths)

        S = self.seq_len
        sample_clip_ir = []
        frame_indices_ir = list(range(num_ir))
        if num_ir < S:
            strip_ir = list(range(num_ir)) + [frame_indices_ir[-1]] * (S - num_ir)
            for s in range(S):
                pool_ir = strip_ir[s * 1:(s + 1) * 1]
                sample_clip_ir.append(list(pool_ir))
        else:
            inter_val_ir = math.ceil(num_ir / S)
            strip_ir = list(range(num_ir)) + [frame_indices_ir[-1]] * (inter_val_ir * S - num_ir)
            for s in range(S):
                pool_ir = strip_ir[inter_val_ir * s:inter_val_ir * (s + 1)]
                sample_clip_ir.append(list(pool_ir))

        sample_clip_ir = np.array(sample_clip_ir)

        if self.sample == 'video_train':
            idx1 = np.random.choice(sample_clip_ir.shape[1], sample_clip_ir.shape[0])
            number_ir = sample_clip_ir[np.arange(len(sample_clip_ir)), idx1]

            imgs_ir = []

            for index in number_ir:
                index = int(index)
                img_path = img_ir_paths[index]
                img = read_image(img_path)
                if self.transform3 is not None:
                    img = self.transform3(img)

                imgs_ir.append(img)
            imgs_ir = torch.cat(imgs_ir, dim=0)

            return imgs_ir,pid_ir

        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))

class VisibleInterventionDataset(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset_rgb,dataset_ir, seq_len=12, sample='evenly', transform1=None,transform2=None,transform3=None, index1=[], index2=[]):

        self.dataset_rgb = dataset_rgb
        self.dataset_ir = dataset_ir
        self.seq_len = seq_len
        self.sample = sample
        self.transform1 = transform1
        self.transform2 = transform2
        self.transform3 = transform3
        self.index1 = index1
        self.index2 = index2

        self.text_length = 77
        self.truncate = True

    def __len__(self):
        return len(self.dataset_rgb)

    def __getitem__(self, index):
        img_rgb_paths,pid_rgb,camid_rgb = self.dataset_rgb[index]
        num_rgb = len(img_rgb_paths)

        rand_ir_index = random.randint(0, len(self.dataset_ir) - 1)
        img_ir_paths, _, _ = self.dataset_ir[rand_ir_index]

        S = self.seq_len
        sample_clip_rgb = []
        frame_indices_rgb = list(range(num_rgb))
        if num_rgb < S:
            strip_rgb = list(range(num_rgb)) + [frame_indices_rgb[-1]] * (S - num_rgb)
            for s in range(S):
                pool_rgb = strip_rgb[s * 1:(s + 1) * 1]
                sample_clip_rgb.append(list(pool_rgb))
        else:
            inter_val_rgb = math.ceil(num_rgb / S)
            strip_rgb = list(range(num_rgb)) + [frame_indices_rgb[-1]] * (inter_val_rgb * S - num_rgb)
            for s in range(S):
                pool_rgb = strip_rgb[inter_val_rgb * s:inter_val_rgb * (s + 1)]
                sample_clip_rgb.append(list(pool_rgb))

        sample_clip_rgb = np.array(sample_clip_rgb)

        if self.sample == 'video_train':

            idx2 = np.random.choice(sample_clip_rgb.shape[1], sample_clip_rgb.shape[0])
            number_rgb = sample_clip_rgb[np.arange(len(sample_clip_rgb)), idx2]
            imgs_rgb = []
            imgs_rgb_dos = []

            for index in number_rgb:
                idx = random.randint(0, 1)
                index = int(index)
                img_path = img_rgb_paths[index]
                img = read_image(img_path)
                img_do = copy.deepcopy(img)

                rand_ir_frame = random.randint(0, len(img_ir_paths) - 1)
                ir_img = read_image(img_ir_paths[rand_ir_frame])  # 原始 IR
                img_do = cross_modal_adain(img_do, ir_img, alpha=0.5,prob=0.6)

                if idx==0:
                    img_dos = self.transform1(img_do)
                    img = self.transform1(img)
                elif idx==1:
                    img_dos = self.transform2(img_do)
                    img = self.transform2(img)

                imgs_rgb_dos.append(img_dos)
                imgs_rgb.append(img)

            imgs_rgb_dos = torch.cat(imgs_rgb_dos, dim=0)
            imgs_rgb = torch.cat(imgs_rgb, dim=0)

            return imgs_rgb,imgs_rgb_dos,pid_rgb

        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))

class InfraredInterventionDataset(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset_ir,dataset_rgb, seq_len=12, sample='evenly', transform1=None,transform2=None,transform3=None, index1=[], index2=[]):
        self.dataset_ir = dataset_ir
        self.dataset_rgb = dataset_rgb

        self.seq_len = seq_len
        self.sample = sample
        self.transform1 = transform1
        self.transform2 = transform2
        self.transform3 = transform3
        self.index1 = index1
        self.index2 = index2

        self.text_length = 77
        self.truncate = True

    def __len__(self):
        return len(self.dataset_ir)

    def __getitem__(self, index):

        img_ir_paths, pid_ir, camid_ir = self.dataset_ir[index]

        num_ir = len(img_ir_paths)

        rand_rgb_index = random.randint(0, len(self.dataset_rgb) - 1)
        img_rgb_paths, _, _ = self.dataset_rgb[rand_rgb_index]

        S = self.seq_len
        sample_clip_ir = []
        frame_indices_ir = list(range(num_ir))
        if num_ir < S:
            strip_ir = list(range(num_ir)) + [frame_indices_ir[-1]] * (S - num_ir)
            for s in range(S):
                pool_ir = strip_ir[s * 1:(s + 1) * 1]
                sample_clip_ir.append(list(pool_ir))
        else:
            inter_val_ir = math.ceil(num_ir / S)
            strip_ir = list(range(num_ir)) + [frame_indices_ir[-1]] * (inter_val_ir * S - num_ir)
            for s in range(S):
                pool_ir = strip_ir[inter_val_ir * s:inter_val_ir * (s + 1)]
                sample_clip_ir.append(list(pool_ir))

        sample_clip_ir = np.array(sample_clip_ir)

        if self.sample == 'video_train':
            idx1 = np.random.choice(sample_clip_ir.shape[1], sample_clip_ir.shape[0])
            number_ir = sample_clip_ir[np.arange(len(sample_clip_ir)), idx1]

            imgs_ir = []
            imgs_ir_dos = []

            for index in number_ir:
                index = int(index)
                img_path = img_ir_paths[index]
                img = read_image(img_path)
                img_do = copy.deepcopy(img)

                rand_rgb_frame = random.randint(0, len(img_rgb_paths) - 1)
                rgb_img = read_image(img_rgb_paths[rand_rgb_frame])  # 原始 IR
                img_do = cross_modal_adain(img_do, rgb_img, alpha=0.5,prob=0.6)

                img = self.transform3(img)
                img_do = self.transform3(img_do)

                imgs_ir.append(img)
                imgs_ir_dos.append(img_do)
            imgs_ir = torch.cat(imgs_ir, dim=0)
            imgs_ir_dos = torch.cat(imgs_ir_dos, dim=0)

            return imgs_ir,imgs_ir_dos,pid_ir

        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))

class VideoDataset_train(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset_ir,dataset_rgb, seq_len=12, sample='evenly', transform1=None,transform2=None,transform3=None, index1=[], index2=[]):
        self.dataset_ir = dataset_ir
        self.dataset_rgb = dataset_rgb
        self.seq_len = seq_len
        self.sample = sample
        self.transform1 = transform1
        self.transform2 = transform2
        self.transform3 = transform3
        self.index1 = index1
        self.index2 = index2

        self.text_length = 77
        self.truncate = True

    def __len__(self):
        return len(self.dataset_rgb)

    def __getitem__(self, index):
        ir_index = self.index2[index]
        rgb_index = self.index1[index]
        img_ir_paths, pid_ir, camid_ir = self.dataset_ir[self.index2[index]]

        num_ir = len(img_ir_paths)

        img_rgb_paths,pid_rgb,camid_rgb = self.dataset_rgb[self.index1[index]]
        num_rgb = len(img_rgb_paths)

        S = self.seq_len
        sample_clip_ir = []
        frame_indices_ir = list(range(num_ir))
        if num_ir < S:
            strip_ir = list(range(num_ir)) + [frame_indices_ir[-1]] * (S - num_ir)
            for s in range(S):
                pool_ir = strip_ir[s * 1:(s + 1) * 1]
                sample_clip_ir.append(list(pool_ir))
        else:
            inter_val_ir = math.ceil(num_ir / S)
            strip_ir = list(range(num_ir)) + [frame_indices_ir[-1]] * (inter_val_ir * S - num_ir)
            for s in range(S):
                pool_ir = strip_ir[inter_val_ir * s:inter_val_ir * (s + 1)]
                sample_clip_ir.append(list(pool_ir))

        sample_clip_ir = np.array(sample_clip_ir)

        sample_clip_rgb = []
        frame_indices_rgb = list(range(num_rgb))
        if num_rgb < S:
            strip_rgb = list(range(num_rgb)) + [frame_indices_rgb[-1]] * (S - num_rgb)
            for s in range(S):
                pool_rgb = strip_rgb[s * 1:(s + 1) * 1]
                sample_clip_rgb.append(list(pool_rgb))
        else:
            inter_val_rgb = math.ceil(num_rgb / S)
            strip_rgb = list(range(num_rgb)) + [frame_indices_rgb[-1]] * (inter_val_rgb * S - num_rgb)
            for s in range(S):
                pool_rgb = strip_rgb[inter_val_rgb * s:inter_val_rgb * (s + 1)]
                sample_clip_rgb.append(list(pool_rgb))

        sample_clip_rgb = np.array(sample_clip_rgb)

        if self.sample == 'random':
            """
            Randomly sample seq_len consecutive frames from num frames,
            if num is smaller than seq_len, then replicate items.
            This sampling strategy is used in training phase.
            """
            frame_indices = range(num_ir)
            rand_end = max(0, len(frame_indices) - self.seq_len - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.seq_len, len(frame_indices))

            indices = frame_indices[begin_index:end_index]
            indices = list(indices)
            for index in indices:
                if len(indices) >= self.seq_len:
                    break
                indices.append(index)
            indices=np.array(indices)
            imgs_ir = []
            for index in indices:
                index=int(index)
                img_path = img_ir_paths[index]
                img = read_image(img_path)

                if self.transform is not None:
                    img = self.transform(img)

                imgs_ir.append(img)
            imgs_ir = torch.cat(imgs_ir, dim=0)

            frame_indices = range(num_rgb)
            rand_end = max(0, len(frame_indices) - self.seq_len - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.seq_len, len(frame_indices))

            indices = frame_indices[begin_index:end_index]

            indices = list(indices)
            for index in indices:
                if len(indices) >= self.seq_len:
                    break

                indices.append(index)
            indices=np.array(indices)
            imgs_rgb = []
            for index in indices:
                index=int(index)
                img_path = img_rgb_paths[index]
                img = read_image(img_path)
                if self.transform is not None:
                    img = self.transform(img)

                imgs_rgb.append(img)
            imgs_rgb = torch.cat(imgs_rgb, dim=0)
            return imgs_ir, pid_ir, camid_ir, imgs_rgb, pid_rgb, camid_rgb
        elif self.sample == 'video_train':
            idx1 = np.random.choice(sample_clip_ir.shape[1], sample_clip_ir.shape[0])
            number_ir = sample_clip_ir[np.arange(len(sample_clip_ir)), idx1]

            imgs_ir = []

            for index in number_ir:
                index = int(index)
                img_path = img_ir_paths[index]
                img = read_image(img_path)
                if self.transform3 is not None:
                    img = self.transform3(img)

                imgs_ir.append(img)
            imgs_ir = torch.cat(imgs_ir, dim=0)

            idx2 = np.random.choice(sample_clip_rgb.shape[1], sample_clip_rgb.shape[0])
            number_rgb = sample_clip_rgb[np.arange(len(sample_clip_rgb)), idx2]
            imgs_rgb = []

            idx = random.randint(0, 1)
            for index in number_rgb:
                index = int(index)
                img_path = img_rgb_paths[index]
                img = read_image(img_path)
                if self.transform1 is not None and self.transform2 is not None:
                    if self.transform1 is not None and self.transform2 is not None:
                        if idx==0:
                            img_trans = self.transform1(img)
                        elif idx==1:
                            img_trans = self.transform2(img)

                imgs_rgb.append(img_trans)
            imgs_rgb = torch.cat(imgs_rgb, dim=0)

            return imgs_rgb,imgs_ir,pid_rgb,pid_ir,rgb_index,ir_index

        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))

class VideoDataset_test(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset, seq_len=12, sample='evenly', transform=None):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)

        S = self.seq_len
        sample_clip_ir = []
        frame_indices_ir = list(range(num))
        if num < S:
            strip_ir = list(range(num)) + [frame_indices_ir[-1]] * (S - num)
            for s in range(S):
                pool_ir = strip_ir[s * 1:(s + 1) * 1]
                sample_clip_ir.append(list(pool_ir))
        else:
            inter_val_ir = math.ceil(num / S)
            strip_ir = list(range(num)) + [frame_indices_ir[-1]] * (inter_val_ir * S - num)
            for s in range(S):
                pool_ir = strip_ir[inter_val_ir * s:inter_val_ir * (s + 1)]
                sample_clip_ir.append(list(pool_ir))

        sample_clip_ir = np.array(sample_clip_ir)

        if self.sample == 'dense':
            """
            Sample all frames in a video into a list of clips, each clip contains seq_len frames, batch_size needs to be set to 1.
            This sampling strategy is used in test phase.
            """
            cur_index=0
            frame_indices = range(num)
            indices_list=[]
            while num-cur_index > self.seq_len:
                indices_list.append(frame_indices[cur_index:cur_index+self.seq_len])
                cur_index+=self.seq_len
            last_seq=frame_indices[cur_index:]
            last_seq = list(last_seq)
            for index in last_seq:
                if len(last_seq) >= self.seq_len:
                    break
                last_seq.append(index)
            indices_list.append(last_seq)
            imgs_list=[]
            for indices in indices_list:
                imgs = []
                for index in indices:
                    index=int(index)
                    img_path = img_paths[index]
                    img = read_image(img_path)

                    if self.transform is not None:
                        img = self.transform(img)
                    img = img.unsqueeze(0)
                    imgs.append(img)
                imgs = torch.cat(imgs, dim=0)

                imgs_list.append(imgs)
            imgs_array = torch.stack(imgs_list)
            return imgs_array, pid, camid

        if self.sample == 'random':
            """
            Randomly sample seq_len consecutive frames from num frames,
            if num is smaller than seq_len, then replicate items.
            This sampling strategy is used in training phase.
            """
            num_ir = len(img_paths)
            frame_indices = range(num_ir)
            rand_end = max(0, len(frame_indices) - self.seq_len - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.seq_len, len(frame_indices))

            indices = frame_indices[begin_index:end_index]
            indices = list(indices)
            for index in indices:
                if len(indices) >= self.seq_len:
                    break
                indices.append(index)
            indices = np.array(indices)
            imgs_ir = []
            for index in indices:
                index = int(index)
                img_path = img_paths[index]
                img = read_image(img_path)

                if self.transform is not None:
                    img = self.transform(img)

                imgs_ir.append(img)
            imgs_ir = torch.cat(imgs_ir, dim=0)
            return imgs_ir, pid, camid

        if self.sample == 'video_test':
            number = sample_clip_ir[:, 0]
            imgs_ir = []
            for index in number:
                index = int(index)
                img_path = img_paths[index]
                img = read_image(img_path)

                if self.transform is not None:
                    img = self.transform(img)

                imgs_ir.append(img)
            imgs_ir = torch.cat(imgs_ir, dim=0)
            return imgs_ir, pid, camid
        elif self.sample == 'video_cluster':
            number = sample_clip_ir[:, 0]
            imgs_ir = []
            for index in number:
                index = int(index)
                img_path = img_paths[index]
                img = read_image(img_path)

                if self.transform is not None:
                    img = self.transform(img)

                imgs_ir.append(img)
            imgs_ir = torch.cat(imgs_ir, dim=0)
            return imgs_ir,img_paths[0], pid, camid
        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))

class VideoDataset_train_evaluation(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset_ir, seq_len=12, sample='evenly', transform=None):
        self.dataset_ir = dataset_ir
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform

    def __len__(self):
        return len(self.dataset_ir)

    def __getitem__(self, index):

        img_ir_paths, pid_ir, camid_ir = self.dataset_ir[index]

        num_ir = len(img_ir_paths)

        if self.sample == 'random':
            """
            Randomly sample seq_len consecutive frames from num frames,
            if num is smaller than seq_len, then replicate items.
            This sampling strategy is used in training phase.
            """
            frame_indices = range(num_ir)
            rand_end = max(0, len(frame_indices) - self.seq_len - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.seq_len, len(frame_indices))

            indices = frame_indices[begin_index:end_index]
            indices = list(indices)
            for index in indices:
                if len(indices) >= self.seq_len:
                    break
                indices.append(index)
            indices=np.array(indices)
            imgs_ir = []
            for index in indices:
                index=int(index)
                img_path = img_ir_paths[index]
                img = read_image(img_path)

                if self.transform is not None:
                    img = self.transform(img)

                imgs_ir.append(img)
            imgs_ir = torch.cat(imgs_ir, dim=0)

            return imgs_ir, pid_ir, camid_ir
        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))
