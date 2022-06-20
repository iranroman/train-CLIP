from __future__ import annotations
from pathlib import Path
from random import randint, choice

import os
import csv
import argparse

import PIL
import clip
import torch
import numpy as np
import pandas as pd

import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from pytorch_lightning import LightningDataModule

# from torchvision.io import VideoReader
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.io.ffmpeg_reader import ffmpeg_parse_infos



class VideoClips:
    def __init__(self, 
                 video_files: list, 
                 frame_rate: int, 
                 clip_size: int=1, 
                 video_metadata: list|None=None
        ):
        self.frame_rate = frame_rate
        self.clip_size = clip_size
        self.files = video_files
        self.video_metadata = video_metadata or [
            self._get_video_metadata(video)
            for video in tqdm.tqdm(self.files)
        ]
        self.durations_in_frames = [
            # np.floor(d['video']['duration']) * self.frame_rate
            np.floor(d['video_duration'] * (self.frame_rate or d['video_fps']))
            for d in self.video_metadata
        ]
        print(self.durations_in_frames)
        self.cumsum_frames = np.cumsum([
            d - (clip_size - 1)
            for d in self.durations_in_frames
        ]).astype(int)
        self.n_clips = self.cumsum_frames[-1] if len(self.cumsum_frames) else 0

    def subset(self, indices):
        return VideoClips(
            [self.files[i] for i in indices],
            frame_rate=self.frame_rate,
            clip_size=self.clip_size,
            video_metadata=[self.video_metadata[i] for i in indices],
        )

    def get_clip_location(self, idx: int):
        video_idx = np.argwhere(self.cumsum_frames > idx)[0] if idx > self.cumsum_frames[0] else 0
        start_idx = idx - (self.cumsum_frames[video_idx-1] if video_idx else 0)
        return int(video_idx), int(start_idx)

    def get_clip(self, idx: int, return_frames=False):
        video_idx, start_idx = self.get_clip_location(idx)
        path = self.files[video_idx]
        # fps = float(self.video_metadata[video_idx]['video']['fps'])
        fps = float(self.video_metadata[video_idx]['video_fps'])
        stop_idx = start_idx + 1. * self.clip_size * fps / (self.frame_rate or fps)

        video = self._get_video(path)
        # clip = self._get_video_segment(video, start_time, end_time, step)
        clip = self._get_video_segment(video, start_idx / fps, stop_idx / fps)
        bounds = (start_idx, stop_idx) if return_frames else (start_idx / fps, stop_idx / fps)
        return clip, video_idx, bounds, fps

    def _get_video(self, video_path):
        return VideoFileClip(video_path)

    def _get_video_segment(self, video, start, end, ):
        return np.array([
            x for _, x in zip(
                range(self.clip_size), 
                video.subclip(start, end).iter_frames(self.frame_rate))
        ])

    def _get_video_metadata(self, path):
        return ffmpeg_parse_infos(str(path))

    # def _get_video(self, path):
    #     # load the video
    #     video = VideoReader(str(path), 'video')
    #     video.set_current_stream("video")
    #     return video

    # def _get_video_segment(self, video, start_time, end_time, step):
    #     return torch.concat([
    #         video.seek(t).__next__()['data']
    #         for t in np.arange(start_time, end_time, step)
    #     ])

    # def _get_video_metadata(self, path):
    #     from torchvision.io import VideoReader
    #     return VideoReader(str(path),'video').get_metadata()


class EpicKitchensAnnotations:
    def __init__(self, ann_path: str, root_path: str, split: str='validation', filter_existing: bool=True, missing_text='unspecified') -> None:
        self.root_path = root_path
        self.missing_text = missing_text

        self.verb_classes = pd.read_csv(os.path.join(
            ann_path, 'EPIC_100_verb_classes.csv'), index_col='id')
        self.noun_classes = pd.read_csv(os.path.join(
            ann_path, 'EPIC_100_noun_classes.csv'), index_col='id')

        action_df = pd.read_csv(os.path.join(
            ann_path, f'EPIC_100_{split}.csv'))
        action_df['filename'] = action_df.apply(lambda d: os.path.join(
            self.root_path, d.participant_id, 'videos', f'{d.video_id}.MP4'), axis=1)
        if filter_existing:
            action_df = action_df[action_df.filename.apply(os.path.isfile)]
        if not len(action_df):
            raise ValueError(f"No actions available in {action_df}")
        self.df = action_df

        self.video_list = list(action_df.filename.unique())

    def get_range(self, video_path, start_frame, n_frames, step):
        video_path = self.video_list[video_path] if isinstance(video_path, int) else video_path
        ts = start_frame + np.arange(0, n_frames) * step
        df = self.df
        df = df[df.filename == video_path]  #.sort_values(by=['start_frame'])
        # print(ts)
        # print(df[['narration', 'start_frame', 'stop_frame']])
        # df = df[
        #     (df.filename == video_path) & 
        #     # (df.start_frame >= ts[0]) & 
        #     # (df.stop_frame <= ts[-1])
        # ]
        return [
            df[(df.start_frame <= t) & (df.stop_frame >= t)]
            for t in ts
        ]

    def get_descriptions(self, *a, **kw):
        return [
            ' and '.join(df.narration.tolist() or [self.missing_text])
            for df in self.get_range(*a, **kw)
        ]

class VideoTextDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 annotation_path: str,
                 image_size=224,
                 resize_ratio=0.75,
                 clip_size=1,
                 tokenizer=None,
                 frame_rate=2,
                 image_mode='RGB',
        ):
        """Create a text image dataset from a directory with congruent text and image names.

        Args:
            folder (str): Folder containing images and text files matched by their paths' respective "stem"
            annotation_file (str): file containing the action annotations 
            image_size (int, optional): The size of outputted images. Defaults to 224.
            resize_ratio (float, optional): Minimum percentage of image contained by resize. Defaults to 0.75.
            shuffle (bool, optional): Whether or not to have shuffling behavior during sampling. Defaults to False.
            custom_tokenizer (bool, optional): Whether or not there is a custom tokenizer. Defaults to False.
            frame_rate (int): the desired number of frames used per second. Defaults to 2
        """
        super().__init__()
        self.annotations = EpicKitchensAnnotations(annotation_path, data_path)
        self.videos = VideoClips(
            self.annotations.video_list, 
            frame_rate=frame_rate, 
            clip_size=clip_size,
        )

        # the image pre-processing
        # self.image_transform = T.Compose([
        #     # T.Lambda(lambda img: img.convert(image_mode) if img.mode != image_mode else img) if image_mode else None,
        #     # T.RandomResizedCrop(image_size, scale=(resize_ratio, 1.), ratio=(1., 1.)),
        #     T.ToTensor(),
        #     # T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)) # why are these values hard-coded?
        # ])
        self.tokenizer = tokenizer or self._tokenizer

    def __len__(self):
        return self.videos.n_clips

    def _tokenizer(self, text):
        return clip.tokenize(text)

    def __getitem__(self, idx: int):
        clip, video_idx, (start_idx, stop_idx), fps = self.videos.get_clip(idx, return_frames=True)
        descs = self.annotations.get_descriptions(
            video_idx, start_idx, len(clip), 
            1. * fps / (self.videos.frame_rate or fps))
        print(descs)
        # clip = self.image_transform(clip)
        clip = torch.tensor(clip)
        descs = self.tokenizer(descs)
        return clip, descs


class TextImageDataModule(LightningDataModule):
    def __init__(self,
                 folder: str,
                 annotation_path: str,
                 batch_size: int,
                 num_workers=0,
                 image_size=224,
                 resize_ratio=0.75,
                 shuffle=False,
                 clip_size=1,
                 custom_tokenizer=None,
        ):
        """Create a text image datamodule from directories with congruent text and image names.
        Args:
            folder (str): Folder containing images and text files matched by their paths' respective "stem"
            batch_size (int): The batch size of each dataloader.
            num_workers (int, optional): The number of workers in the DataLoader. Defaults to 0.
            image_size (int, optional): The size of outputted images. Defaults to 224.
            resize_ratio (float, optional): Minimum percentage of image contained by resize. Defaults to 0.75.
            shuffle (bool, optional): Whether or not to have shuffling behavior during sampling. Defaults to False.
            custom_tokenizer (transformers.AutoTokenizer, optional): The tokenizer to use on the text. Defaults to None.
        """
        super().__init__()
        self.folder = folder
        self.annotation_path = annotation_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.resize_ratio = resize_ratio
        self.shuffle = shuffle
        self.clip_size = clip_size
        self.custom_tokenizer = custom_tokenizer
    
    @staticmethod
    def add_argparse_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--folder', type=str, required=True, help='directory of your training folder')
        parser.add_argument('--annotation_path', type=str, required=True, help='directory of your training folder')
        parser.add_argument('--batch_size', type=int, help='size of the batch')
        parser.add_argument('--num_workers', type=int, default=0, help='number of workers for the dataloaders')
        parser.add_argument('--image_size', type=int, default=224, help='size of the images')
        parser.add_argument('--resize_ratio', type=float, default=0.75, help='minimum size of images during random crop')
        parser.add_argument('--shuffle', type=bool, default=False, help='whether to use shuffling during sampling')
        parser.add_argument('--clip_size', type=int, default=1, help='whether to use shuffling during sampling')
        return parser
    
    def setup(self, stage=None):
        self.dataset = VideoTextDataset(
            self.folder, self.annotation_path, 
            image_size=self.image_size, 
            resize_ratio=self.resize_ratio, 
            clip_size=self.clip_size,
            tokenizer=self.custom_tokenizer)
    
    def train_dataloader(self):
        return DataLoader(
            self.dataset, 
            batch_size=self.batch_size, shuffle=self.shuffle, 
            num_workers=self.num_workers, drop_last=True, 
            collate_fn=self.dl_collate_fn)
    
    def dl_collate_fn(self, batch):
        return [torch.stack(xs) for xs in zip(*batch)]


if __name__ == '__main__':
    # import cv2
    # import time
    def test(*a, **kw):
        dm = TextImageDataModule(*a, **kw)
        dm.setup()
        dl = dm.train_dataloader()
        for clips,  texts in tqdm.tqdm(dl):
            print(clips.shape, texts.shape)


    import fire
    fire.Fire(test)