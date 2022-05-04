# Originally found in https://github.com/lucidrains/DALLE-pytorch
from pathlib import Path
from random import randint, choice

import PIL
import argparse
import clip
import torch
import csv
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.io import VideoReader
from pytorch_lightning import LightningDataModule



class TextImageDataset(Dataset):
    def __init__(self,
                 folder: str,
                 annotation_file: str,
                 image_size=224,
                 resize_ratio=0.75,
                 shuffle=False,
                 custom_tokenizer=False,
                 frame_rate = 2,
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
        self.shuffle = shuffle
        self.frame_rate = frame_rate 
        path = Path(folder)

        self.video_files = [*path.glob('**/*.MP4')]
        self.total_frames_per_video =  {video:np.floor(VideoReader(video,'video').get_metadata()['video']['duration'])*self.frame_rate for video in self.video_files}
        self.total_frames = np.sum(list(self.total_frames_per_video.values())) * self.frame_rate
        self.cumsum_of_frames_per_video = np.cumsum(self.total_frames_per_video)

        # the image pre-processing
        self.resize_ratio = resize_ratio
        self.image_transform = T.Compose([
            T.Lambda(self.fix_img),
            T.RandomResizedCrop(image_size,
                                scale=(self.resize_ratio, 1.),
                                ratio=(1., 1.)),
            T.ToTensor(),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)) # why are these values hard-coded?
        ])

        # the tokenizer used for text
        self.custom_tokenizer = custom_tokenizer

        # the actions and timestamps per video
        self.timed_actions = load_timestamped_actions(annotation_file)

    def __len__(self):
        return self.total_frames

    def load_timestamped_actions(self, folder):
        video_ids = [os.path.basename(f).stem for f in self.video_files]
        with open(folder) as f:
            reader = csv.reader(f, delimiter=',')
            timed_actions = [[[row[4], row[5], row[8]] for row in reader if row[2]==video_id] for video_id in video_ids]
        return timed_actions

    def fix_img(self, img): # channel conversion if needed
        return img.convert('RGB') if img.mode != 'RGB' else img

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def __getitem__(self, ind):
        video_idx = np.argwhere(self.cumsum_of_frames_per_video > ind)[0] if ind > self.cumsum_of_frames_per_video[0] else 0
        video_id = self.video_files[video_idx]
    
        frames_modulus = ind - self.cumsum_of_frames_per_video[video_idx-1] if ind > self.cumsum_of_frames_per_video[0] else ind
        time = frames_modulus / self.frame_rate + np.random.uniform(low=-0.1, high=1.0)
    
        # load the video
        video = VideoReader(video_path, 'video')
        video.set_current_stream("video")
        
        frame = video.seek(time).next()
        image = frame['data']
        timestamp = frame['pts']

        # find the action in the timestamp
        descriptions = [a for (st,et,a) in self.timed_actions[video_idx] if time > st and time < et]
        description = choice(descriptions) if len(descriptions) > 0 else 'unspecified'

        tokenized_text = description if self.custom_tokenizer else clip.tokenize(description)[0]

        try:
            image_tensor = self.image_transform(frame)
        except (PIL.UnidentifiedImageError, OSError) as corrupt_image_exceptions:
            print(f"An exception occurred trying to load file {image_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        # Success
        return image_tensor, tokenized_text

class TextImageDataModule(LightningDataModule):
    def __init__(self,
                 folder: str,
                 annotation_file: str,
                 batch_size: int,
                 num_workers=0,
                 image_size=224,
                 resize_ratio=0.75,
                 shuffle=False,
                 custom_tokenizer=None
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
        self.folder =folder
        self.annotation_file = annotation_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.resize_ratio = resize_ratio
        self.shuffle = shuffle
        self.custom_tokenizer = custom_tokenizer
    
    @staticmethod
    def add_argparse_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--folder', type=str, required=True, help='directory of your training folder')
        parser.add_argument('--annotation_file', type=str, required=True, help='directory of your training folder')
        parser.add_argument('--batch_size', type=int, help='size of the batch')
        parser.add_argument('--num_workers', type=int, default=0, help='number of workers for the dataloaders')
        parser.add_argument('--image_size', type=int, default=224, help='size of the images')
        parser.add_argument('--resize_ratio', type=float, default=0.75, help='minimum size of images during random crop')
        parser.add_argument('--shuffle', type=bool, default=False, help='whether to use shuffling during sampling')
        return parser
    
    def setup(self, stage=None):
        self.dataset = TextImageDataset(self.folder, self.annotation_file, image_size=self.image_size, resize_ratio=self.resize_ratio, shuffle=self.shuffle, custom_tokenizer=not self.custom_tokenizer is None)
    
    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers, drop_last=True , collate_fn=self.dl_collate_fn)
    
    def dl_collate_fn(self, batch):
        if self.custom_tokenizer is None:
            return torch.stack([row[0] for row in batch]), torch.stack([row[1] for row in batch])
        else:
            return torch.stack([row[0] for row in batch]), self.custom_tokenizer([row[1] for row in batch], padding=True, truncation=True, return_tensors="pt")
