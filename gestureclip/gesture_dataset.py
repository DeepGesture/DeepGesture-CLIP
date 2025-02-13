import argparse
import random
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import lightning as L
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

meta_data_structure = {
    'agent': None,
    'motion_filename': None,
    'motion': None,
    'original_text': None,
    'text_filename': None,
    'text': None,
    'audio': None,
    'audio_filename': None,
}


def remove_none(d):
    return {k: v for k, v in d.items() if v is not None}


class MotionTextDataset(Dataset):
    AGENTS_AVAILABLE = ['main-agent', 'interloctr']
    FILEEXT = {
        'motion': 'bvh_expmap_30fps.pkl',
        'text': 'pkl',
        'audio': 'pkl'
    }
    FOLDERNAME = {
        'motion': 'diffusion_genea23_sm0_0_30fps',
        'text': 'data2vec-text',
        'audio': 'data2vec-audio'
    }

    error_files = (
        'trn_2023_v0_139_main-agent.bvh_expmap_30fps.pkl'
        'trn_2023_v0_181_interloctr.bvh_expmap_30fps.pkl'
    )

    def __init__(self, data_folder, agents=None, modules_to_load=['motion', 'text', 'audio'], mirrored=True) -> None:
        super().__init__()
        if isinstance(data_folder, str):
            data_folder = Path(data_folder)
            assert data_folder.exists(), f'Folder {data_folder} does not exist'

        if agents is None:
            agents = self.AGENTS_AVAILABLE

        self.data_folder = data_folder
        self.modules_to_load = modules_to_load

        self.data_files = []
        for agent in agents:
            for file in (self.data_folder / agent / self.FOLDERNAME['motion']).glob(f'*.{self.FILEEXT["motion"]}'):
                if file.name in self.error_files or file.name.replace('_mirrored', '') in self.error_files:
                    continue
                if 'mirrored' in file.name and not mirrored:
                    continue

                data_point = deepcopy(meta_data_structure)
                data_point['agent'] = agent
                file_name = file.with_suffix('').with_suffix('').name
                if 'motion' in modules_to_load:
                    data_point['motion_filename'] = file

                if mirrored:
                    file_name = file_name.replace('_mirrored', '')

                if 'text' in modules_to_load:
                    data_point['text_filename'] = self.data_folder / agent / self.FOLDERNAME['text'] / f"{file_name}.{self.FILEEXT['text']}"

                if 'audio' in modules_to_load:
                    data_point['audio_filename'] = self.data_folder / agent / self.FOLDERNAME['audio'] / f"{file_name}.{self.FILEEXT['audio']}"

                self.data_files.append(remove_none(data_point))

        random.shuffle(self.data_files)

    def __len__(self) -> int:
        return len(self.data_files)

    def load_motion(self, motion_filename):
        x = pd.read_pickle(motion_filename)
        return torch.from_numpy(x.values).float()

    def load_audio(self, audio_filename):
        x = pd.read_pickle(audio_filename)
        return torch.from_numpy(x.values).float()

    def load_text(self, text_filename):
        x = pd.read_pickle(text_filename)
        return torch.from_numpy(x.values).float()

    def ensure_same_length(self, data_point):
        def interpolate(x, n):
            """Interpolate on first dimension with F.interpolate"""
            return rearrange(F.interpolate(rearrange(x, 't c -> 1 c t'), size=n), '1 c t -> t c')

        ensure_length = None
        if 'motion' in self.modules_to_load:
            if ensure_length is None:
                ensure_length = data_point['motion'].shape[0]
            else:
                data_point['motion'] = interpolate(data_point['motion'], ensure_length)

        if 'text' in self.modules_to_load:
            if ensure_length is None:
                ensure_length = data_point['text'].shape[0]
            else:
                data_point['text'] = interpolate(data_point['text'], ensure_length)

        if 'audio' in self.modules_to_load:
            if ensure_length is None:
                ensure_length = data_point['audio'].shape[0]
            else:
                data_point['audio'] = interpolate(data_point['audio'], ensure_length)

    def __getitem__(self, index: int):
        data_point = self.data_files[index]
        if 'motion' in self.modules_to_load:
            data_point['motion'] = self.load_motion(data_point['motion_filename'])
        if 'text' in self.modules_to_load:
            data_point['text'] = self.load_text(data_point['text_filename'])
        if 'audio' in self.modules_to_load:
            data_point['audio'] = self.load_audio(data_point['audio_filename'])

        self.ensure_same_length(data_point)

        return data_point


def custom_collate(batch):
    """Outputs will be of shape: b t c"""
    collated_batch = defaultdict(list)
    for data_point in batch:
        collated_batch['agent'].append(data_point['agent'])
        for k, v in data_point.items():
            if isinstance(v, str):
                collated_batch[k].append(v)
            elif isinstance(v, torch.Tensor):
                collated_batch[k].append(v)
                collated_batch[f"{k}_length"].append(v.shape[0])

    for k, v in collated_batch.items():
        if k.endswith('_length'):
            collated_batch[k] = torch.tensor(v)
        elif isinstance(v[0], torch.Tensor):
            collated_batch[k] = pad_sequence(v, batch_first=True)
        else:
            collated_batch[k] = v

    return collated_batch


class DataModule(L.LightningDataModule):

    def __init__(self, data_loc, agents, modalities, batch_size: int = 32, num_worker: int = 1):
        super().__init__()
        self.save_hyperparameters()

        self.data_loc = data_loc
        self.agents = agents
        self.batch_size = batch_size
        self.num_workers = num_worker

    def setup(self, stage: str):
        self.train_dataset = MotionTextDataset(f'{self.data_loc}/trn', agents=self.agents)
        self.val_dataset = MotionTextDataset(f'{self.data_loc}/val', agents=self.agents)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=custom_collate, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=custom_collate, num_workers=self.num_workers)

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--agents", type=str, nargs='+', default=['main-agent', 'interloctr'], help="Agents' data to use")
        parser.add_argument("--batch_size", type=int, default=64, help="batch size for training")
        parser.add_argument("--num_workers", type=int, default=20, help="number of workers for training")
        parser.add_argument("--data_dir", type=str, default="data/chunks", help="path to data")
        parser.add_argument("--motion_dim", type=int, default=74, help="dimension of motion")
        parser.add_argument("--inputs_dim", type=int, default=768, help="dimension of motion")
        parser.add_argument("--modalities", type=str, nargs='+', default=['motion', 'text', 'audio'], help="modalities to train on, motion will be the final output")
        return parser


if __name__ == "__main__":
    """
    python motion_text_dataset.py --data_dir data/chunks --agents main-agent interloctr --batch_size 32 --num_workers 1
    """
    import argparse
    import lightning as L
    from lightning.pytorch.callbacks import ModelCheckpoint

    # Create the parser
    parser = argparse.ArgumentParser(description='Motion-Text-Audio Dataset Training')

    # Add DataModule specific arguments
    parser = DataModule.add_argparse_args(parser)

    # Add any additional training arguments
    parser.add_argument('--max_epochs', type=int, default=100, help='maximum number of training epochs')
    parser.add_argument('--gpu', type=str, default="cuda:0", help='GPU')
    args = parser.parse_args()

    # Initialize the DataModule
    data_module = DataModule(
        data_loc=args.data_dir,
        agents=args.agents,
        modalities=args.modalities,
        batch_size=args.batch_size,
        num_worker=args.num_workers
    )

    # Set up trainer
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.gpu,
        devices=1,
        callbacks=[
            ModelCheckpoint(
                dirpath='checkpoints',
                filename='model-{epoch:02d}-{val_loss:.2f}',
                save_top_k=3,
                monitor='val_loss',
                mode='min'
            )
        ],
    )

    # Example usage of the dataloader
    data_module.setup(stage='fit')
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    # Print some information about the dataset
    print("\nDataset Information:")
    print(f"Number of training samples: {len(data_module.train_dataset)}")
    print(f"Number of validation samples: {len(data_module.val_dataset)}")

    # Example of accessing a batch
    batch = next(iter(train_loader))
    print("\nBatch contents:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: shape = {value.shape}")
        elif isinstance(value, list):
            print(f"{key}: length = {len(value)}")

    print("\nAvailable modalities:", args.modalities)
    print("Using agents:", args.agents)
