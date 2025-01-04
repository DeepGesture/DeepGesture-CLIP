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
from lightning.pytorch.utilities import grad_norm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from model import CLIP
# from clip.dataset import MotionTextDataset

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
    return {k:v for k,v in d.items() if v is not None}

    
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


class ClipModel(L.LightningModule):
    def __init__(
        self,
        args,
        inputs_dim,
        motion_dim,
        embed_dim,
        context_length,
        transformer_width,
        transformer_heads,
        transformer_layers,
        input_modalities=["audio", "text"],
    ):
        super().__init__()
        self.save_hyperparameters(args)
        self.input_modalities = input_modalities
        if "motion" in self.input_modalities:
            self.input_modalities.remove("motion")
        
        self.model = CLIP(
            inputs_dim=inputs_dim,
            motion_dim=motion_dim,
            embed_dim=embed_dim,
            context_length=context_length,
            transformer_width=transformer_width,
            transformer_heads=transformer_heads,
            transformer_layers=transformer_layers,
        )
        self.loss = nn.CrossEntropyLoss()

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--embed_dim", type=int, default=512, help="embedding dimension")
        parser.add_argument("--context_length", type=int, default=500, help="context length")
        parser.add_argument("--transformer_width", type=int, default=768, help="transformer width")
        parser.add_argument("--transformer_heads", type=int, default=8, help="transformer heads")
        parser.add_argument("--transformer_layers", type=int, default=6, help="transformer layers")
        parser.add_argument("--lr", type=float, default=5e-6, help="learning rate")
        return parser

    def training_step(self, batch, batch_idx):
        loss = self._run_loss_computation(batch)
        self.log("train_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._run_loss_computation(batch)
        self.log("val_loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return loss
    
    def _run_loss_computation_second(self, batch):
        batch_size = batch['motion'].shape[0]
        inputs = torch.stack([batch[modality] for modality in self.input_modalities]).sum(0)
        similarity_matrix = self.model(inputs, batch['motion'])
        ground_truth = torch.eye(similarity_matrix.shape[1], device=similarity_matrix.device).unsqueeze(0).expand(batch_size,-1,-1)
        total_loss = F.mse_loss(similarity_matrix,ground_truth)
        return total_loss
    
    def _run_loss_computation(self, batch):
        batch_size = batch['motion'].shape[0]
        inputs = torch.stack([batch[modality] for modality in self.input_modalities]).sum(0)
        logits_per_image, logits_per_text = self.model(inputs, batch['motion'])
        ground_truth = torch.arange(batch_size, device=logits_per_image.device)
        total_loss = (self.loss(logits_per_image,ground_truth) + self.loss(logits_per_text,ground_truth))/2
        return total_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,weight_decay=1e-6)
    
    def on_before_optimizer_step(self, optimizer):
        self.log_dict(grad_norm(self, norm_type=2))
       
    @torch.inference_mode() 
    def embed(self, batch):
        assert isinstance(batch, dict) and all([modality in batch for modality in self.input_modalities]), f"Ensure the input modalities {self.input_modalities} are in the dictionary"
        assert [batch[modality].ndim == 2 for modality in self.input_modalities], f"Ensure the input modalities {self.input_modalities} be of two dimension (length, {self.hparams.inputs_dim}), currently only supports one element at inference at a time."
        assert all([batch[modality].shape[1] == self.hparams.inputs_dim for modality in self.input_modalities]), f"Ensure the input modalities {self.input_modalities} be of two dimension (length, {self.hparams.inputs_dim}), currently only supports one element at inference at a time."
        
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        
        for modality in self.input_modalities:
            batch[modality] = batch[modality].to(dtype).to(device)
        
        max_length = max([batch[modality].shape[0] for modality in self.input_modalities])
        inputs = torch.stack([rearrange(F.interpolate(rearrange(batch[modality], "l c -> 1 c l"), size=max_length), "1 c l -> l c") for modality in self.input_modalities]).sum(0)
        # return inputs
        embeddings = []

        for chunked_x in torch.split(inputs, self.hparams.context_length, dim=0):
            orig_len = chunked_x.shape[0]
            inp = F.pad(chunked_x.unsqueeze(0), (0, 0, 0, self.hparams.context_length - chunked_x.shape[0]))
            emb = self.model.encode_inputs(inp)
            embeddings.append(emb.squeeze(0)[:orig_len])
        return torch.concat(embeddings, dim=0)
    
    
    @torch.inference_mode() 
    def embed_motion(self, batch):
        assert isinstance(batch, dict) and 'motion' in batch, f"Ensure the input modalities {self.input_modalities} are in the dictionary"
        assert batch['motion'].ndim == 2, f"Ensure the input modalities {self.input_modalities} be of two dimension (length, {self.hparams.inputs_dim}), currently only supports one element at inference at a time."
        assert batch['motion'].shape[1] == self.hparams.motion_dim, f"Ensure the input modalities {self.input_modalities} be of two dimension (length, {self.hparams.inputs_dim}), currently only supports one element at inference at a time."
        
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        batch['motion'] = batch['motion'].to(dtype).to(device)
        

        embeddings = []
        for chunked_x in torch.split(batch['motion'], self.hparams.context_length, dim=0):
            orig_len = chunked_x.shape[0]
            inp = F.pad(chunked_x.unsqueeze(0), (0, 0, 0, self.hparams.context_length - chunked_x.shape[0]))
            emb = self.model.encode_motion(inp)
            embeddings.append(emb.squeeze(0)[:orig_len])
        return torch.concat(embeddings, dim=0)
 
 
        


