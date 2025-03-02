import argparse

import lightning as L
from lightning.pytorch.callbacks import (DeviceStatsMonitor, EarlyStopping, ModelCheckpoint)
from lightning.pytorch.loggers import TensorBoardLogger

from gestureclip.lightning_module import ClipModel
from gestureclip.motion_text_dataset import DataModule


def main(args):
    data_module = DataModule(args.data_dir, args.agents, args.modalities, batch_size=args.batch_size, num_worker=args.num_workers)
    model = ClipModel(
        args,
        args.inputs_dim,
        args.motion_dim,
        args.embed_dim,
        args.context_length,
        args.transformer_width,
        args.transformer_heads,
        args.transformer_layers,
        input_modalities=args.modalities,
    )

    tb_logger = TensorBoardLogger(save_dir=args.logdir, name=args.run_name,
                                  sub_dir=args.subdir if hasattr(args, 'subdir') else None)
    print(args.gpus, type(args.gpus))
    trainer = L.Trainer(
        devices=args.gpus,
        log_every_n_steps=10,
        callbacks=[
            ModelCheckpoint(monitor="val_loss"),
            EarlyStopping(monitor="val_loss", patience=5, mode="min"),
            DeviceStatsMonitor(cpu_stats=False)
        ],
        logger=tb_logger,
        precision='bf16-mixed',
        max_epochs=-1,
    )
    trainer.fit(model, data_module, args.checkpoint_path)


if __name__ == "__main__":
    """
    python train.py --gpu cuda:0 --run_name=NewData --logdir=lightning_logs --checkpoint_path=None --data_dir=data/genea2023 --agents=1 --modalities=motion,text,audio --batch_size=1 --num_workers=0 --inputs_dim=2048 --motion_dim=2048 --embed_dim=512 --context_length=512 --transformer_width=512 --transformer_heads=8 --transformer_layers=12 --lr=1e-5
    """
    parser = argparse.ArgumentParser()
    # parser.add_argument("--gpus", nargs='+', type=int, default=[0])
    parser.add_argument("--gpu", type=str, default="mps", required=True, help="cuda:0 or mps GPU")
    parser.add_argument("--run_name", type=str, default="NewData")
    parser.add_argument("--logdir", type=str, default="lightning_logs")
    parser.add_argument("--checkpoint_path", "-c", type=str, default=None, help="path to checkpoint to resume training")
    parser = DataModule.add_argparse_args(parser)
    parser = ClipModel.add_argparse_args(parser)
    args = parser.parse_args()

    for lr in [1e-5, 5e-6, 3e-6, 1e-6]:
        args.lr = lr
        args.subdir = f"lr_{lr}"
        print(args)
        main(args)
