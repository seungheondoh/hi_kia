import json
import os
from argparse import ArgumentParser, Namespace, ArgumentTypeError
from pathlib import Path

import wandb
import torchaudio

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.plugins import DDPPlugin

from models.model import AudioModel
from models.lightning_model import SpeechCls
from loader.dataloader import DataPipeline

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')

def get_wandb_logger(model):
    logger = WandbLogger()
    logger.watch(model)
    return logger 

def get_checkpoint_callback(save_path) -> ModelCheckpoint:
    prefix = save_path
    suffix = "best"
    checkpoint_callback = ModelCheckpoint(
        dirpath=prefix,
        filename=suffix,
        save_top_k=1,
        save_last= False,
        monitor="val_loss",
        mode="min",
        save_weights_only=True,
        verbose=True,
    )
    return checkpoint_callback

def get_early_stop_callback() -> EarlyStopping:
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=30, verbose=True, mode="min"
    )
    return early_stop_callback

def save_hparams(args, save_path):
    save_config = OmegaConf.create(vars(args))
    os.makedirs(save_path, exist_ok=True)
    OmegaConf.save(config=save_config, f= Path(save_path, "hparams.yaml"))

def main(args) -> None:
    if args.reproduce:
        seed_everything(42)

    save_path = f"exp/{args.data_type}/{args.cv_split}/{args.freeze_type}"
    save_hparams(args, save_path)

    wandb.init(config=args)
    wandb.run.name = f"{args.data_type}/{args.cv_split}/{args.freeze_type}"
    args = wandb.config

    pipeline = DataPipeline(
            root = args.root,
            data_type = args.data_type,
            cv_split = args.cv_split,
            batch_size = args.batch_size,
            num_workers = args.num_workers
    )
    model = AudioModel(args.data_type, args.feature_type, args.freeze_type)

    runner = SpeechCls(
            model = model,
            lr = args.lr, 
            max_epochs = args.max_epochs,
            batch_size = args.batch_size
    )

    logger = get_wandb_logger(runner)
    checkpoint_callback = get_checkpoint_callback(save_path)
    early_stop_callback = get_early_stop_callback()
    trainer = Trainer(
                    max_epochs= args.max_epochs,
                    num_nodes=args.num_nodes,
                    gpus= args.gpus,
                    accelerator= args.accelerator,
                    logger=logger,
                    sync_batchnorm=True,
                    reload_dataloaders_every_epoch=True,
                    resume_from_checkpoint=None,
                    replace_sampler_ddp=False,
                    callbacks=[
                        early_stop_callback,
                        checkpoint_callback
                    ],
                    plugins=DDPPlugin(find_unused_parameters=True)
                )

    trainer.fit(runner, datamodule=pipeline)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--root", default="./dataset", type=str)
    parser.add_argument("--data_type", default="HIKIA", type=str)
    parser.add_argument("--feature_type", default="wav2vec", type=str)
    parser.add_argument("--freeze_type", default="feature", type=str)
    parser.add_argument("--cv_split", default="M1", type=str)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    # runner 
    parser.add_argument("--lr", default=5e-5, type=float)
    # trainer
    parser.add_argument("--max_epochs", default=200, type=int)
    parser.add_argument("--gpus", default="1", type=str)
    parser.add_argument("--accelerator", default="ddp", type=str)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument("--reproduce", default=False, type=str2bool)
    # parser.add_argument("--deterministic", default=True, type=str2bool)
    # parser.add_argument("--benchmark", default=False, type=str2bool)

    args = parser.parse_args()
    main(args)