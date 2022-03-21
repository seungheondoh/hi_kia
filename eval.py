import json
import os
from argparse import ArgumentParser, Namespace, ArgumentTypeError
from pathlib import Path
import torch
import wandb
import torchaudio

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.plugins import DDPPlugin

import matplotlib.pyplot as plt
from models.model import AudioModel
from models.lightning_model import SpeechCls
from loader.dataloader import DataPipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

def save_cm(predict, label, label_name, save_path):
    predict_ = [label_name[i] for i in predict]
    label_ = [label_name[i] for i in label]
    cm = confusion_matrix(label_, predict_, labels=label_name)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_name)
    disp.plot(xticks_rotation="vertical")
    plt.savefig(os.path.join(save_path, 'cm.png'), dpi=150)

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


def save_hparams(args, save_path):
    save_config = OmegaConf.create(vars(args))
    os.makedirs(save_path, exist_ok=True)
    OmegaConf.save(config=save_config, f= Path(save_path, "hparams.yaml"))

def main(args) -> None:
    if args.reproduce:
        seed_everything(42)

    save_path = f"exp/{args.data_type}/{args.cv_split}/{args.freeze_type}"
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

    state_dict = torch.load(os.path.join(save_path, "best.ckpt"))
    runner.load_state_dict(state_dict.get("state_dict"))

    logger = get_wandb_logger(runner)
    
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
                    plugins=DDPPlugin(find_unused_parameters=False)
                )

    trainer.test(runner, datamodule=pipeline)

    # fold-wise evaluation
    with open(Path(save_path, "results.json"), mode="w") as io:
        json.dump(runner.test_results, io, indent=4)
    
    # fold-wise inference
    torch.save(runner.inference, Path(save_path, "inference.pt"))

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
    parser.add_argument("--reproduce", default=True, type=str2bool)
    # parser.add_argument("--deterministic", default=True, type=str2bool)
    # parser.add_argument("--benchmark", default=False, type=str2bool)

    args = parser.parse_args()
    main(args)