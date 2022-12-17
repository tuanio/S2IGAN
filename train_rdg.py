import wandb
import hydra
import torch
import argparse
from torch import nn
from dotenv import load_dotenv
from torchsummary import summary
from torch.utils.data import DataLoader
from data.dataset import RDGDataset
from data.data_collator import rdg_collate_fn
from omegaconf import DictConfig, OmegaConf

config_path = "conf"
config_name = "rdg_config"


@hydra.main(version_base=None, config_path=config_path, config_name=config_name)
def main(cfg: DictConfig):
    if cfg.experiment.log_wandb:
        wandb.init(project="speech2image", name="RDG")
        
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    multi_gpu = torch.cuda.device_count() > 1

    train_set = RDGDataset(**cfg.data.train)
    test_set = RDGDataset(**cfg.data.test)

    real_img, similar_img, wrong_img, spec, spec_len, label = train_set[0]

    print(real_img.size())
    print(similar_img.size())
    print(wrong_img.size())
    print(spec.size())
    print(spec_len)
    print(label)


if __name__ == "__main__":
    load_dotenv()
    main()
