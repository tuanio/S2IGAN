import hydra
import torch
import argparse
from torch import nn
from dotenv import load_dotenv
from torchsummary import summary
from data.dataset import SENDataset
from torch.utils.data import DataLoader
from data.dataloader import sen_collate_fn
from omegaconf import DictConfig, OmegaConf
from s2igan.sen import ImageEncoder, SpeechEncoder
from s2igan.sen.utils import sen_train_epoch, sen_eval_epoch
from s2igan.loss import SENLoss

config_path = "conf"
config_name = "sen_config"


@hydra.main(version_base=None, config_path=config_path, config_name=config_name)
def main(cfg: DictConfig):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    multi_gpu = torch.cuda.device_count() > 1

    train_set = SENDataset(**cfg.data.train)
    test_set = SENDataset(**cfg.data.test)

    bs = cfg.data.general.batch_size
    nwkers = cfg.data.general.num_workers
    train_dataloader = DataLoader(
        train_set, bs, shuffle=True, num_workers=nwkers, collate_fn=sen_collate_fn
    )
    test_dataloder = DataLoader(
        test_set, bs, shuffle=False, num_workers=nwkers, collate_fn=sen_collate_fn
    )

    image_encoder = ImageEncoder(**cfg.model.image_encoder)
    speech_encoder = SpeechEncoder(**cfg.model.speech_encoder)
    classifier = nn.Linear(**cfg.model.classifier)

    if multi_gpu:
        image_encoder = nn.DataParallel(image_encoder, device_ids=[0, 1])
        speech_encoder = nn.DataParallel(speech_encoder, device_ids=[0, 1])
        classifier = nn.DataParallel(classifier, device_ids=[0, 1])

    image_encoder = image_encoder.to(device)
    speech_encoder = speech_encoder.to(device)
    classifier = classifier.to(device)

    try:
        image_encoder = torch.compile(image_encoder)
        speech_encoder = torch.compile(speech_encoder)
        classifier = torch.compile(classifier)
    except:
        print("Can't activate Pytorch 2.0")

    if multi_gpu:
        model_params = (
            image_encoder.module.get_params()
            + speech_encoder.module.get_params()
            + list(classifier.module.parameters())
        )
        # image_encoder_summary = summary(image_encoder.module, [(3, 299, 299,)])
        # speech_encoder_summary = summary(
        #     speech_encoder.module,
        #     [(cfg.data.general.n_mels, 250,), (1,)],
        # )
        # classifier_summary = summary(
        #     classifier.module, [(cfg.model.image_encoder.output_dim,)]
        # )
    else:
        model_params = (
            image_encoder.get_params()
            + speech_encoder.get_params()
            + list(classifier.parameters())
        )
        # image_encoder_summary = summary(image_encoder, [(3, 299, 299,)])
        # speech_encoder_summary = summary(
        #     speech_encoder,
        #     [(cfg.data.general.n_mels, 500,), (1,)],
        #     dtypes=[torch.float, torch.long],
        # )
        # classifier_summary = summary(
        #     classifier, [(cfg.model.image_encoder.output_dim,)]
        # )

    # print(image_encoder_summary)
    # print(speech_encoder_summary)
    # print(classifier_summary)

    optimizer = torch.optim.AdamW(model_params, **cfg.optimizer)
    scheduler = None
    if cfg.scheduler.use:
        steps_per_epoch = len(train_dataloader)
        sched_dict = dict(
            epochs=cfg.experiment.max_epoch, steps_per_epoch=steps_per_epoch
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, **sched_dict, **cfg.scheduler.args
        )
    criterion = SENLoss(**cfg.loss).to(device)

    log_wandb = cfg.experiment.log_wandb

    if cfg.experiment.train:
        for epoch in range(cfg.experiment.max_epoch):
            sen_train_epoch(
                image_encoder,
                speech_encoder,
                classifier,
                train_dataloader,
                optimizer,
                scheduler,
                criterion,
                device,
                epoch,
                log_wandb,
            )
            sen_eval_epoch(
                image_encoder,
                speech_encoder,
                classifier,
                test_dataloder,
                criterion,
                device,
                epoch,
                log_wandb,
            )

    if cfg.experiment.test:
        eval_result = sen_eval_epoch(
            image_encoder,
            speech_encoder,
            classifier,
            test_dataloder,
            criterion,
            device,
            epoch,
            log_wandb,
        )
        print("Eval result:", eval_result)


if __name__ == "__main__":
    load_dotenv()
    main()
