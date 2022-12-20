import torch
from tqdm import tqdm

import wandb

from torchvision import transforms as T


def get_transform(img_dim):
    return T.Compose([T.Resize(img_dim), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


Resizer = {64: get_transform(64), 128: get_transform(128), 256: get_transform(256)}


def update_D(models, optimizers, schedulers, criterion, specific_params, device):
    original_real_img, origin_similar_img, original_wrong_img, spec, spec_len = batch

    real_imgs, wrong_imgs, similar_imgs = {}, {}, {}
    for img_dim in specific_params.img_dims:
        real_imgs[img_dim] = Resizer[img_dim](original_real_img)
        wrong_imgs[img_dim] = Resizer[img_dim](original_wrong_img)
        similar_imgs[img_dim] = Resizer[img_dim](origin_similar_img)

    for key in optimizers["disc"].keys():
        optimizers["disc"][key].zero_grad()

    bs = original_real_img.size(0)

    Z = torch.randn(bs, specific_params.latent_space_dim, device=device)
    A = models["sed"](spec, spec_len)

    fake_imgs, mu, logvar = models["gen"](Z, A)

    zero_labels = torch.zeros(bs, device=device, dtype=torch.float)
    one_labels = torch.ones(bs, device=device, dtype=torch.float)
    two_labels = torch.zeros(bs, device=device, dtype=torch.float) + 2

    D_loss = 0
    for img_dim in specific_params.img_dims:
        optimizers["disc"][img_dim].zero_grad()

        real_img = real_imgs[img_dim]
        wrong_img = wrong_imgs[img_dim]

        real_out = models["disc"][img_dim](real_img, mu.detach())
        wrong_out = models["disc"][img_dim](wrong_img, mu.detach())
        fake_out = models["disc"][img_dim](fake_imgs[img_dim].detach(), mu.detach())

        loss_real_cond = criterions["bce"](real_out["cond"], one_labels)
        loss_real_uncond = criterions["bce"](real_out["uncond"], one_labels)
        # ---
        loss_wrong_cond = criterions["bce"](wrong_out["cond"], zero_labels)
        loss_wrong_uncond = criterions["bce"](wrong_out["uncond"], one_labels)
        # ---
        loss_fake_cond = criterions["bce"](fake_out["cond"], zero_labels)
        loss_fake_uncond = criterions["bce"](fake_out["uncond"], zero_labels)

        curr_D_loss = (
            loss_real_cond
            + loss_real_uncond
            + loss_fake_cond
            + loss_fake_uncond
            + loss_wrong_cond
            + loss_wrong_uncond
        )
        curr_D_loss.backward()
        optimizers["disc"][img_dim].step()
        schedulers["disc"][img_dim].step()

        D_loss += curr_D_loss.detach().item()

    return {"D_loss": D_loss}


def update_RS(models, optimizers, schedulers, criterion, specific_params, device):
    original_real_img, origin_similar_img, original_wrong_img, spec, spec_len = batch

    real_imgs, wrong_imgs, similar_imgs = {}, {}, {}
    for img_dim in specific_params.img_dims:
        real_imgs[img_dim] = Resizer[img_dim](original_real_img)
        wrong_imgs[img_dim] = Resizer[img_dim](original_wrong_img)
        similar_imgs[img_dim] = Resizer[img_dim](origin_similar_img)

    optimizers["rs"].zero_grad()

    bs = original_real_img.size(0)

    Z = torch.randn(bs, specific_params.latent_space_dim, device=device)
    A = models["sed"](spec, spec_len)

    fake_imgs, mu, logvar = models["gen"](Z, A)

    zero_labels = torch.zeros(bs, device=device, dtype=torch.float)
    one_labels = torch.ones(bs, device=device, dtype=torch.float)
    two_labels = torch.zeros(bs, device=device, dtype=torch.float) + 2

    real_img = Resizer[256](origin_real_img)
    similar_img = Resizer[256](origin_similar_img)
    wrong_img = Resizer[256](original_wrong_img)

    real_feat = models["ied"](real_img)
    similar_feat = models["ied"](similar_img)
    fake_feat = models["ied"](fake_imgs[256].detach())
    wrong_feat = models["ied"](wrong_img)

    R1 = models["rs"](similar_feat.detach(), real_feat.detach())
    R2 = models["rs"](wrong_feat.detach(), real_feat.detach())
    R3 = models["rs"](real_feat.detach(), real_feat.detach())
    R_GT_FI = models["rs"](fake_feat.detach(), real_feat.detach())

    rs_loss = criterions["rs"](R1, R2, R3, R_GT_FI, zero_labels, one_labels, two_labels)

    rs_loss.backward()
    optimizers["rs"].step()
    schedulers["rs"].step()

    return {"RS_loss": rs_loss.detach().item()}


def update_G(models, optimizers, schedulers, criterion, specific_params, device):
    original_real_img, origin_similar_img, original_wrong_img, spec, spec_len = batch

    real_imgs, wrong_imgs, similar_imgs = {}, {}, {}
    for img_dim in specific_params.img_dims:
        real_imgs[img_dim] = Resizer[img_dim](original_real_img)
        wrong_imgs[img_dim] = Resizer[img_dim](original_wrong_img)
        similar_imgs[img_dim] = Resizer[img_dim](origin_similar_img)

    optimizers["gen"].zero_grad()

    bs = original_real_img.size(0)

    zero_labels = torch.zeros(bs, device=device, dtype=torch.float)
    one_labels = torch.ones(bs, device=device, dtype=torch.float)
    two_labels = torch.zeros(bs, device=device, dtype=torch.float) + 2

    Z = torch.randn(bs, specific_params.latent_space_dim, device=device)
    A = models["sed"](spec, spec_len)

    G_loss = 0
    for img_dim in specific_params.img_dims:

        fake_out = models["disc"][img_dim](fake_imgs[img_dim], mu)
        G_loss += criterion["ce"](fake_out["cond"], one_labels) + criterion["ce"](
            fake_out["uncond"], one_labels
        )

        real_feat = models["ied"](real_imgs[img_dim])
        fake_feat = models["ied"](fake_imgs[img_dim])
        rs_out = models["rs"](real_feat, fake_feat)

        G_loss += criterion["ce"](rs_out, one_labels.long())

    G_loss += criterion["kl"](mu, logvar) * specific_params.kl_loss_coef
    G_loss.backward()
    optimizers["gen"].step()
    schedulers["gen"].step()

    return {"G_loss": G_loss.detach().item()}


def rdg_train_epoch(
    models,
    dataloader,
    optimizers,
    schedulers,
    criterions,
    specific_params,
    device,
    epoch,
    log_wandb,
):
    size = len(dataloader)
    pbar = tqdm(dataloader, total=size)
    for (
        original_real_img,
        origin_similar_img,
        original_wrong_img,
        spec,
        spec_len,
    ) in pbar:
        # original_real_img, origin_similar_img, original_wrong_img, spec, spec_len
        batch = (
            original_real_img.to(device),
            origin_similar_img.to(device),
            original_wrong_img.to(device),
            spec.to(device),
            spec_len.to(device),
        )

        D_loss = update_D(
            models,
            batch,
            optimizers["disc"],
            schedulers["disc"],
            criterion,
            specific_params,
            device,
        )
        RS_loss = update_RS(
            models,
            batch,
            optimizers["rs"],
            schedulers["rs"],
            criterion,
            specific_params,
            device,
        )
        G_loss = update_G(
            models,
            batch,
            optimizers["gen"],
            schedulers["gen"],
            criterion,
            specific_params,
            device,
        )

        if log_wandb:
            wandb.log({"train/G_loss": G_loss.item()})
            wandb.log({"train/D_loss": D_loss.item()})
            wandb.log({"train/RS_loss": RS_loss.item()})
            wandb.log({"train/epoch": epoch})
            wandb.log({"train/lr-OneCycleLR_G": schedulers["gen"].get_last_lr()[0]})

        pbar.set_description(
            f"[Epoch: {epoch}] G_Loss: {G_loss:.2f} | D_Loss: {D_loss:.2f} | RS_loss: {RS_loss:.2f}"
        )
