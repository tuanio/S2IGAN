import torch
from tqdm import tqdm

import wandb

from torchvision import transforms as T


def get_transform(img_dim):
    return T.Compose([T.Resize(img_dim), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


Resizer = {64: get_transform(64), 128: get_transform(128), 256: get_transform(256)}


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
        original_real_img, origin_similar_img, original_wrong_img, spec, spec_len = (
            original_real_img.to(device),
            origin_similar_img.to(device),
            original_wrong_img.to(device),
            spec.to(device),
            spec_len.to(device),
        )

        for i in optimizers.keys():
            optimizers[i].zero_grad()

        bs = original_real_img.size(0)

        Z = torch.randn(bs, specific_params.latent_space_dim, device=device)
        A = models["sed"](spec, spec_len)

        fake_imgs, mu, logvar = models["gen"](Z, A)

        zero_labels = torch.zeros(bs, device=device, dtype=torch.float)
        one_labels = torch.ones(bs, device=device, dtype=torch.float)
        two_labels = torch.zeros(bs, device=device, dtype=torch.float) + 2

        # ---- Update D

        disc_loss = 0

        real_imgs, wrong_imgs, similar_imgs = {}, {}, {}
        for img_dim in specific_params.img_dims:
            real_imgs[img_dim] = Resizer[img_dim](original_real_img)
            wrong_imgs[img_dim] = Resizer[img_dim](original_wrong_img)
            similar_imgs[img_dim] = Resizer[img_dim](origin_similar_img)

        disc_labels = torch.ones(bs, device=device)
        for img_dim in specific_params.img_dims:
            disc_name = f"disc_{img_dim}"

            real_img = real_imgs[img_dim]
            wrong_img = wrong_imgs[img_dim]

            real_out = models[disc_name](real_img, mu.detach())
            wrong_out = models[disc_name](wrong_img, mu.detach())
            fake_out = models[disc_name](fake_imgs[img_dim].detach(), mu.detach())

            loss_real_cond = criterions["bce"](real_out["cond"], one_labels)
            loss_real_uncond = criterions["bce"](real_out["uncond"], one_labels)
            # ---
            loss_wrong_cond = criterions["bce"](wrong_out["cond"], zero_labels)
            loss_wrong_uncond = criterions["bce"](wrong_out["uncond"], one_labels)
            # ---
            loss_fake_cond = criterions["bce"](fake_out["cond"], zero_labels)
            loss_fake_uncond = criterions["bce"](fake_out["uncond"], zero_labels)

            disc_loss += (
                loss_real_cond
                + loss_real_uncond
                + loss_fake_cond
                + loss_fake_uncond
                + loss_wrong_cond
                + loss_wrong_uncond
            )

        disc_loss.backward()
        optimizers["disc"].step()
        schedulers["disc"].step()

        # ---- Update G and RS
        # --- RS

        real_img = real_imgs[256]
        similar_img = similar_imgs[256]
        wrong_img = wrong_imgs[256]

        real_feat = models["ied"](real_img)
        similar_feat = models["ied"](similar_img)
        fake_feat = models["ied"](fake_imgs[256].detach())
        wrong_feat = models["ied"](wrong_img)

        R1 = models["rs"](similar_feat, real_feat)
        R2 = models["rs"](wrong_feat, real_feat)
        R3 = models["rs"](real_feat, real_feat)
        R_GT_FI = models["rs"](fake_feat, real_feat)

        # loss here
        rs_loss = criterions["rs"](
            R1, R2, R3, R_GT_FI, zero_labels, one_labels, two_labels
        )

        rs_loss.backward()

        # --- G
        gen_loss = 0
        for img_dim in specific_params.img_dims:
            disc_name = f"disc_{img_dim}"
            fake_out = models[disc_name](fake_imgs[img_dim], mu.detach())
            gen_loss += criterions["bce"](fake_out["cond"], one_labels)
            gen_loss += criterions["bce"](fake_out["uncond"], one_labels)

        gen_loss += criterions["kl"](mu, logvar) * specific_params.kl_loss_coef
        gen_loss.backward()
        optimizers["gen"].step()
        schedulers["gen"].step()

        if log_wandb:
            wandb.log({"train/G_loss": gen_loss.item()})
            wandb.log({"train/D_loss": disc_loss.item()})
            wandb.log({"train/RS_loss": rs_loss.item()})
            wandb.log({"train/epoch": epoch})
            wandb.log({"train/lr-OneCycleLR_G": schedulers["gen"].get_last_lr()[0]})
            wandb.log({"train/lr-OneCycleLR_D": schedulers["disc"].get_last_lr()[0]})

        pbar.set_description(
            f"[Epoch: {epoch}] G_Loss: {gen_loss.item():.2f} | D_Loss: {disc_loss.item():.2f}"
        )
