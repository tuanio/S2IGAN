import torch
from tqdm import tqdm


def sen_train_epoch(
    image_encoder,
    speech_encoder,
    classifier,
    dataloader,
    optimizer,
    scheduler,
    criterion,
    device,
    epoch,
    log_wandb: bool = False,
):
    size = len(dataloader)
    run_loss = 0
    pbar = tqdm(dataloader, total=size)
    for (imgs, specs, len_specs, labels) in pbar:
        imgs, specs, len_specs, labels = (
            imgs.to(device),
            specs.to(device),
            len_specs.to(device),
            labels.to(device),
        )

        optimizer.zero_grad()

        V = image_encoder(imgs)
        A = speech_encoder(specs, len_specs)
        cls_img = classifier(V)
        cls_speech = classifier(A)

        loss = criterion(V, A, cls_img, cls_speech, labels)
        loss.backward()

        optimizer.step()
        if scheduler:
            scheduler.step()

        loss = loss.item()
        run_loss += loss

        if log_wandb:
            wandb.log({"train/loss": loss})

        pbar.set_description(f"[Epoch: {epoch}] | Loss: {loss:.2f}")

    return {"loss": run_loss / size}


def sen_eval_epoch(
    image_encoder,
    speech_encoder,
    classifier,
    dataloader,
    criterion,
    device,
    epoch,
    log_wandb: bool = False,
):
    size = len(dataloader)
    run_loss = 0
    run_img_acc = 0
    run_speech_acc = 0
    pbar = tqdm(dataloader, total=size)
    with torch.no_grad():
        for (imgs, specs, len_specs, labels) in pbar:
            imgs, specs, len_specs, labels = (
                imgs.to(device),
                specs.to(device),
                len_specs.to(device),
                labels.to(device),
            )

            V = image_encoder(imgs)
            A = speech_encoder(specs, len_specs)
            cls_img = classifier(V)
            cls_speech = classifier(A)

            loss = criterion(V, A, cls_img, cls_speech, labels)

            loss = loss.item()

            if log_wandb:
                wandb.log({"train/loss": loss})

            img_acc = (cls_img.argmax(-1) == labels).sum() / labels.size(0) * 100
            speech_acc = (cls_speech.argmax(-1) == labels).sum() / labels.size(0) * 100

            run_loss += loss
            run_img_acc += img_acc.item()
            run_speech_acc += speech_acc.item()

            pbar.set_description(
                f"[Epoch: {epoch}] Loss: {loss:.2f} | Image Acc: {img_acc.item():.2f}% | Speech Acc: {speech_acc.item():.2f}%"
            )

    return {
        "loss": run_loss / size,
        "img_acc": run_img_acc / size,
        "speech_acc": run_speech_acc / size,
    }
