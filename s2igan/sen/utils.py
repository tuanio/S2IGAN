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

        if log_wandb:
            wandb.log({"train/loss": loss})

        pbar.set_description(f"Epoch: {epoch}, Loss: {loss:.2f}")
