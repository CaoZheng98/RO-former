import torch


# Train utilities
def train_epoch(model, data_loader, optimizer, bos_len, scheduler, loss_func, device,n_epoch):
    """
    Train model for a single epoch.

    :param model: A torch.nn.Module implementing the Transformer model.
    :param data_loader: A PyTorch DataLoader, providing the trainings data in mini batches.
    :param optimizer: One of PyTorch optimizer classes.
    :param scheduler: scheduler of learning rate.
    :param loss_func: The loss function to minimize.
    :param decode_mode: decoding mode in Transformer.
    :param device: device for data and models
    """
    # set model to train mode (important for dropout)
    model.train()
    cnt = 0
    loss_mean = 0
    for src, trg, real_data in data_loader:
        if cnt==0:
            batch_size = src.size(0)
        # delete previously stored gradients from the model
        optimizer.zero_grad()
        # push data to GPU (if available)
        src, trg, real_data = src.to(device), trg.to(device), real_data.to(device)
        # get model predictions
        y_hat = model(src, trg)
        # calculate loss
        loss = loss_func(y_hat, real_data, bos_len)
        if torch.isnan(loss).any():
            loss = loss_func(y_hat, real_data, bos_len)
        # calculate gradients
        loss.backward()
        # update the weights
        optimizer.step()
        # calculate mean loss
        cnt += 1
        if trg.shape[0] == batch_size:
            loss_mean = loss_mean + (loss.item() - loss_mean) / cnt  # Welford’s method
        if cnt % 10 == 0:
            print(f'Train Epoch: {n_epoch} [{cnt}/{len(data_loader)}]\tLoss: {loss}')
    scheduler.step()
    return loss_mean



