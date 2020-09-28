from tqdm import tqdm
import torch
import config

def train_fn(model, data_lodaer, optimizer):
    model.train()
    fin_loss = 0
    tk = tqdm(data_loader, total=len(data_lodaer))
    for data in tk0:
        # for every batch
        for k, v in data.items:
            data[k] = v.to(config.DEVICE)
        # _ refers to predictions
        optimizer.zero_grad()
        _, loss = model(**data)
        loss.backward()
        optimizer.step()
        fin_loss += loss.item()
    return fin_loss / len(data_lodaer)

def eval_fn(model, data_lodaer, optimizer):
    model.eval()
    fin_loss = 0
    fin_preds = []
    tk = tqdm(data_loader, total=len(data_lodaer))
    for data in tk0:
        # for every batch
        for k, v in data.items:
            data[k] = v.to(config.DEVICE)
        # _ refers to predictions
        batch_preds, loss = model(**data)
        fin_loss += loss.item()
        fin_preds.append(batch_preds)
    return fin_preds fin_loss / len(data_lodaer)
    