import torch
from tqdm import tqdm
from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
import wandb
import math
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from imgs2csv import images_to_csv_with_metadata

def train_one_epoch(model, dl, optimizer, loss_fn, indices = [1,2,5,6,9], epoch=1, device='cpu', use_wandb=False):
    model.to(device)
    model.train()
    running_loss = 0.0
    pbar = tqdm(dl,desc=f"Epoch {epoch+1}")
    n_steps_per_epoch = math.ceil(len(dl.dataset) / dl.batch_size)
    for step, (inputs, labels) in enumerate(pbar):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = 0.0
        for i in indices:
            loss += loss_fn(outputs[i], labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        avg_loss = running_loss/(step+1)

        pbar.set_postfix(train_loss=avg_loss)
        if use_wandb:
            metrics = {"train/train_loss": avg_loss}
            wandb.log(metrics, step=step+1+n_steps_per_epoch*epoch)
    
    return avg_loss

def val_one_epoch(model, dl, loss_fn, indices = [1,2,5,6,9], device='cpu'):
    model.to(device)
    model.eval()
    running_loss = 0.0
    running_loss1 = 0.0

    pbar = tqdm(dl,desc=f"Validation: ")
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            loss = 0.0
            for i in indices:
                loss += loss_fn(outputs[i], labels)

            running_loss += loss.item()
            avg_loss = running_loss/(i+1)

            # This is for checking the original error/ wanted error
            loss1 = loss_fn(outputs[1], labels)
            running_loss1 += loss1.item()
            avg_loss1 = running_loss1/(i+1)



            pbar.set_postfix(val_loss=avg_loss)
    
    return avg_loss1



def train(model, optimizer, loss_fn, dataloaders, config, indices = [1,2,5,6,9], scheduler = None, device = 'cpu', use_wandb=False):
    model.to(device)
    best_loss = float('inf')
    os.makedirs('ckpts', exist_ok=True)
    cfg = config['train']

    for epoch in range(cfg['epochs']):
        model.train()
        train_loss = train_one_epoch(model, dataloaders['train'], optimizer, loss_fn, epoch=epoch, device=device, use_wandb=use_wandb)
        if (epoch+1)%cfg['val_interval']==0:
            val_loss = val_one_epoch(model, dataloaders['val'], loss_fn, device=device)
            if use_wandb:
                val_metrics = {"val/val_loss": val_loss}
                wandb.log(val_metrics,step=wandb.run.step)
            if val_loss<best_loss:
                best_loss = val_loss
                model_name = type(model).__name__+'_'+device.type+'epoch'+str(epoch)+str(datetime.now())[8:18]+'.pt'
                model_path = os.path.join('ckpts', model_name)
                torch.save(model.state_dict(), model_path)
                if use_wandb:
                    artifact = wandb.Artifact("trained_models", type="model", metadata=config['model'])
                    artifact.add_file(model_path)
                    artifact.save()
                    wandb.log_artifact(artifact)

        if scheduler:
            scheduler.step()
    return model

def predict(model, dl, model_name = None, device='cpu', dest = 'predictions', use_wandb=False):
    model.to(device)
    model.eval()
    os.makedirs('predictions', exist_ok=True)
    if model_name is None:
        model_name = type(model).__name__+'_'+str(datetime.now())[8:16]
    save_dir = os.path.join('predictions',model_name)
    print(save_dir)
    os.makedirs(save_dir)
    pbar = tqdm(dl,desc=f"Predicting.... ")
    with torch.no_grad():
        for i, (inputs, image_names) in enumerate(pbar):
            inputs = inputs.to(device)
            im_names = [s.split('/')[-1] for s in image_names]
            outputs = model(inputs)
            for x,name in zip(outputs, im_names):
                img = to_pil_image(x)
                img.save(os.path.join(save_dir, name))

    predictions_csv = 'predictions/pred_'+model_name+'.csv'
    images_to_csv_with_metadata(save_dir, predictions_csv)
    if use_wandb:
        artifact = wandb.Artifact("csv_files")
        artifact.add_file(predictions_csv)
        artifact.save()
        wandb.log_artifact(artifact)