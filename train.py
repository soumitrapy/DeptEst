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

def train_one_epoch(model, dl, optimizer, loss_fn, epoch=1, device='cpu', use_wandb=False):
    model.to(device)
    model.train()
    running_loss = 0.0
    pbar = tqdm(dl,desc=f"Epoch {epoch+1}")
    n_steps_per_epoch = math.ceil(len(dl.dataset) / dl.batch_size)
    for step, (inputs, labels) in enumerate(pbar):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        avg_loss = running_loss/(step+1)

        pbar.set_postfix(train_loss=avg_loss)
        if use_wandb:
            metrics = {"train/train_loss": avg_loss}
            wandb.log(metrics, step=step+1+n_steps_per_epoch*epoch)
    
    return avg_loss

def val_one_epoch(model, dl, loss_fn, device='cpu'):
    model.to(device)
    model.eval()
    running_loss = 0.0

    pbar = tqdm(dl,desc=f"Validation: ")
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            running_loss += loss.item()
            avg_loss = running_loss/(i+1)

            pbar.set_postfix(val_loss=avg_loss)
    
    return avg_loss



def train(model, optimizer, loss_fn, dataloaders, config, scheduler = None, device = 'cpu', use_wandb=False):
    model.to(device)
    best_loss = float('inf')
    os.makedirs('ckpts', exist_ok=True)
    cfg = config['train']

    for epoch in range(cfg['epochs']):
        model.train()
        train_loss = train_one_epoch(model, dataloaders['train'], optimizer, loss_fn, epoch=epoch, device=device, use_wandb=use_wandb)
        if (epoch+1)%config['val_interval']==0:
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

def predict(model, dl, device='cpu', dest = 'predictions'):
    model.to(device)
    model.eval()
    os.makedirs('predictions', exist_ok=True)
    save_dir = os.path.join('predictions',type(model).__name__+'_'+str(datetime.now())[8:16])
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

    images_to_csv_with_metadata(save_dir, 'predictions/predictions.csv')
    
    


# def evaluate(model, dl, loss_fn, device = 'cpu', use_wandb=False):
#     test_loss, test_acc = 0, 0
#     # test_loss, test_acc = val_one_epoch(model, dl, loss_fn, device=device)
#     indices, highest_losses, hardest_examples, true_labels, predictions = get_examples(model, dl.dataset, loss_fn, n = 10, hard = 10, device = device)
#     if use_wandb:
#         wandb.summary.update({"loss": test_loss, "accuracy": test_acc})
#         wandb.log({"examples":
#             [wandb.Image(hard_example, caption=str(int(pred)) + "," +  str(int(label)))
#              for hard_example, pred, label in zip(hardest_examples, predictions, true_labels)]})
#     return indices, true_labels, predictions

# def get_examples(model, testing_set, loss_fn, n = 10, hard = 10, device = 'cpu'):
#     model.eval()
#     loader = DataLoader(testing_set, 1, shuffle=True)

#     # get the losses and predictions for each item in the dataset
#     losses = None
#     predictions = None
#     i = 0
#     with torch.no_grad():
#         for data, target in loader:
#             i += 1
#             if i>10:
#                 break
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             loss = loss_fn(output, target)
#             pred = output.argmax(dim=1, keepdim=True)

#             if losses is None:
#                 losses = loss.view((1, 1))
#                 predictions = pred
#             else:
#                 losses = torch.cat((losses, loss.view((1, 1))), 0)
#                 predictions = torch.cat((predictions, pred), 0)

#     argsort_loss = torch.argsort(losses, dim=0)
#     print(argsort_loss.shape)
#     indices = argsort_loss[-hard] #


#     # highest_k_losses = losses[argsort_loss[-k:]]
#     # hardest_k_examples = testing_set[argsort_loss[-k:]][0]
#     # true_labels = testing_set[argsort_loss[-k:]][1]
#     # predicted_labels = predictions[argsort_loss[-k:]]

#     # return highest_k_losses, hardest_k_examples, true_labels, predicted_labels
#     return indices, losses[indices], testing_set[indices][0], testing_set[indices][1], predictions[indices]
