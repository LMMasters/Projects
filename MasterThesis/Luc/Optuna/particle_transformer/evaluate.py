from process_data import get_training_dataloaders
from particle_transformer import ParticleTransformer
import torch
import torch.optim as optim
import wandb
import os
import argparse
import yaml
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import pyplot as plt
    
parser = argparse.ArgumentParser(description='Training for four top model')
parser.add_argument('--gpu_device', type=int)
parser.add_argument('--n_batches', type=int, default=100)
args = parser.parse_args()


if args.gpu_device is None:
    device = torch.device("cpu")
else:
    device = torch.device("cuda")
    torch.cuda.set_device(args.gpu_device)


_, _, test_loader = get_training_dataloaders(n_batches=args.n_batches)

model = ParticleTransformer(input_dim=10, num_classes=2, aux_dim=8, trim=False, fc_params=[[128, 0.1], [512, 0.1], [128, 0.1]], for_inference=True)
model.load_state_dict(torch.load("model.pt", map_location=torch.device(device)))
model.eval()
model.to(device)

real_labels_list = []
pred_labels_list = []
with torch.no_grad():
    for batch in test_loader:
        batch_aux, batch_tokens, batch_momenta, batch_mask, batch_labels = batch

        pred = model.forward(batch_tokens.to(device), v=batch_momenta.to(device), mask=batch_mask.to(device), aux=batch_aux.to(device))[:,1]

        real_labels_list.append(batch_labels.numpy())
        pred_labels_list.append(pred.numpy())

real_labels = np.concatenate(real_labels_list)
pred_labels = np.concatenate(pred_labels_list)

print("AUC score:", roc_auc_score(real_labels, pred_labels))

fpr, tpr, _ = roc_curve(real_labels, pred_labels)

plt.plot(fpr, tpr)
plt.savefig('ROC.pdf')