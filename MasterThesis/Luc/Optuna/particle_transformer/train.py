#import sys
#sys.path.append('C:/Users/lucbu/Documents/Master Thesis/Zhongyi/')

from particle_transformer.process_data import get_training_dataloaders
from particle_transformer.particle_transformer import ParticleTransformer
import torch
import torch.optim as optim
import wandb
import os
import argparse
import yaml
import torch.nn as nn

def rob_funcs(trial, data_loc, lr=1e-3, lr_patience=10, n_epochs=100, n_batches=100, input_dim=10, num_classes=2, aux_dim=8, fc_params=[[128, 0.1], [512, 0.1], [128, 0.1]]):
    parser = argparse.ArgumentParser(description='Training for four top model')
    parser.add_argument('--gpu_device', type=int)
    parser.add_argument('--lr', type=float, default=lr)
    parser.add_argument('--lr_patience', type=int, default=lr_patience)
    parser.add_argument('--n_epochs', type=int, default=n_epochs)
    parser.add_argument('--n_batches', type=int, default=n_batches)
    args = parser.parse_args()
    
    device = torch.device("cpu")
    #if args.gpu_device is None:
    #    device = torch.device("cpu")
    #else:
    #    device = torch.device("cuda")
    #    torch.cuda.set_device(args.gpu_device)
    
    train_loader, val_loader, test_loader = get_training_dataloaders(n_batches=args.n_batches, data_loc=data_loc)
    
    model = ParticleTransformer(input_dim=input_dim, num_classes=num_classes, aux_dim=aux_dim, trim=False, fc_params=fc_params)
    model.train()
    model.to(device)
    
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=args.lr_patience)
    
    #wandb.init(project='four-top-transformer')
    
    best_val_loss = float("inf")
    for _ in range(args.n_epochs):
        model.train()
        for batch in train_loader:
            batch_aux, batch_tokens, batch_momenta, batch_mask, batch_labels = batch
    
            optimizer.zero_grad()
            pred = model.forward(batch_tokens.to(device), v=batch_momenta.to(device), mask=batch_mask.to(device), aux=batch_aux.to(device))
            loss = loss_func(pred, batch_labels.to(device))
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            val_data_size = 0
            for batch in val_loader:
                batch_aux, batch_tokens, batch_momenta, batch_mask, batch_labels = batch
    
                pred = model.forward(batch_tokens.to(device), v=batch_momenta.to(device), mask=batch_mask.to(device), aux=batch_aux.to(device))
                loss_now = loss_func(pred, batch_labels.to(device))
    
                val_loss = val_loss*val_data_size + loss_now.item()*batch[0].shape[0]
                val_data_size += batch[0].shape[0]
                val_loss /= val_data_size
    
            # Update LR
            scheduler.step(val_loss)
    
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                #torch.save(model.state_dict(), os.path.join(wandb.run.dir, "model.pt"))
    
            #wandb.log({ "batch_loss": loss,
            #    "learning_rate": optimizer.param_groups[0]['lr'],
            #    "val_loss": val_loss})
    score = best_val_loss      
    return score