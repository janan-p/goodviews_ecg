import numpy as np
from datetime import datetime
from tqdm import tqdm

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune

from config import args
from data import get_data
from model import get_model
from utils.metrics import Evaluator
from utils.logger import Logger
from utils.utils import set_seeds, set_devices
from utils.loss import get_contrastive_loss
from utils.lr_scheduler import LR_Scheduler
from sklearn.metrics import roc_auc_score

seed = set_seeds(args)
device = set_devices(args)
logger = Logger(args)

# Load Data, Create Model
train_loader, val_loader, test_loader = get_data(args)
model = get_model(args, device=device)

nlabels = 4
classifier = nn.Linear(args.embed_size, nlabels).to(device)

criterion = nn.CrossEntropyLoss(reduction='none').to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = LR_Scheduler(optimizer, args.scheduler, args.lr, args.epochs, from_iter=args.lr_sch_start, warmup_iters=args.warmup_iters, functional=True)

best_val_auc = 0
best_epoch = -1

### TRAINING
pbar = tqdm(total=args.epochs, initial=0, bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}")
for epoch in range(1, args.epochs + 1):
    model.train()
    classifier.train()
    logger.evaluator.reset()
    logger.loss = 0

    for (idx, train_batch) in enumerate(train_loader):
        train_x, train_y, train_fnames = train_batch

        train_x= train_x.to(device)

        features = model(train_x)
        logits = classifier(features)

        loss = criterion(logits, train_y.long().to(device)).mean()
        
        print(loss)
        logger.loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
       
    ### VALIDATION
    model.eval()
    classifier.eval()
    y_true, y_scores = [], []
    logger.evaluator.reset()
    with torch.no_grad():
        for batch in val_loader:
            val_x, val_y, _= batch
            val_x, val_y = val_x.to(device), val_y.long().to(device)

            features = model(val_x)
            logits = classifier(features)
            probs = torch.softmax(logits, dim=1)

            y_scores.append(probs.cpu())
            y_true.append(val_y.cpu())

            loss = criterion(logits, val_y.long()).mean()
            print (loss)
    y_scores = torch.cat(y_scores).numpy()
    y_true = torch.cat(y_true).numpy()
    val_auc = roc_auc_score(y_true=y_true, y_score=y_scores, multi_class='ovr')

    print(f"Epoch {epoch}/{args.epochs} - Loss: {logger.loss:.4f} - Val AUC: {val_auc:.4f}")
    pbar.set_description(f"Epoch {epoch} AUC {val_auc:.4f}")
    pbar.update(1)
    # model.train()

    # Save best model
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_epoch = epoch
        ckpt = {
            'model': model.state_dict(),
            'classifier': classifier.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'args': args  # Optional: helpful for debugging later
        }
        save_dir = f"{args.dir_result}/{args.name}/ckpts"
        os.makedirs(save_dir, exist_ok=True)
        torch.save(ckpt, os.path.join(save_dir, "model.pth"))

pbar.close()

#if args.epochs > 0:
#    ckpt = logger.save(model, optimizer, epoch, last=True)
#    logger.writer.close()

print("\n Finished training..........Starting Test")

with torch.no_grad():
    model.eval()
    classifier.eval()
    y_pred = []
    y_target = []

    for (i,test_batch) in enumerate(test_loader):
        test_x, test_y, test_fnames = test_batch
        
        test_x = test_x.to(device)
        test_pred = classifier(model(test_x))
        test_probs = torch.softmax(test_pred, dim=1)
        y_pred.append(test_probs.cpu())
        y_target.append(test_y.cpu())

    y_pred = torch.cat(y_pred, dim=0).numpy()
    y_target = torch.cat(y_target,dim=0).numpy()

    test_auc = roc_auc_score(y_true=y_target, y_score=y_pred, multi_class='ovr')
    print(f"Test AUC:{test_auc}")

