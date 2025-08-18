import os
import sys
import numpy as np
from datetime import datetime
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune

from config import args
from data import get_data
from model import get_model
from utils.utils import set_seeds, set_devices
from sklearn.metrics import roc_auc_score
from torch.serialization import add_safe_globals
import argparse
from profiling import PerformanceMonitor

# Set device
device = set_devices(args)

# Load Data, Create Model
_, _, test_loader = get_data(args)
model = get_model(args, device=device)

nlabels = 4
classifier = nn.Linear(args.embed_size, nlabels).to(device)

# Check if checkpoint exists
ckpt_path = os.path.join(args.dir_result, args.name, "ckpts/model.pth")
if not os.path.exists(ckpt_path):
    print(f"Invalid checkpoint path: {ckpt_path}")
    sys.exit(1)

# Load checkpoint
add_safe_globals([argparse.Namespace])
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt['model'])
classifier.load_state_dict(ckpt['classifier'])
model.eval()
classifier.eval()
print(f"Loaded model from {ckpt_path}")

def eval_auc(loader):
    model.eval() 
    classifier.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(device), y.long().to(device)
            feats = model(x)
            logits = classifier(feats)
            probs = torch.softmax(logits, dim=1)
            y_pred.append(probs.cpu())
            y_true.append(y.cpu())
    y_pred = torch.cat(y_pred).numpy()
    y_true = torch.cat(y_true).numpy()
    return roc_auc_score(y_true=y_true, y_score=y_pred, multi_class="ovr")

# Collect prunable parameters
def collect_params(model, classifier):
    params = []
    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            params.append((module, 'weight'))
    params.append((classifier, 'weight'))
    return params

# Baseline Evaluation
test_auc = eval_auc(test_loader)
print(f"Baseline Test AUC: {test_auc:.4f}")

# Apply Pruning
params_to_prune = collect_params(model, classifier)
prune.global_unstructured(
    params_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=args.prune_amount
)
print(f"Applied global unstructured pruning (amount={args.prune_amount})")

test_auc_pruned = eval_auc(test_loader)
print(f"Pruned Test AUC (no fine-tune): {test_auc_pruned:.4f}")

# Make pruning permanent & Save
for module, _ in params_to_prune:
    prune.remove(module, 'weight')

save_dir = f"{args.dir_result}/{args.save_name}/ckpts"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "model_pruned.pth")

torch.save({
    'model': model.state_dict(),
    'classifier': classifier.state_dict(),
    'args': args
}, save_path)

print(f"Saved pruned model to {save_path}")

