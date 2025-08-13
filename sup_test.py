import os
import sys
import numpy as np
from datetime import datetime
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from config import args
from data import get_data
from model import get_model
from utils.utils import set_seeds, set_devices
from sklearn.metrics import roc_auc_score
from torch.serialization import add_safe_globals
import argparse
from profiling import PerformanceMonitor

# Optional: monitoring
monitor = PerformanceMonitor(interval=10)  
monitor.start()

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

# Optional: quantization
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
print("Quantized model to 8-bit")

# --- TEST LOOP ---
y_pred = []
y_target = []

with torch.no_grad():
    for i, test_batch in enumerate(test_loader):
        if args.viewtype in ['clocstime', 'clocslead']:
            test_x1, test_x2, test_y, test_group, test_fnames = test_batch
            test_x = torch.cat((test_x1, test_x2), dim=0)
            test_y = torch.cat((test_y, test_y), dim=0)
        else:
            test_x, test_y, test_fnames = test_batch

        test_x = test_x.to(device)
        test_pred = classifier(model(test_x))
        test_probs = torch.softmax(test_pred, dim=1)  # convert to probabilities

        y_pred.append(test_probs.cpu())
        y_target.append(test_y)

# Convert to numpy arrays
y_pred = torch.cat(y_pred, dim=0).numpy()  # shape: (num_samples, nlabels)
y_target = torch.cat(y_target, dim=0).to(torch.int64).numpy() # integer labels shape: (num_samples,)

# --- SAVE PREDICTIONS ---
df_preds = pd.DataFrame(y_pred, columns=[f"pred_class_{i}" for i in range(nlabels)])
df_preds["true_class"] = y_target

save_path = os.path.join(args.dir_result, f"{args.name}_test_predictions.csv")
df_preds.to_csv(save_path, index=False)
print(f"Saved per-instance predictions to: {save_path}")

# --- CALCULATE TEST AUC ---
y_target_onehot = np.eye(nlabels)[y_target]  # one-hot encode for AUC calc
test_auc = roc_auc_score(y_true=y_target_onehot, y_score=y_pred, multi_class="ovr")
print(f"Test AUC: {test_auc:.4f}")

# --- BOOTSTRAP 95% CI ---
n_bootstraps = 1000
rng = np.random.default_rng(seed=42)
auc_array = []

for _ in range(n_bootstraps):
    sample_indices = rng.integers(0, len(y_target), len(y_target))
    y_true_sample = y_target[sample_indices]
    y_pred_sample = y_pred[sample_indices]
    y_true_onehot_sample = np.eye(nlabels)[y_true_sample]
    auc = roc_auc_score(y_true_onehot_sample, y_pred_sample, multi_class="ovr")
    auc_array.append(auc)

ci_lower = np.percentile(auc_array, 2.5)
ci_upper = np.percentile(auc_array, 97.5)

print(f"95% CI: ({ci_lower:.4f}, {ci_upper:.4f})")

monitor.stop()
monitor.report()

