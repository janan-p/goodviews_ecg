import torch
import torchprofile
from model.resnetsup import resnet34

# Instance of model
model = resnet34(num_classes=4)

# Same parameters as in sup_train.py
x = torch.randn(1, 12, 2048)

with torch.no_grad():
    macs = torchprofile.profile_macs(model, x)

print(macs)
