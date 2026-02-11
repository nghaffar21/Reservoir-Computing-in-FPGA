# This code converts a PyTorch model to HLS and syntheses it for FPGA deployment.
# It includes bitstream generation, deployment, timing the inference, and evaluating the predictions.
# It is the modified version of the tutorials 7a to 7c of the hls4ml official tutorials: https://github.com/fastmachinelearning/hls4ml-tutorial/tree/main 
#
# The modifications are:
# - The original documentation is written in Keras, but here we use PyTorch.
# - The Neural Network used is simpler than the one used in the original tutorial, and it is just for demonstration purposes.
# - A simple custom matrix multiplication layer is added to the model to show how to add custom layers in hls4ml.
# - The configuration is modified to optimize the model for FPGA deployment. This includes changing the precision of the layers, setting the reuse factor, and most importantly removing the softmax layer.
# 
# Written by: Nima Ghaffarzadeh, Matteo Mendula
# CTTC, Barcelona, Spain, Feb 2026

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# A simple NN with only the input and output layers
# You can replace this with your actual architecture and weights
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Example custom matrix (16×16)
        self.rc = nn.Linear(16, 16, bias=False)

        self.fc1 = nn.Linear(16, 5)
        #self.softmax = nn.Softmax(dim=1) # Modified to optimize

    def forward(self, x):
        # x shape: (batch, 16)
        x = self.rc(x)
        x = self.fc1(x)
        #x = self.softmax(x) # Modified to optimize
        return x

X_test = np.load('X_test.npy')

# Load Model
model = SimpleNet()
model.eval()

# --- Timing the inference ---

X_test_torch = torch.from_numpy(X_test).float()

# warm-up
with torch.no_grad():
    model(X_test_torch)

# timing
from datetime import datetime
timea = datetime.now()
with torch.no_grad():
    y_torch = model(X_test_torch)
timeb = datetime.now()

dt = (timeb - timea).total_seconds()
print("Torch inference time:", dt)
print("Throughput:", len(X_test) / dt)

# --- End of timing the inference ---

print(model)

for name, param in model.named_parameters():
    print(name, param.shape)

# Convert to hls4ml
import hls4ml
import plotting

config = hls4ml.utils.config_from_pytorch_model(model, input_shape=(16,), granularity='name', backend='VivadoAccelerator')
#config['LayerName']['softmax']['exp_table_t'] = 'ap_fixed<18,8>' # Modified to optimize
#config['LayerName']['softmax']['inv_table_t'] = 'ap_fixed<18,4>' # Modified to optimize
config['Model']['Precision'] = 'ap_fixed<16,6>' # Modified to optimize
for layer in ['fc1']: #v2
    config['LayerName'][layer]['ReuseFactor'] = 64 # Modified to optimize
print("-----------------------------------")
print("Configuration")
plotting.print_dict(config)
print("-----------------------------------")
hls_model = hls4ml.converters.convert_from_pytorch_model(
    model, hls_config=config, output_dir='model_1/hls4ml_prj_pynq_matteo', backend='VivadoAccelerator', board='pynq-z2' # Even though we are using PYNQ-Z1, we can specify PYNQ-Z2 here because it has the same FPGA
)
hls_model.compile()

plotting.print_dict(hls4ml.backends.get_backend('VivadoAccelerator').create_initial_config())

# Predict
y_test = np.load('y_test.npy', allow_pickle=True)
y_hls = hls_model.predict(np.ascontiguousarray(X_test))

# Calculate error
mse = np.mean((y_hls - y_test) ** 2)
rmse = np.sqrt(mse)
#print(f"RMSE: {rmse}")

# Calculate WMAPE error
def wmape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))

print("WMAPE of HLS model: ", wmape(y_test, y_hls))
print("WMAPE of PyTorch model: ", wmape(y_test, y_torch.cpu().numpy()))

np.save('model_1/pytorch_best_model.pth', y_hls)

print("output shape: ", y_torch.shape, y_hls.shape, y_test.shape)

# Synthesize and make bitfile - this line might take a long while
# You can comment this line if you don't have Vivado installed or if you just want to test the prediction without synthesizing the hardware
# hls_model.build(csim=False, export=True, bitfile=True)
