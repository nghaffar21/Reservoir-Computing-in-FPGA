# This code refines the data, creates an NN PyTorch model, and converts it to HLS and syntheses it for FPGA deployment, without generating the bitstream.
# It is the modified version of tutorial 1 of the hls4ml official tutorials: https://github.com/fastmachinelearning/hls4ml-tutorial/tree/main 
#
# The modification is that the original documentation is written in Keras, but here we use PyTorch.
# 
# Written by: Nima Ghaffarzadeh, CTTC, Barcelona, Spain, Feb 2026

# Getting started
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time

# Fetch the jet tagging dataset from Open ML
data = fetch_openml('hls4ml_lhc_jets_hlf')
X, y = data['data'], data['target']

# Print information about the dataset
print(data['feature_names'])
print(X.shape, y.shape)
print(X[:5])
print(y[:5])

# One-hot encoding and train-test split
le = LabelEncoder()
y = le.fit_transform(y)
y_categorical = np.eye(5)[y]  # One-hot encoding in numpy
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)
print(y_categorical[:5])

# Standardize the features
scaler = StandardScaler()
X_train_val = scaler.fit_transform(X_train_val)
X_test = scaler.transform(X_test)

# Save the processed data
np.save('X_train_val.npy', X_train_val)
np.save('X_test.npy', X_test)
np.save('y_train_val.npy', y_train_val)
np.save('y_test.npy', y_test)
np.save('classes.npy', le.classes_)

# Define the PyTorch model
class JetTaggingModel(nn.Module):
    def __init__(self):
        super(JetTaggingModel, self).__init__()
        # LeCun uniform initialization approximation
        self.fc1 = nn.Linear(16, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 32)
        self.relu3 = nn.ReLU()
        self.output = nn.Linear(32, 5)
        self.softmax = nn.Softmax(dim=1)
        
        # Apply LeCun uniform initialization and L1 regularization setup
        self._init_weights()
        
    def _init_weights(self):
        # LeCun uniform initialization
        for m in [self.fc1, self.fc2, self.fc3, self.output]:
            nn.init.kaiming_uniform_(m.weight, nonlinearity='linear')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.output(x)
        x = self.softmax(x)
        return x

# Instantiate the model
model = JetTaggingModel()

# L1 regularization function
def l1_regularization(model, lambda_l1=0.0001):
    l1_loss = 0
    for param in model.parameters():
        l1_loss += torch.sum(torch.abs(param))
    return lambda_l1 * l1_loss

# Training setup
train = True
if train:
    # Split training data into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42
    )
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)
    
    # Define optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, 
        eps=0.000001, cooldown=2, min_lr=0.0000001
    )
    
    # Training loop
    num_epochs = 10
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            
            outputs = model(batch_X)
            # CrossEntropyLoss expects class indices, not one-hot
            loss = criterion(outputs, batch_y)
            
            # Add L1 regularization
            loss += l1_regularization(model, lambda_l1=0.0001)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            _, labels = torch.max(batch_y, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss += l1_regularization(model, lambda_l1=0.0001)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                _, labels = torch.max(batch_y, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'model_1/pytorch_best_model.pth')
            print(f'Saved best model with val_loss: {best_val_loss:.4f}')
else:
    model.load_state_dict(torch.load('model_1/pytorch_best_model.pth'))

# Check performance
from sklearn.metrics import accuracy_score

model.eval()
X_test_tensor = torch.FloatTensor(X_test)

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

with torch.no_grad():
    y_pytorch = model(X_test_tensor).numpy()

print("Accuracy: {}".format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pytorch, axis=1))))

# Calculate WMAPE error
def wmape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))

print("WMAPE: ", wmape(y_test, y_pytorch))

# ROC curve (assuming you have the plotting module)
import plotting
import matplotlib.pyplot as plt
plt.figure(figsize=(9, 9))
_ = plotting.makeRoc(y_test, y_pytorch, le.classes_)

import hls4ml

config = hls4ml.utils.config_from_pytorch_model(model, input_shape=(16,), granularity='name', backend='VivadoAccelerator')
print("-----------------------------------")
print("Configuration")
plotting.print_dict(config)
print("-----------------------------------")
hls_model = hls4ml.converters.convert_from_pytorch_model(
    model, hls_config=config, backend='VivadoAccelerator', output_dir='model_1/hls4ml_prj', part='xc7z020clg400-1'
)

hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file=None)

# Compile and predict with HLS model
hls_model.compile()
X_test = np.ascontiguousarray(X_test)
y_hls = hls_model.predict(X_test)

# Compile and predict - NO CHANGES NEEDED
hls_model.compile()
X_test = np.ascontiguousarray(X_test)
y_hls = hls_model.predict(X_test)

# Compare
print("PyTorch Accuracy: {}".format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pytorch, axis=1))))
print("hls4ml Accuracy: {}".format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_hls, axis=1))))

fig, ax = plt.subplots(figsize=(9, 9))
_ = plotting.makeRoc(y_test, y_pytorch, le.classes_)
plt.gca().set_prop_cycle(None)  # reset the colors
_ = plotting.makeRoc(y_test, y_hls, le.classes_, linestyle='--')

from matplotlib.lines import Line2D

lines = [Line2D([0], [0], ls='-'), Line2D([0], [0], ls='--')]
from matplotlib.legend import Legend

leg = Legend(ax, lines, labels=['pytorch', 'hls4ml'], loc='lower right', frameon=False)
ax.add_artist(leg)

hls4ml.report.read_vivado_report('model_1/hls4ml_prj/')

# Synthesize and make bitfile - this line might take a long while
# You can comment this line if you don't have Vivado installed or if you just want to test the prediction without synthesizing the hardware
hls_model.build(csim=False)

