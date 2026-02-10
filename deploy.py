from axi_stream_driver import NeuralNetworkOverlay
import numpy as np

X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

nn = NeuralNetworkOverlay('hls4ml_nn.bit', X_test.shape, y_test.shape)

y_hw, latency, throughput = nn.predict(X_test, profile=True)


# Calculate MSE and RMSE errors
mse = np.mean((y_hw - y_test) ** 2)
rmse = np.sqrt(mse)
#print(f"RMSE: {rmse}")

# Calculate WMAPE error
def wmape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))

print("WMAPE: ", wmape(y_test, y_hw))

np.save('y_hw.npy', y_hw)
