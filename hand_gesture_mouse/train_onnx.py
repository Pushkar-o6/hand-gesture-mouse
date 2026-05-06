import os
import argparse
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    print("PyTorch is not installed. Please run: pip install torch")
    exit(1)

# Sequence Shape parameters (Must match our recorder & inference engine)
SEQ_LEN = 15
NUM_FEATURES = 42 
NUM_CLASSES = 5

class GestureLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(GestureLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # x is (Batch, Seq_Len, Input_Dim)
        out, (hn, cn) = self.lstm(x)
        # We only care about the last output in the sequence
        out = out[:, -1, :] 
        out = self.fc(out)
        out = self.softmax(out)
        return out

def train_and_export(data_path="sequence_samples.npz", out_path="temporal_gesture_model.onnx", epochs=100):
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please run record_sequences.py first.")
        return
        
    print(f"Loading data from {data_path}...")
    data = np.load(data_path, allow_pickle=True)
    X = data['X'].astype(np.float32)  # Should be (N, 15, 42)
    y = data['y'].astype(np.int64)    # Should be (N,)
    
    print(f"Loaded {len(X)} samples.")
    
    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y)
    
    model = GestureLSTM(input_dim=NUM_FEATURES, hidden_dim=64, num_classes=NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    print("Training Model...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == y_tensor).sum().item()
            acc = 100 * correct / len(y_tensor)
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {loss.item():.4f}, Acc: {acc:.2f}%")

    print("\nTraining Complete! Exporting to ONNX...")
    model.eval()
    
    # Exporting requires a dummy input tensor matching the correct shape
    dummy_input = torch.randn(1, SEQ_LEN, NUM_FEATURES)
    
    torch.onnx.export(
        model, 
        dummy_input, 
        out_path, 
        export_params=True,
        opset_version=11, 
        do_constant_folding=True,
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    print(f"[Success] ONNX Model exported to '{out_path}'")
    print("Drop this file into the 'hand_gesture_mouse' folder for the app to recognize it!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="sequence_samples.npz", help="Input npz file")
    parser.add_argument("--out", default="temporal_gesture_model.onnx", help="Output ONNX model name")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training iterations")
    args = parser.parse_args()
    
    train_and_export(args.data, args.out, args.epochs)