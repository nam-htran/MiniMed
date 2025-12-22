# run/train_verifier.py
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from pathlib import Path
from src.models.verifier import MultiSignalVerifier

# --- CONFIG ---
# --- NÂNG CẤP: Tăng INPUT_DIM từ 7 lên 8 ---
INPUT_DIM = 8 # Features: [nli, gcot, in_kg, causality, len, src_deg, tgt_deg, provenance]
MODEL_PATH = Path("models/verifier_weights.pth")
MODEL_PATH.parent.mkdir(exist_ok=True)

def create_dummy_data(num_samples=1000):
    """Tạo dữ liệu giả lập để huấn luyện."""
    print("... Creating dummy training data ...")
    
    # Positive samples (good paths)
    pos_features = np.random.rand(num_samples // 2, INPUT_DIM)
    pos_features[:, 0] = np.random.uniform(0.7, 1.0, num_samples // 2) # High NLI
    pos_features[:, 2] = 1.0 # In KG
    pos_features[:, 7] = np.random.uniform(0.8, 1.0, num_samples // 2) # High Provenance (PrimeKG, PSG)
    pos_labels = np.ones(num_samples // 2)

    # Negative samples (bad paths)
    neg_features = np.random.rand(num_samples // 2, INPUT_DIM)
    neg_features[:, 0] = np.random.uniform(0.0, 0.4, num_samples // 2) # Low NLI
    neg_features[:, 7] = np.random.uniform(0.3, 0.5, num_samples // 2) # Low Provenance (RTX-KG2, Default)
    neg_labels = np.zeros(num_samples // 2)

    X = np.vstack([pos_features, neg_features])
    y = np.concatenate([pos_labels, neg_labels])

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def main():
    # ... (Logic huấn luyện giữ nguyên)
    print("🚀 Starting Verifier Model Training...")
    X_train, y_train = create_dummy_data()
    dataset = TensorDataset(X_train, y_train.unsqueeze(1))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = MultiSignalVerifier(input_dim=INPUT_DIM)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 20
    for epoch in range(epochs):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✅ Model weights saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()