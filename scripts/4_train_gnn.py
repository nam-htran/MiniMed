# run/train_gnn_next_hop.py
import logging
import torch
import torch.optim as optim
import os
import glob
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from src.models.dual_tower_gnn import CoGCoT_DualTower_GNN
import numpy as np

logging.basicConfig(level=logging.CRITICAL)

# --- CONFIG ---
DATA_DIR = "data/processed_gnn_data"
OUTPUT_PATH = "models/gnn_dual_tower_weights.pth"
EPOCHS = 10
LEARNING_RATE = 1e-4

def generate_training_samples_from_tensor(ckg_data, gold_path_text):
    # Logic tạo sample từ data đã pre-process
    # Đây là logic giả lập: Map text path -> node index trong ckg_data
    # Thực tế cần mapping chính xác hơn từ ID node sang index
    # Ở đây ta dùng random negative sampling đơn giản để demo
    try:
        # Lấy danh sách node types
        ntypes = ckg_data.node_types
        if not ntypes: return []
        
        # Chọn đại diện 1 node type chính (VD: Disease)
        target_ntype = ntypes[0]
        num_nodes = ckg_data[target_ntype].x.shape[0]
        
        if num_nodes < 2: return []
        
        # Random Positive pair (Giả lập Next Hop)
        src_idx = np.random.randint(0, num_nodes)
        pos_idx = np.random.randint(0, num_nodes)
        
        # Negative samples
        neg_indices = np.random.choice(num_nodes, 5, replace=True).tolist()
        
        return [(target_ntype, src_idx, pos_idx, neg_indices)]
    except:
        return []

def main():
    print("🚀 Starting Optimized GNN Training...")
    
    files = glob.glob(os.path.join(DATA_DIR, "*.pt"))
    if not files:
        print(f"❌ No .pt files found in {DATA_DIR}. Run prepare_gnn_dataset.py first.")
        return

    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Load 1 sample để init model
    sample_0 = torch.load(files[0])
    model = CoGCoT_DualTower_GNN(
        sample_0['ckg_data'].metadata(), 
        sample_0['psg_data'].metadata(), 
        hidden_channels=128, query_dim=384
    )
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0
        valid_batches = 0
        
        # Shuffle files
        np.random.shuffle(files)
        
        pbar = tqdm(files, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for file_path in pbar:
            try:
                data = torch.load(file_path)
                ckg_data = data['ckg_data']
                psg_data = data['psg_data']
                query_text = data['query_text']
                
                query_emb = encoder.encode(query_text, convert_to_tensor=True)
                
                # Forward Pass
                node_embs, _ = model(ckg_data, psg_data, query_emb)
                
                # Generate Samples & Compute Loss
                samples = generate_training_samples_from_tensor(ckg_data, data['target_path'])
                
                batch_loss = 0
                for ntype, src_idx, pos_idx, neg_idxs in samples:
                    if ntype not in node_embs: continue
                    
                    emb_matrix = node_embs[ntype]
                    src_emb = emb_matrix[src_idx]
                    pos_emb = emb_matrix[pos_idx]
                    
                    pos_score = torch.dot(src_emb, pos_emb)
                    
                    for neg_idx in neg_idxs:
                        neg_emb = emb_matrix[neg_idx]
                        neg_score = torch.dot(src_emb, neg_emb)
                        batch_loss += loss_fn(pos_score - neg_score, torch.tensor(1.0))
                
                if batch_loss != 0:
                    optimizer.zero_grad()
                    batch_loss.backward()
                    optimizer.step()
                    total_loss += batch_loss.item()
                    valid_batches += 1
                    
                pbar.set_postfix({'loss': total_loss / (valid_batches + 1e-9)})
                
            except Exception:
                continue

        print(f"Epoch {epoch+1} Avg Loss: {total_loss / (valid_batches + 1e-9):.4f}")

    torch.save(model.state_dict(), OUTPUT_PATH)
    print(f"✅ GNN Model saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()