import torch
import torch.nn as nn

class TinyRecursiveModel(nn.Module):
    """
    Triển khai kiến trúc TRM dựa trên paper 'Less is More'.
    Sử dụng 1 mạng tiny network f(.) duy nhất để đệ quy.
    """
    def __init__(self, input_dim=384, hidden_dim=512, output_vocab_size=10000): # input_dim khớp với MiniLM
        super().__init__()
        
        # Embedding projection (cho x và z)
        self.embedding_proj = nn.Linear(input_dim, hidden_dim)
        
        # Tiny Network (2 layers as per paper)
        self.tiny_net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), # Input: [z_current, y_current, x]
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Output Head (Project latent z về vocabulary hoặc embedding space của câu trả lời)
        self.output_head = nn.Linear(hidden_dim, output_vocab_size) 

    def forward(self, x_emb, initial_z_emb, n_steps=6, t_loops=3):
        """
        x_emb: Embedding của Question
        initial_z_emb: Embedding của Reasoning (MedCOT trace)
        """
        batch_size = x_emb.size(0)
        device = x_emb.device
        
        # Khởi tạo y (answer) ban đầu là random hoặc zero
        y_curr = torch.zeros_like(x_emb).to(device)
        z_curr = initial_z_emb
        
        # Deep Supervision Outputs
        outputs = []
        
        # Vòng lặp đệ quy theo paper (Algorithm 3)
        # TRM chạy T lần, mỗi lần n bước cập nhật z
        for t in range(t_loops):
            # 1. Update latent z (Reasoning improvement)
            for _ in range(n_steps):
                # Input concat: [z, y, x]
                combined = torch.cat([z_curr, y_curr, x_emb], dim=-1)
                z_new = self.tiny_net(combined)
                z_curr = z_new + z_curr # Residual connection
            
            # 2. Update y (Answer generation based on refined z)
            # Trong paper, y được update bởi f_H (ở đây ta dùng chung mạng tiny_net hoặc 1 nhánh riêng)
            # Ở đây ta đơn giản hóa theo biến thể Single Network của TRM
            combined_y = torch.cat([z_curr, y_curr, x_emb], dim=-1)
            y_new_feat = self.tiny_net(combined_y)
            y_curr = y_new_feat # Latent của Y
            
            # Dự đoán output tại step này (Deep Supervision)
            logits = self.output_head(y_curr)
            outputs.append(logits)
            
        return outputs # List các outputs qua từng vòng lặp T

