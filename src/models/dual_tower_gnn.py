# src/models/dual_tower_gnn.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear

class CoGCoT_DualTower_GNN(torch.nn.Module):
    """
    Graph Chain-of-Thought Dual Tower GNN (Robust version for edgeless PSG)
    """
    def __init__(self, ckg_metadata, psg_metadata, hidden_channels, query_dim, num_think_steps=2):
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.num_think_steps = num_think_steps

        # --- CKG Tower ---
        self.ckg_lin_dict = torch.nn.ModuleDict()
        for node_type in ckg_metadata[0]:
            self.ckg_lin_dict[node_type] = Linear(-1, hidden_channels)
        
        self.ckg_convs = torch.nn.ModuleList()
        for _ in range(num_think_steps):
            conv = HGTConv(hidden_channels, hidden_channels, ckg_metadata, heads=2)
            self.ckg_convs.append(conv)

        # --- PSG Tower (nếu có) ---
        self.use_psg = bool(psg_metadata[0])
        if self.use_psg:
            self.psg_lin_dict = torch.nn.ModuleDict()
            for node_type in psg_metadata[0]:
                self.psg_lin_dict[node_type] = Linear(-1, hidden_channels)
            # Chỉ khởi tạo conv nếu PSG có cạnh, nếu không sẽ gây lỗi metadata
            if psg_metadata[1]:
                self.psg_conv = HGTConv(hidden_channels, hidden_channels, psg_metadata, heads=2)
            else:
                self.psg_conv = None

        # --- Interaction Layer ---
        self.query_proj = Linear(query_dim, hidden_channels)

    def forward(self, ckg_data, psg_data, query_emb):
        # 1. Initial Projection for CKG nodes
        ckg_x_dict = {
            node_type: self.ckg_lin_dict[node_type](ckg_data[node_type].x).relu()
            for node_type in ckg_data.node_types
        }

        # 2. Process PSG Tower (if applicable and has edges)
        psg_context_vector = None
        
        # --- SỬA ĐỔI QUAN TRỌNG ---
        # Chỉ xử lý tháp PSG nếu nó tồn tại, có model conv, và quan trọng nhất: CÓ CẠNH
        if self.use_psg and self.psg_conv is not None and psg_data.edge_types:
            psg_x_dict = {
                node_type: self.psg_lin_dict[node_type](psg_data[node_type].x).relu()
                for node_type in psg_data.node_types
            }
            # Chỉ gọi conv nếu có cạnh
            psg_x_dict = self.psg_conv(psg_x_dict, psg_data.edge_index_dict)
            
            # Aggregate PSG info into a single context vector
            all_psg_embs = torch.cat([x for x in psg_x_dict.values()], dim=0)
            psg_context_vector = torch.mean(all_psg_embs, dim=0, keepdim=True)
        # ---------------------------

        # 3. Project Query
        query_vector = self.query_proj(query_emb.unsqueeze(0)).relu()

        # 4. Iterative Reasoning (Graph Chain-of-Thought)
        thought_vectors = []
        for i in range(self.num_think_steps):
            fused_query_vector = query_vector
            if psg_context_vector is not None:
                fused_query_vector = fused_query_vector + psg_context_vector

            for node_type in ckg_x_dict:
                ckg_x_dict[node_type] = ckg_x_dict[node_type] + fused_query_vector

            ckg_x_dict = self.ckg_convs[i](ckg_x_dict, ckg_data.edge_index_dict)
            
            current_thought = torch.mean(torch.cat([x for x in ckg_x_dict.values()]), dim=0)
            thought_vectors.append(current_thought.cpu().numpy())

        return ckg_x_dict, thought_vectors