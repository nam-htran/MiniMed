# src/models/verifier.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiSignalVerifier(nn.Module):
    """
    Một meta-model nhỏ để học cách kết hợp các tín hiệu verification.
    Input: Một vector chứa các điểm số từ các tín hiệu khác nhau (NLI, GCoT, etc.).
    Output: Một điểm tin cậy duy nhất (0-1).
    """
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super(MultiSignalVerifier, self).__init__()
        
        # Kiến trúc đơn giản nhưng hiệu quả: MLP 2 lớp
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), # Giúp ổn định quá trình học
            nn.ReLU(),
            nn.Dropout(0.3),            # Chống overfitting
            nn.Linear(hidden_dim, 1)    # Output là một logit duy nhất
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Tensor kích thước (batch_size, input_dim) chứa các tín hiệu.
        
        Returns:
            torch.Tensor: Logit score cho mỗi path.
        """
        return self.net(x)

