# -*- coding: utf-8 -*-
"""토큰 임베딩 + 위치 임베딩 과제 템플릿."""

import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    """
    token ID를 Transformer 입력 벡터로 바꿉니다.

    구현할 구조:
    - token embedding: nn.Embedding(vocab_size, emb_dim)
    - position embedding: nn.Embedding(context_length, emb_dim)
    - token embedding + position embedding
    - dropout
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        context_length: int,
        drop_rate: float = 0.1,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.context_length = context_length
        # TODO: token_embedding, position_embedding, dropout을 정의하세요.
        # 토근임베딩, 위치임베딩을 토큰id를 통해 테이블을 랜덤값으로 생성
        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        self.position_embedding = nn.Embedding(context_length, emb_dim)
        self.dropout = nn.Dropout(drop_rate)

    """
    ex) 
    x = [
        [10, 20, 30],
        [40, 50, 60].
    ]
    emb_dim = 3 (벡터 차원의 개수 3)
    token_emb: (2, 3, 3)    
    tokenN_vector = [N-1, N-2, N-3] 
    [
        [
            token10_vector, 
            token20_vector, 
            token40_vector, 
        ],
        [
            token40_vector, 
            token50_vector, 
            token60_vector, 
        ]
    ]

    positions = [0, 1, 2]
    [
        position0_vector,
        position1_vector,
        position2_vector,
    ]
    
    첫 번째 샘플:
    token10_vector + position0_vector
    token20_vector + position1_vector
    token30_vector + position2_vector

    두 번째 샘플:
    token40_vector + position0_vector
    token50_vector + position1_vector
    token60_vector + position2_vector
    
    token10_vector = [0.20, 0.50, -0.10]
    position0_vector = [0.01, -0.03, 0.04]
    ->[0.21, 0.47, -0.06]
    """
    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        TODO: token embedding과 position embedding을 더한 뒤 dropout을 적용합니다.        
        Args:
            x: (batch_size, seq_len) token IDs

        Returns:
            (batch_size, seq_len, emb_dim)
        """
        seq_len = x.shape[1]
        
        token_emb = self.token_embedding(x)
        
        positions = torch.arange(seq_len, device=x.device)
        pos_emb = self.position_embedding(positions)
        
        out = token_emb + pos_emb

        return self.dropout(out)

