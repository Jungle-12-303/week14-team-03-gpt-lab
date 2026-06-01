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
        vocab_size: int, # 사전 크기
        emb_dim: int, # 임베딩 크기
        context_length: int,
        drop_rate: float = 0.1,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.context_length = context_length
        # DONE: token_embedding, position_embedding, dropout을 정의하세요.
        self.token_embedding = nn.Embedding(vocab_size, emb_dim) # (사전 크기 X 임베딩 차원)만큼 토큰 임베딩
        self.position_embedding = nn.Embedding(context_length, emb_dim) # (컨텍스트 길이 X 임베딩 차원)만큼 위치 임베딩
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        TODO: token embedding과 position embedding을 더한 뒤 dropout을 적용합니다.

        Args:
            x: (batch_size, seq_len) token IDs

        Returns:
            (batch_size, seq_len, emb_dim)
        """
        seq_len = x.size(1) # 또는 seq_len = x.shape[1] 도 가능

        tok_emb = self.token_embedding(x)

        positions = torch.arange(seq_len, device=x.device) # device=x.device: positions 텐서를 x와 같은 장치에 생성
        pos_emb = self.position_embedding(positions)

        x = tok_emb + pos_emb
        x = self.dropout(x)

        return x
