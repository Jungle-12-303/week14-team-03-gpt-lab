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
        vocab_size: int, #어휘 사전 크기
        emb_dim: int, #각 토큰을 몇 차원 벡터로 바꿀지 의미
        context_length: int, #모델이 한 번에 볼 수있는 최대 토큰 개수
        drop_rate: float = 0.1, #dropout 비율
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.context_length = context_length #모델이 한 번에 보는 토큰 개수
        # TODO: token_embedding, position_embedding, dropout을 정의하세요

        # 토큰 id vocab_size 수만큼 몇 차원 벡터를 하나씩 만들어라
        self.token_embedding = nn.Embedding(vocab_size, emb_dim)

        #위치 context_length 수만큼 몇 차원 벡터를 하나씩 만들어라
        self.position_embedding = nn.Embedding(context_length, emb_dim)

        #학습할 때 일부 값을 랜덤하게 0으로 만드는 층
        self.dropout = nn.Dropout(drop_rate)


        # raise NotImplementedError("InputEmbedding.__init__을 구현하세요.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        TODO: token embedding과 position embedding을 더한 뒤 dropout을 적용합니다.

        Args:
            x: (batch_size, seq_len) token IDs

        Returns:
            (batch_size, seq_len, emb_dim)
        """

        # x = [2,5,1]
        #    [4,3,0]
        # 행  열  가져온다
        batch_size, seq_len = x.shape

        # x의 각 행에 해당하는 벡터들을 가져와서 저장한다
        token_embeds = self.token_embedding(x)

        # positions = tenser([0,1,2])
        positions = torch.arange(seq_len, device=x.device)

        # positions의 각 행에 해당하는 벡터들 가져와서 저장한다
        position_embeds = self.position_embedding(positions)

        embeddings = token_embeds + position_embeds

        return self.dropout(embeddings)
        # raise NotImplementedError("InputEmbedding.forward를 구현하세요.")
