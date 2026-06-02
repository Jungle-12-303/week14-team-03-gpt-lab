# -*- coding: utf-8 -*-
"""Multi-Head Self-Attention 과제 템플릿."""

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    GPT의 causal self-attention을 구현합니다.

    구현할 핵심:
    - Q/K/V projection
    - head 분리: (B, T, C) -> (B, n_heads, T, head_dim)
    - attention score = QK^T / sqrt(head_dim)
    - causal mask로 미래 토큰 가리기
    - attention weight와 V를 곱한 뒤 head를 다시 합치기
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        drop_rate: float = 0.1,
        qkv_bias: bool = False,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # DONE: qkv projection, output projection, dropout을 정의하세요.
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=qkv_bias) # 입력 hidden state에서 Q/K/V를 한 번에 만드는 선형층입니다.
        self.out_proj = nn.Linear(d_model, d_model) # 여러 head를 합친 뒤 다시 d_model 차원으로 섞는 출력 projection입니다.
        self.attn_dropout = nn.Dropout(drop_rate) # attention 확률에 적용하는 dropout입니다.
        self.out_dropout = nn.Dropout(drop_rate) # 최종 projection 결과에 적용하는 dropout입니다.

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: bool = True,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        DONE: multi-head attention forward를 실행합니다.

        Args:
            x: (batch_size, seq_len, d_model)
            causal_mask: True이면 미래 위치를 볼 수 없게 mask 처리
            return_attention_weights: True이면 attention weight도 함께 반환
        """
        # 입력 shape를 과제 표기와 맞춰 B, T, C로 읽습니다.
        batch_size, seq_len, d_model = x.shape
        if d_model != self.d_model:
            raise ValueError("input last dimension must match d_model")

        # x에서 Q, K, V를 만들고 마지막 차원을 기준으로 3개로 나눕니다.
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # 각 tensor를 (B, T, C)에서 (B, n_heads, T, head_dim)으로 바꿉니다.
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # QK^T를 head_dim의 제곱근으로 나눠 attention score를 안정화합니다.
        attn_scores = q @ k.transpose(-2, -1)
        attn_scores = attn_scores / (self.head_dim**0.5)

        if causal_mask:
            # 현재 위치가 미래 토큰을 보지 못하도록 주대각선 위 영역을 가립니다.
            future_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
                diagonal=1,
            )
            attn_scores = attn_scores.masked_fill(future_mask, float("-inf"))

        # score를 확률 분포로 바꿔 각 토큰이 볼 위치의 가중치를 만듭니다.
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_weights)

        context = attn_probs @ v # attention weight로 V를 가중합해 각 head의 출력을 만듭니다.


        # head 차원을 다시 합쳐 (B, T, C) 형태로 되돌립니다.
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.d_model)

        # output projection과 dropout을 적용해 최종 attention 출력을 만듭니다.
        output = self.out_proj(context)
        output = self.out_dropout(output)

        if return_attention_weights:
            # 테스트에서 causal mask를 확인할 수 있도록 attention weight도 반환합니다.
            return output, attn_weights
        return output        