# -*- coding: utf-8 -*-
"""GPT 모델 구성 요소 과제 템플릿."""

import torch
import torch.nn as nn

try:
    from .attention import MultiHeadAttention
    from .embeddings import InputEmbedding
except ImportError:
    from attention import MultiHeadAttention
    from embeddings import InputEmbedding

# 토큰 벡터를 정규화
# 트랜스포머 block 과정에서 토큰 벡터 값이 너무 커지거나 작아질 수 있다
class LayerNorm(nn.Module):
    """마지막 차원 기준 Layer Normalization."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    # [2,4,6]
    # [1,3,5]
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """TODO: 마지막 차원의 평균과 분산으로 정규화한 뒤 gamma/beta를 적용합니다."""
        
        # mean = [[4.0]
        #        [3.0]]
        mean = x.mean(dim = -1, keepdim = True)
        
        # [2.667]
        # [2.667]
        var = x.var(dim = -1, keepdim = True, unbiased = False)

        # 정규화
        # 각 값에서 평균 뺴고 표준편차로 나눈다
        # [[-1.2247, 0.0000, 1.2247],
        # [-1.2247, 0.0000, 1.2247]]
        norm_x = (x - mean) / torch.sqrt(var + self.eps)

        # gamma, beta 적용
        # 학습하면서 gamma, beta를 바꾼다
        output = self.gamma * norm_x + self.beta

        # raise NotImplementedError("LayerNorm.forward를 구현하세요.")

# 큰 양수는 거의 그대로 통과
# 큰 음수는 거의 0에 가깝게 줄인다
# 0 근처는 부드럽게 조절
# 부드럽다는 거 -> 학습할 때 값이 급격하게 죽거나 튀는 현상이 줄어든다
class GELU(nn.Module):
    """GPT FeedForward에서 사용하는 GELU 활성화 함수."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """TODO: tanh 근사식 또는 torch 연산으로 GELU를 구현합니다."""
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0/torch.pi)) *
                                        (x + 0.044715 * torch.pow(x,3))
                        ))
        # raise NotImplementedError("GELU.forward를 구현하세요.")


class FeedForward(nn.Module):
    """Transformer FFN: Linear -> GELU -> Linear -> Dropout."""

    def __init__(self, d_model: int, dropout: float = 0.1, mult: int = 4):
        super().__init__()
        # TODO: d_model -> mult*d_model -> d_model 구조의 작은 MLP를 정의하세요.
        self.layers = nn.Sequential(
            nn.Linear(d_model, mult * d_model), #토큰 벡터를 4차원에서 16차원을 키운다
            GELU(), #활성화 함수, 큰 양수는 거의 그대로 통과/큰 음수는 거의 0에 가깝게 줄임/0 근처 값은 부드럽게 조절
            nn.Linear(mult * d_model, d_model), # 16차원의 토큰 벡터를 4차원으로 줄인다
            nn.Dropout(dropout) #학습 중에 일부 값을 랜덤하게 0으로 만든다
        )

        # raise NotImplementedError("FeedForward.__init__을 구현하세요.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """TODO: FeedForward 네트워크를 통과시킵니다."""
        return self.layers(x) #실제 계산은 여기서 일어난다
        # raise NotImplementedError("FeedForward.forward를 구현하세요.")


class TransformerBlock(nn.Module):
    """
    GPT block: LayerNorm -> Causal Self-Attention -> residual,
    LayerNorm -> FeedForward -> residual.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        drop_rate: float = 0.1,
        qkv_bias: bool = False,
    ):
        super().__init__()
        # TODO: attention, ffn, layernorm, dropout을 정의하세요.
        self.att = MultiHeadAttention(
            d_model = d_model,
            n_heads = n_heads,
            dropout = drop_rate,
            qkv_bias = qkv_bias,
        )

        # 각 토큰 벡터를 
        self.ffn = FeedForward(
            d_model = d_model,
            dropout = drop_rate,
        )
        
        #각 토큰 벡터 값 분포 안정화
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.drop_shortcut = nn.Dropout(drop_rate)
        # raise NotImplementedError("TransformerBlock.__init__을 구현하세요.")

    def forward(self, x: torch.Tensor, causal_mask: bool = True) -> torch.Tensor:
        """TODO: attention과 ffn을 residual connection으로 연결합니다."""
        # 1. Attention block + residaul connection
        shortcut = x
        x = self.norm1(x) #attention에 넣기전 LayerNorm
        x = self.att(x, causal_mask = causal_mask) #attention 적용
        x = self.drop_shortcut(x) #
        x = x + shortcut #숏컷 연결

        # 2. FeedForward block + residual connection
        shortcut = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x
        # raise NotImplementedError("TransformerBlock.forward를 구현하세요.")


class GPTModel(nn.Module):
    """InputEmbedding -> TransformerBlock N개 -> LayerNorm -> LM head."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        # TODO: embedding, blocks, final layernorm, lm_head를 정의하세요.
        raise NotImplementedError("GPTModel.__init__을 구현하세요.")

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: logits를 만들고, targets가 있으면 cross entropy loss도 함께 반환합니다.

        Returns:
            targets가 None이면 logits
            targets가 있으면 (loss, logits)
        """
        raise NotImplementedError("GPTModel.forward를 구현하세요.")


def generate_text_simple(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
) -> torch.Tensor:
    """TODO: greedy 방식으로 max_new_tokens만큼 다음 토큰을 이어 붙입니다."""
    raise NotImplementedError("generate_text_simple을 구현하세요.")
