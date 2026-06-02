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


class LayerNorm(nn.Module):
    """마지막 차원 기준 Layer Normalization."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """DONE: 마지막 차원의 평균과 분산으로 정규화한 뒤 gamma/beta를 적용합니다."""
        mean = x.mean(dim=-1, keepdim=True)  # 각 토큰 벡터의 마지막 차원(d_model) 평균을 구해 중심을 0 근처로 옮길 준비를 합니다.
        var = x.var(dim=-1, keepdim=True, unbiased=False) # 각 토큰 벡터의 마지막 차원 분산을 구하며, LayerNorm에서는 표본분산이 아닌 모집단분산을 씁니다.
        norm_x = (x - mean) / torch.sqrt(var + self.eps) # 평균을 빼고 표준편차로 나누어 값의 스케일을 평균 0, 분산 1 근처로 맞춥니다.
        return self.gamma * norm_x + self.beta # 학습 가능한 gamma(scale)/beta(shift)로 정규화된 값을 다시 유연하게 확대·이동합니다.


class GELU(nn.Module):
    """GPT FeedForward에서 사용하는 GELU 활성화 함수."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """DONE: tanh 근사식 또는 torch 연산으로 GELU를 구현합니다."""
        sqrt_2_over_pi = torch.sqrt(torch.tensor(2.0 / torch.pi, device=x.device, dtype=x.dtype))  # GELU tanh 근사식에 쓰이는 sqrt(2/pi) 상수를 입력과 같은 장치·dtype으로 만듭니다.
        cubic_term = 0.044715 * torch.pow(x, 3)  # GELU 근사식이 ReLU보다 부드러운 곡선을 만들도록 x^3 보정항을 계산합니다.
        inner = sqrt_2_over_pi * (x + cubic_term)  # tanh 안에 들어갈 값을 만들며, 입력 x와 cubic 보정항을 함께 사용합니다.
        return 0.5 * x * (1.0 + torch.tanh(inner))  # 입력을 확률적으로 통과시키는 것처럼 부드럽게 게이팅하는 GELU 출력을 반환합니다.


class FeedForward(nn.Module):
    """Transformer FFN: Linear -> GELU -> Linear -> Dropout."""

    def __init__(self, d_model: int, dropout: float = 0.1, mult: int = 4):
        super().__init__()
        # TODO: d_model -> mult*d_model -> d_model 구조의 작은 MLP를 정의하세요.
        hidden_dim = mult * d_model  # GPT 계열 FFN은 보통 임베딩 차원을 4배로 넓혀 더 풍부한 중간 표현을 만듭니다.
        self.layers = nn.Sequential(  # Linear, GELU, Linear, Dropout을 한 번에 순서대로 실행하는 작은 신경망을 정의합니다.
            nn.Linear(d_model, hidden_dim),  # 각 토큰 벡터를 d_model 차원에서 더 넓은 hidden_dim 차원으로 확장합니다.
            GELU(),  # 확장된 특징에 비선형성을 넣어 단순 선형 변환만으로는 표현하지 못하는 패턴을 학습하게 합니다.
            nn.Linear(hidden_dim, d_model),  # 넓어진 hidden_dim 표현을 다시 Transformer의 기본 폭인 d_model 차원으로 되돌립니다.
            nn.Dropout(dropout),  # 훈련 중 일부 값을 0으로 만들어 과적합을 줄이고 모델이 특정 뉴런에만 의존하지 않게 합니다.
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """TODO: FeedForward 네트워크를 통과시킵니다."""
        return self.layers(x)  # 배치와 시퀀스 차원은 유지한 채 마지막 d_model 차원에만 FFN을 적용합니다.


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
        self.attention = MultiHeadAttention(d_model, n_heads, drop_rate, qkv_bias)  # 한 토큰이 이전 토큰들을 참고하도록 causal multi-head self-attention을 준비합니다.
        self.ffn = FeedForward(d_model, dropout=drop_rate)  # attention이 섞은 정보를 토큰별 MLP로 한 번 더 가공하는 feed-forward 네트워크를 준비합니다.
        self.norm1 = LayerNorm(d_model)  # attention 앞에서 입력 분포를 안정화하는 첫 번째 Pre-LayerNorm입니다.
        self.norm2 = LayerNorm(d_model)  # FFN 앞에서 입력 분포를 안정화하는 두 번째 Pre-LayerNorm입니다.
        self.dropout = nn.Dropout(drop_rate)  # residual branch에 dropout을 적용해 훈련 중 과적합을 줄이는 층입니다.

    def forward(self, x: torch.Tensor, causal_mask: bool = True) -> torch.Tensor:
        """TODO: attention과 ffn을 residual connection으로 연결합니다."""
        shortcut = x  # residual connection을 위해 attention에 들어가기 전 원본 입력을 보관합니다.
        x = self.norm1(x)  # Pre-LayerNorm 방식으로 attention을 계산하기 전에 각 토큰 벡터를 정규화합니다.
        x = self.attention(x, causal_mask=causal_mask)  # 현재 토큰이 허용된 과거 문맥을 보면서 새로운 표현을 만들게 합니다.
        x = self.dropout(x)  # attention 결과 일부를 훈련 중 무작위로 끄며 residual 경로의 일반화를 돕습니다.
        x = x + shortcut  # attention 결과에 원본 입력을 더해 정보와 기울기가 깊은 층까지 잘 흐르게 합니다.
        shortcut = x  # FFN residual connection을 위해 attention까지 끝난 현재 표현을 다시 보관합니다.
        x = self.norm2(x)  # FFN에 넣기 전에 다시 정규화하여 다음 변환의 입력 스케일을 안정화합니다.
        x = self.ffn(x)  # 각 토큰 위치별로 독립적인 MLP를 적용해 attention 뒤의 표현력을 높입니다.
        x = self.dropout(x)  # FFN 결과에도 dropout을 적용해 훈련 중 더 견고한 표현을 학습하게 합니다.
        return x + shortcut  # FFN 결과와 FFN 이전 표현을 더해 Transformer block의 최종 출력을 만듭니다.


class GPTModel(nn.Module):
    """InputEmbedding -> TransformerBlock N개 -> LayerNorm -> LM head."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        # TODO: embedding, blocks, final layernorm, lm_head를 정의하세요.
        self.embedding = InputEmbedding(  # 토큰 ID를 토큰 임베딩과 위치 임베딩이 더해진 연속 벡터로 바꾸는 입력 모듈입니다.
            config["vocab_size"],  # tokenizer가 만들 수 있는 전체 토큰 ID 개수이며, token embedding 행 개수가 됩니다.
            config["emb_dim"],  # 각 토큰을 표현할 벡터 차원이며, Transformer 내부의 d_model과 같습니다.
            config["context_length"],  # 모델이 한 번에 처리할 수 있는 최대 토큰 길이이며, position embedding 길이가 됩니다.
            config["drop_rate"],  # 임베딩 결과에 적용할 dropout 비율입니다.
        )
        blocks = []  # 설정된 n_layers만큼 TransformerBlock을 만든 뒤 ModuleList로 감싸기 위한 임시 리스트입니다.
        for _ in range(config["n_layers"]):  # GPT는 같은 구조의 TransformerBlock을 여러 층 쌓아 깊은 문맥 표현을 만듭니다.
            blocks.append(  # 방금 만든 TransformerBlock을 순서대로 리스트에 넣어 앞에서 뒤로 실행되게 합니다.
                TransformerBlock(  # causal self-attention과 FFN, residual connection을 포함하는 GPT 기본 블록입니다.
                    config["emb_dim"],  # block 내부의 입력·출력 hidden 차원을 임베딩 차원과 같게 맞춥니다.
                    config["n_heads"],  # attention을 몇 개의 head로 나눌지 정해 다양한 관계를 병렬로 보게 합니다.
                    config["drop_rate"],  # block 내부 attention/FFN/residual dropout에 사용할 비율입니다.
                    config.get("qkv_bias", False),  # Q/K/V projection에 bias를 둘지 정하며, 설정이 없으면 GPT-2 스타일처럼 False를 씁니다.
                )
            )
        self.blocks = nn.ModuleList(blocks)  # PyTorch가 block들의 파라미터를 추적하도록 일반 리스트를 ModuleList로 등록합니다.
        self.final_norm = LayerNorm(config["emb_dim"])  # 모든 TransformerBlock을 지난 hidden state를 출력 head 전에 한 번 더 안정화합니다.
        self.lm_head = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)  # 각 위치의 hidden state를 전체 어휘의 다음 토큰 logits로 투영합니다.

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
        if idx.dim() != 2:  # GPT 입력은 (batch_size, seq_len) 형태의 2차원 토큰 ID 텐서여야 합니다.
            raise ValueError("idx must have shape (batch_size, seq_len)")  # 잘못된 입력 shape가 들어오면 embedding에서 애매하게 실패하기 전에 명확히 알립니다.
        if idx.size(1) > self.config["context_length"]:  # 위치 임베딩은 context_length까지만 있으므로 더 긴 입력은 처리할 수 없습니다.
            raise ValueError("idx sequence length exceeds context_length")  # 호출자가 generate처럼 오른쪽 context만 잘라 넣어야 함을 알려 줍니다.
        x = self.embedding(idx)  # 정수 토큰 ID를 Transformer가 처리할 수 있는 (B, T, emb_dim) 연속 벡터로 바꿉니다.
        for block in self.blocks:  # 등록된 TransformerBlock을 입력 순서대로 하나씩 통과시킵니다.
            x = block(x, causal_mask=True)  # 언어 모델이 미래 토큰을 보지 못하도록 causal mask를 켠 채 block을 실행합니다.
        x = self.final_norm(x)  # 여러 block을 거친 최종 hidden state를 vocab projection 전에 정규화합니다.
        logits = self.lm_head(x)  # 각 배치·위치마다 vocab_size개 다음 토큰 점수를 계산합니다.
        if targets is None:  # 정답 토큰이 없으면 추론 상황이므로 loss 없이 logits만 반환합니다.
            return logits  # 반환 shape는 (batch_size, seq_len, vocab_size)입니다.
        loss = nn.functional.cross_entropy(  # 다음 토큰 분류 문제로 보고 모든 위치의 cross entropy를 평균냅니다.
            logits.reshape(-1, logits.size(-1)),  # (B, T, vocab_size)를 (B*T, vocab_size)로 펼쳐 각 위치를 독립 샘플처럼 만듭니다.
            targets.reshape(-1),  # (B, T) 정답 토큰 ID를 (B*T)로 펼쳐 logits의 각 행과 맞춥니다.
        )
        return loss, logits  # 학습 루프에서 바로 쓸 수 있도록 scalar loss와 원본 logits를 함께 반환합니다.


def generate_text_simple(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
) -> torch.Tensor:
    """TODO: greedy 방식으로 max_new_tokens만큼 다음 토큰을 이어 붙입니다."""
    was_training = model.training  # 생성 중 dropout을 끄되, 함수가 끝난 뒤 원래 train/eval 상태로 되돌리기 위해 현재 상태를 저장합니다.
    model.eval()  # 탐욕적 생성에서는 같은 입력이 같은 출력을 내도록 dropout 같은 훈련 전용 동작을 비활성화합니다.
    with torch.no_grad():  # 텍스트 생성은 역전파가 필요 없으므로 기울기 기록을 끄고 메모리 사용을 줄입니다.
        for _ in range(max_new_tokens):  # 요청한 새 토큰 수만큼 한 번에 하나씩 반복해서 생성합니다.
            idx_cond = idx[:, -context_size:]  # 위치 임베딩 한계를 넘지 않도록 현재 문맥의 마지막 context_size개 토큰만 모델에 넣습니다.
            logits = model(idx_cond)  # 현재 문맥을 바탕으로 각 위치의 다음 토큰 점수(logits)를 계산합니다.
            if isinstance(logits, tuple):  # 혹시 모델이 (loss, logits) 형태를 반환하는 경우에도 생성 함수가 logits만 쓰도록 방어합니다.
                logits = logits[1]  # 튜플의 두 번째 원소가 실제 다음 토큰 점수 텐서입니다.
            next_token_logits = logits[:, -1, :]  # 마지막 위치의 logits만 다음에 이어 붙일 토큰을 고르는 데 사용합니다.
            idx_next = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # 가장 점수가 큰 토큰 ID를 고르는 greedy decoding을 수행합니다.
            idx = torch.cat((idx, idx_next), dim=1)  # 새로 고른 토큰을 기존 토큰 시퀀스 오른쪽 끝에 이어 붙입니다.
    if was_training:  # 함수 호출 전 모델이 train 모드였다면 생성이 끝난 뒤 그 상태를 복구합니다.
        model.train()  # 사용자가 이어서 훈련할 때 dropout 등이 원래처럼 동작하도록 train 모드로 되돌립니다.
    return idx  # 시작 토큰 뒤에 max_new_tokens개가 추가된 전체 토큰 ID 시퀀스를 반환합니다.
