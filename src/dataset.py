# -*- coding: utf-8 -*-
"""GPT 사전 학습용 Dataset/DataLoader 과제 템플릿."""

import torch
from torch.utils.data import DataLoader, Dataset

# GPTDataset: 샘플 하나 만드는 법 정의
class GPTDataset(Dataset):
    """
    token ID 리스트를 다음 토큰 예측용 input/target 쌍으로 자릅니다.

    예: token_ids=[10, 11, 12, 13], context_length=3
    - input:  [10, 11, 12]
    - target: [11, 12, 13]
    """

    def __init__(
        self,
        token_ids: list[int],
        context_length: int,
        stride: int | None = None, # stride(보폭): 슬라이딩 윈도우가 한 번에 몇 칸씩 이동할지 결정
    ):
        self.token_ids = token_ids
        self.context_length = context_length
        self.stride = stride if stride is not None else context_length

        # DONE: 만들 수 있는 학습 샘플 개수를 self._length에 저장하세요.
        max_start = len(token_ids) - self.context_length - 1
        self._length = max(0, max_start//self.stride + 1)

    def __len__(self) -> int:
        """DONE: 전체 샘플 개수를 반환합니다."""
        return self._length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        DONE: idx번째 input_ids와 target_ids를 LongTensor로 반환합니다.

        Returns:
            input_ids: (context_length,)
            target_ids: (context_length,)
        """
        start = idx * self.stride
        input_ids = self.token_ids[start : start + self.context_length]
        target_ids = self.token_ids[start + 1 : start + 1 + self.context_length]

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long)
        )


# DataLoader: 샘플 여러 개를 batch_size만큼 묶어서 학습 루프에 공급
def create_dataloader(
    token_ids: list[int],
    context_length: int,
    batch_size: int = 8,
    stride: int | None = None,
    drop_last: bool = False,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """DONE: GPTDataset을 만들고 torch.utils.data.DataLoader로 감싸 반환합니다."""
    dataset = GPTDataset(
        token_ids = token_ids,
        context_length = context_length,
        stride = stride
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
