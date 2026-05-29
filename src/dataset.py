# -*- coding: utf-8 -*-
"""GPT 사전 학습용 Dataset/DataLoader 과제 템플릿."""

import torch
from torch.utils.data import DataLoader, Dataset


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
        stride: int | None = None,
    ):
        self.token_ids = token_ids
        self.context_length = context_length
        self.stride = stride if stride is not None else context_length

        self.input_ids = []
        self.target_ids = []

        #[10, 11, 12, 13 ,14 ,15]에서    for문 0, 3, 1
        for i in range(0, len(token_ids) - context_length, self.stride):
            input_chunk = token_ids[i:i + context_length]
            target_chunk = token_ids[i + 1: i + context_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

        # TODO: 만들 수 있는 학습 샘플 개수를 self._length에 저장하세요.
        self.length = len(self.input_ids)
        # raise NotImplementedError("GPTDataset.__init__에서 self._length를 구현하세요.")

    def __len__(self) -> int:
        """TODO: 전체 샘플 개수를 반환합니다."""
        return self.length
        # raise NotImplementedError("GPTDataset.__len__을 구현하세요.")

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: idx번째 input_ids와 target_ids를 LongTensor로 반환합니다.
        
        Returns:
            input_ids: (context_length,)
            target_ids: (context_length,)
        """
        return self.input_ids[idx], self.target_ids[idx]
        # raise NotImplementedError("GPTDataset.__getitem__을 구현하세요.")


def create_dataloader(
    token_ids: list[int],
    context_length: int,
    batch_size: int = 8,
    stride: int | None = None,
    drop_last: bool = False,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """TODO: GPTDataset을 만들고 torch.utils.data.DataLoader로 감싸 반환합니다."""

    #
    if stride is None:
        stride = context_length
    
    #GPTDataset : 데이터 자르기
    dataset = GPTDataset(
        token_ids = token_ids,
        context_length = context_length,
        stride = stride
    )
    

    #dataset을 DataLoader로 감싸서 반환
    return DataLoader(
        dataset,
        batch_size = batch_size, #한 번에 몇개의 데이터를 묶을 건지에 대한 숫자
        shuffle = shuffle, # 데이터 순서 섞을지 말지
        drop_last = drop_last, # 마지막 batch 부족하면 버릴지 말지
        num_workers = num_workers,

    )

    # raise NotImplementedError("create_dataloader를 구현하세요.")
