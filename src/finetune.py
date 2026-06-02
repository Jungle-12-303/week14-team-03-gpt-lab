# -*- coding: utf-8 -*-
"""NSMC 감성 분류 미세 조정 과제 템플릿."""

from pathlib import Path
import csv
import json
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset

try:
    from .model import GPTModel
except ImportError:
    from model import GPTModel


def append_sentiment_result(path: str | Path, result: dict) -> None:
    """감성 분류 train/eval 결과를 JSON history 파일에 누적 저장합니다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            history = json.load(f)
    else:
        history = {
            "task": "sentiment_classification",
            "history": [],
        }

    # train_epoch_sentiment와 evaluate_sentiment를 번갈아 호출해도 같은 파일에 순서대로 쌓입니다.
    # 예: {"split": "train", ...}, {"split": "val", ...}, {"split": "test", ...}
    history["history"].append(result)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def make_sentiment_dataset(
    train_tsv_path: str | Path,
    test_tsv_path: str | Path | None = None,
    val_ratio: float = 0.08,
    seed: int = 42,
    output_dir: str | Path | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    TODO: NSMC TSV를 읽어 train/validation/test 감성 분류 데이터를 만듭니다.

    반환 형식:
        [{"text": "리뷰", "label": 0 또는 1}, ...]
    """
    def write_jsonl(path: str | Path, rows: list[dict]) -> None:
        """감성 분류 제출 형식인 JSONL로 한 줄에 샘플 하나씩 저장합니다."""
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                # json.dump로 전체 배열을 저장하면 요구서의 .jsonl 형식과 달라집니다.
                # JSONL은 큰 데이터도 줄 단위로 읽기 쉬우므로 train/val/test 파일에 적합합니다.
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def read_nsmc_tsv(path: str | Path) -> list[dict]:
        rows = []

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")

            for row in reader:
                text = row.get("document")
                label = row.get("label")

                if text is None or label is None:
                    continue

                text = text.strip()

                if text == "":
                    continue

                rows.append({
                    "text": text,
                    "label": int(label),
                })

        return rows

    train_data = read_nsmc_tsv(train_tsv_path)

    rng = random.Random(seed)
    rng.shuffle(train_data)

    val_size = int(len(train_data) * val_ratio)
    val_data = train_data[:val_size]
    train_data = train_data[val_size:]

    if test_tsv_path is None:
        test_data = []
    else:
        test_data = read_nsmc_tsv(test_tsv_path)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for name, data in [
            ("nsmc_sentiment_train.jsonl", train_data),
            ("nsmc_sentiment_val.jsonl", val_data),
            ("nsmc_sentiment_test.jsonl", test_data),
        ]:
            write_jsonl(output_dir / name, data)

    return train_data, val_data, test_data
    # raise NotImplementedError("make_sentiment_dataset을 구현하세요.")


class ReviewSentimentDataset(Dataset):
    """감성 분류용 Dataset. 리뷰 하나와 label 하나를 반환합니다."""

    def __init__(
        self,
        data: list[dict],
        tokenizer,
        max_length: int = 128,
        pad_id: int | None = None,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.get_pad_id() if pad_id is None else pad_id

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """TODO: text를 encode하고 max_length까지 자르거나 padding한 뒤 label과 함께 반환합니다."""
        item = self.data[idx]

        token_ids = self.tokenizer.encode(item["text"])

        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]

        pad_length = self.max_length - len(token_ids)

        if pad_length > 0:
            token_ids = token_ids + [self.pad_id] * pad_length

        input_ids = torch.tensor(token_ids, dtype=torch.long)
        label = int(item["label"])

        return input_ids, label
        # raise NotImplementedError("ReviewSentimentDataset.__getitem__을 구현하세요.")


class GPTForSequenceClassification(nn.Module):
    """
    GPT backbone 위에 감성 분류용 Linear head를 붙인 모델.

    주의: LM head는 다음 토큰 예측용입니다. 감성 분류는 hidden state 위에 별도 classifier를 붙입니다.
    """

    def __init__(
        self,
        gpt_model: GPTModel,
        num_labels: int = 2,
        drop_rate: float = 0.1,
    ):
        super().__init__()
        self.gpt = gpt_model
        self.num_labels = num_labels
        # TODO: dropout과 classifier를 정의하세요. classifier 입력 차원은 gpt_model.config["emb_dim"]입니다.
        emb_dim = gpt_model.config["emb_dim"]

        self.dropout = nn.Dropout(drop_rate)
        self.classifier = nn.Linear(emb_dim, num_labels)

        # raise NotImplementedError("GPTForSequenceClassification.__init__을 구현하세요.")

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: GPT hidden state에서 문장 대표 벡터를 뽑아 분류 logits를 만듭니다.

        labels가 있으면 (loss, logits), 없으면 logits를 반환합니다.
        """
        x = self.gpt.embedding(input_ids)
        x = self.gpt.blocks(x)
        x = self.gpt.final_norm(x)

        pad_id = 0
        valid_mask = input_ids != pad_id
        last_token_idx = valid_mask.sum(dim=1) - 1
        last_token_idx = last_token_idx.clamp(min=0)

        batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)
        sentence_repr = x[batch_idx, last_token_idx]

        sentence_repr = self.dropout(sentence_repr)
        logits = self.classifier(sentence_repr)

        if labels is None:
            return logits

        loss = nn.functional.cross_entropy(logits, labels)
        return loss, logits
        # raise NotImplementedError("GPTForSequenceClassification.forward를 구현하세요.")


def train_epoch_sentiment(
    model: GPTForSequenceClassification,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    epoch: int | None = None,
    results_path: str | Path | None = None,
) -> tuple[float, float]:
    """TODO: 감성 분류 모델을 1 epoch 훈련하고 (평균 loss, accuracy)를 반환합니다."""
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for input_ids, labels in train_loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        loss, logits = model(input_ids, labels)

        loss.backward()
        optimizer.step()

        batch_size = input_ids.size(0)
        preds = torch.argmax(logits, dim=-1)

        total_loss += loss.item() * batch_size
        total_correct += (preds == labels).sum().item()
        total_count += batch_size

    avg_loss = total_loss / total_count if total_count > 0 else float("nan")
    accuracy = total_correct / total_count if total_count > 0 else float("nan")

    if results_path is not None:
        append_sentiment_result(
            results_path,
            {
                "split": "train",
                "epoch": epoch,
                "loss": avg_loss,
                "accuracy": accuracy,
                "num_examples": total_count,
            },
        )

    return avg_loss, accuracy
    # raise NotImplementedError("train_epoch_sentiment를 구현하세요.")


def evaluate_sentiment(
    model: GPTForSequenceClassification,
    data_loader,
    device: torch.device,
    *,
    split: str = "val",
    epoch: int | None = None,
    results_path: str | Path | None = None,
) -> tuple[float, float]:
    """TODO: 감성 분류 모델을 평가하고 (평균 loss, accuracy)를 반환합니다."""
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for input_ids, labels in data_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            loss, logits = model(input_ids, labels)

            batch_size = input_ids.size(0)
            preds = torch.argmax(logits, dim=-1)

            total_loss += loss.item() * batch_size
            total_correct += (preds == labels).sum().item()
            total_count += batch_size

    avg_loss = total_loss / total_count if total_count > 0 else float("nan")
    accuracy = total_correct / total_count if total_count > 0 else float("nan")

    if results_path is not None:
        append_sentiment_result(
            results_path,
            {
                "split": split,
                "epoch": epoch,
                "loss": avg_loss,
                "accuracy": accuracy,
                "num_examples": total_count,
            },
        )

    return avg_loss, accuracy
    # raise NotImplementedError("evaluate_sentiment를 구현하세요.")
