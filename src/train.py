# -*- coding: utf-8 -*-
"""GPT 사전 학습 유틸리티 과제 템플릿."""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

try:
    from .model import GPTModel
except ImportError:
    from model import GPTModel


def save_training_results(path: str | Path, results: dict) -> None:
    """학습 중간/최종 결과를 JSON 파일로 저장합니다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # loss, 생성 샘플, 설정값은 REPORT.md에 옮겨 적어야 하므로 파일로 남겨 둡니다.
    # ensure_ascii=False를 쓰면 한국어 생성 샘플이 \uXXXX 형태로 깨져 보이지 않습니다.
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


# 모델이 얼마나 틀렸는지 숫자로 계산하는 함수
# batch 1개 loss
def calc_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: GPTModel,
    device: torch.device,
) -> torch.Tensor:
    """TODO: 한 배치를 device로 옮긴 뒤 다음 토큰 예측 cross entropy loss를 계산합니다."""

    #1. input과 target을 model이 있는 device로 이동 -> 모델이 GPU에 있으면 데이터도 GPU에 있어야 한다
    #input_batch라는 텐서를 CPU/GPU 중 현재 학습에 사용할 장치로 옮긴다

    input_batch = input_batch.to(device) #모델에게 보여주는 토큰
    target_batch = target_batch.to(device)# 모델이 맞혀야 하는 정답 토큰

    #2. GPT 모델에 input을 넣어서 다음 토큰 후보 점수(logits)를 얻음
    logits = model(input_batch) #2행 4열

    #3. 교차 엔트로피 손실 계산, logits와 정답 target_batch를 비교해서 loss를 계산한다
    loss = F.cross_entropy(
        logits.flatten(0,1), # (B, T, vocab_size) -> (B*T, vocab_size) 행,열을 합쳐서 1행으로 만든다
        target_batch.flatten() #(B, T) -> (B*T)
    )

    return loss


    raise NotImplementedError("calc_loss_batch를 구현하세요.")

#DataLoader 안에 있는 batch들을 돌면서 loss를 계산하고 평균을 낸다
#여러 batch 전체 평균 loss
def calc_loss_loader(
    data_loader,
    model: GPTModel,
    device: torch.device,
    num_batches: int | None = None,
) -> float:
    """TODO: data_loader의 평균 loss를 계산합니다. 검증에서는 torch.no_grad()를 사용하세요."""

    #data_loader가 비어 있으면 평균을 낼 수 없으므로 nan 반환
    if len(data_loader) == 0:
        return float("nan")

    #num_batches가 None이면 전체 batch를 사용
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        #요청한 batch 수가 실제 batch 수보다 크면 실제 batch 수까지만 사용
        num_batches = min(num_batches, len(data_loader))

    if num_batches == 0:
        return float("nan")

    # 평가용 loss는 dropout 같은 학습 전용 랜덤 동작을 끈 상태에서 재야 합니다.
    # 다만 이 함수는 학습 루프 중간에서도 호출되므로, 호출자가 원래 train 모드였는지
    # eval 모드였는지를 기억해 두었다가 마지막에 그대로 복구합니다.
    was_training = model.training
    model.eval()

    total_loss = 0.0

    try:
        # loss 값만 확인하므로 gradient 그래프를 만들 필요가 없습니다.
        # torch.no_grad()를 사용하면 메모리를 덜 쓰고 평가 속도도 조금 빨라집니다.
        with torch.no_grad():
            for i, (input_batch, target_batch) in enumerate(data_loader):

                # num_batches개까지만 계산해 긴 validation loader도 빠르게 샘플 평가할 수 있게 합니다.
                if i >= num_batches:
                    break

                loss = calc_loss_batch(
                    input_batch=input_batch,
                    target_batch=target_batch,
                    model=model,
                    device=device,
                )

                # loss는 scalar Tensor이므로 Python float 값만 꺼내 평균 계산에 사용합니다.
                total_loss += loss.item()
    finally:
        # 평가 호출이 끝난 뒤 모델 모드를 원래 상태로 되돌립니다.
        # train_model() 안에서 validation을 잰 뒤 학습을 이어갈 때 dropout이 다시 켜져야 합니다.
        model.train(was_training)

    #batch별 loss의 평균 반환
    return total_loss / num_batches


    # raise NotImplementedError("calc_loss_loader를 구현하세요.")

#학습 중간 저장 함수
#colab이나 컴퓨터 꺼지면 처음부터 학습해야 되기 떄문에 이걸 막기 위해 중간 저장을 한다
def save_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    path: str,
) -> None:
    """TODO: model/optimizer 상태, epoch, global_step을 torch.save로 저장합니다."""
    checkpoint = {
        "model_state_dict" : model.state_dict(), #모델 가중치
        "optimizer_state_dict" : optimizer.state_dict(), #optimizer 상태
        "epoch" : epoch,
        "global_step" : global_step, # 전체 학습 step 수
    }

    torch.save(checkpoint, path)
    # raise NotImplementedError("save_checkpoint를 구현하세요.")

# 저장해둔 checkpoint 파일을 다시 불러와서 모델 상태를 복원하는 함수
def load_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer | None,
    path: str,
    device: torch.device,
) -> tuple[int, int]:
    """TODO: torch.load로 checkpoint를 읽어 model/optimizer 상태를 복원합니다."""

    #path에 있는 checkpoint 파일 읽는다
    checkpoint = torch.load(path, map_location=device)

    #저장된 모델 가중치를 현재 model에 넣는다
    model.load_state_dict(checkpoint["model_state_dict"])

    #optimizer 상태 복원
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint["epoch"]
    global_step = checkpoint["global_step"]

    return epoch, global_step

    # raise NotImplementedError("load_checkpoint를 구현하세요.")

# 다음 토큰을 하나씩 생성해서 문장을 이어 붙이는 함수
def generate(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_id: int | None = None,
) -> torch.Tensor:
    """TODO: temperature와 top-k 샘플링을 지원하는 생성 함수를 구현합니다."""
    model.eval() ##평가 모드 : dropout 같은 랜덤 동작 끈다

    with torch.no_grad(): #gradient 계산 끄기
        for _ in range(max_new_tokens):
            #1. 모델이 볼 수 있는 최대 길이만큼만 자른다
            idx_cond = idx[:, -context_size:] #

            #2. 모델에 넣어서 logits를 얻는다
            logits = model(idx_cond)

            #3. 마지막 위치의 logits만 사용한다.
            logits = logits[:, -1, :]

            #4. top-k가 있으면 상위 k개 후보만 남긴다
            if top_k is not None:
                top_k = min(top_k, logits.size(-1))

                top_logits, _ = torch.topk(logits, top_k)

                min_val = top_logits[:, -1].unsqueeze(-1)

                logits = torch.where(
                    logits < min_val,
                    torch.tensor(float("-inf"), device=logits.device),
                    logits,
                )

            #5. temperature 적용 : 랜덤성 조절값
            if temperature <= 0.0:
                # temperature가 0 이하이면 greedy 방식으로 처리

                #가장 높은 점수만 고른다
                next_id = torch.argmax(logits, dim= -1, keepdim = True)
            else:
                logits = logits / temperature

                probs = F.softmax(logits, dim = -1) #점수를 확률로 바꾼다

                next_id = torch.multinomial(probs, num_samples = 1) #확률을 보고 하나를 뽑는다

            #6. eos 토큰이 나오면 종료
            if eos_id is not None and next_id.item() == eos_id: #
                idx = torch.cat((idx, next_id), dim = 1)
                break

            #7. 새 토큰을 기존 idx 뒤에 붙인다
            idx = torch.cat((idx, next_id), dim=1)

    return idx

    # raise NotImplementedError("generate를 구현하세요.")


def generate_and_print_sample(
    model: GPTModel,
    tokenizer,
    device: torch.device,
    start_context: str,
    max_new_tokens: int = 50,
    context_size: int = 256,
    temperature: float = 0.8,
    top_k: int | None = 40,
) -> str:
    """TODO: start_context를 encode하고 generate 후 decode하여 출력합니다."""
    model.eval()

    encoded = tokenizer.encode(start_context)
    idx = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)

    eos_id = tokenizer.get_eos_id() if hasattr(tokenizer, "get_eos_id") else None

    generated_ids = generate(
        model=model,
        idx=idx,
        max_new_tokens=max_new_tokens,
        context_size=context_size,
        temperature=temperature,
        top_k=top_k,
        eos_id=eos_id,
    )

    generated_text = tokenizer.decode(generated_ids[0].tolist())
    print(generated_text)
    return generated_text
    # raise NotImplementedError("generate_and_print_sample을 구현하세요.")


def train_model(
    model: GPTModel,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
    start_context: str,
    tokenizer,
    ckpt_freq: int | None = None,
    start_epoch: int = 0,
    global_step: int = 0,
    results_path: str | Path | None = None,
) -> list[float]:
    """TODO: 사전 학습 루프를 구현하고 epoch별 train loss 리스트를 반환합니다."""
    model.to(device)

    train_losses = []
    history = {
        "task": "pretraining",
        "model_config": dict(model.config),
        "num_epochs": num_epochs,
        "eval_freq": eval_freq,
        "eval_iter": eval_iter,
        "start_context": start_context,
        "train_losses": train_losses,
        "epoch_history": [],
        "eval_history": [],
        "last_global_step": global_step,
    }

    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()

        epoch_loss = 0.0
        batch_count = 0

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()

            loss = calc_loss_batch(
                input_batch=input_batch,
                target_batch=target_batch,
                model=model,
                device=device,
            )

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1
            global_step += 1

            if eval_freq > 0 and global_step % eval_freq == 0:
                model.eval()

                train_loss = calc_loss_loader(
                    train_loader,
                    model,
                    device,
                    num_batches=eval_iter,
                )

                val_loss = calc_loss_loader(
                    val_loader,
                    model,
                    device,
                    num_batches=eval_iter,
                )

                print(
                    f"Epoch {epoch + 1}, step {global_step}: "
                    f"train loss {train_loss:.4f}, val loss {val_loss:.4f}"
                )

                # 생성 샘플도 REPORT.md에 필요하므로 print만 하지 않고 history에도 저장합니다.
                # context_size는 모델의 position embedding 길이를 넘으면 안 되므로 config 값을 사용합니다.
                generated_text = generate_and_print_sample(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    start_context=start_context,
                    context_size=model.config.get("context_length", 256),
                )

                history["eval_history"].append({
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "generated_sample": generated_text,
                })
                history["last_global_step"] = global_step

                # Colab 런타임이 끊겨도 마지막 평가 결과를 잃지 않도록 매 평가마다 저장합니다.
                if results_path is not None:
                    save_training_results(results_path, history)

                model.train()

            if ckpt_freq is not None and ckpt_freq > 0 and global_step % ckpt_freq == 0:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    global_step=global_step,
                    path=f"checkpoint_step_{global_step}.pt",
                )

        avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else float("nan")
        train_losses.append(avg_epoch_loss)
        history["epoch_history"].append({
            "epoch": epoch + 1,
            "train_loss": avg_epoch_loss,
            "global_step": global_step,
        })
        history["last_global_step"] = global_step

        # eval_freq가 커서 중간 평가가 한 번도 없어도 epoch 평균 loss는 저장합니다.
        if results_path is not None:
            save_training_results(results_path, history)

    return train_losses
    # raise NotImplementedError("train_model을 구현하세요.")


def plot_losses(train_losses: list[float], val_losses: list[float] | None = None) -> None:
    """훈련/검증 손실 그래프를 그리는 제공 함수."""
    plt.plot(train_losses, label="Train")
    if val_losses is not None:
        plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training / Validation Loss")
    plt.show()
