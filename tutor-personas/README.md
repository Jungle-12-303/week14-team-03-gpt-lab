# Mini GPT 튜터 페르소나 7종

이 폴더는 mini GPT 과제를 단계별로 질문하기 위한 채팅 페르소나 프롬프트 모음입니다.

## 사용 방법

새 채팅을 열고, 원하는 단계의 파일 내용을 첫 메시지 또는 시스템 지시문으로 붙여 넣으면 됩니다.

## 목록

1. `01-bpe-tokenizer-tutor.md`: tokenizer 자료구조와 UTF-8 byte-level BPE
2. `02-dataset-inputembedding-tutor.md`: 데이터 window, input-target shift, embedding
3. `03-multiheadattention-tutor.md`: Q/K/V, causal mask, attention shape
4. `04-gptmodel-components-tutor.md`: GPTModel 조립, residual, LayerNorm, FFN
5. `05-pretraining-train-tutor.md`: next token prediction, loss, optimizer
6. `06-finetune-classification-tutor.md`: 감성 분류 fine-tuning, classification head
7. `07-integration-debugging-tutor.md`: 전체 테스트와 통합 디버깅

## 역할 구분

- 1번은 tokenizer 자료구조
- 2번은 데이터와 embedding 입력
- 3번은 attention 계산
- 4번은 GPT 조립
- 5번은 모델 학습
- 6번은 분류용 fine-tuning
- 7번은 전체 디버깅
