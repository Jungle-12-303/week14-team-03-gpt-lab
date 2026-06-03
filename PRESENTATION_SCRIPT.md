# mini GPT 구현 과제 발표 대본

> `[화면]`으로 시작하는 줄은 발표자가 읽는 문장이 아니라, 슬라이드나 보고서 그림을 넘길 때 참고하는 메모입니다.

## 0. 오프닝

안녕하세요. 3조 발표를 맡은 주호석입니다.

저희 프로젝트의 목표는 PyTorch만 사용해서 mini GPT를 직접 구현하고, NSMC 영화 리뷰 데이터로 사전 학습과 감성 분류 미세 조정을 수행하는 것이었습니다.

이 프로젝트에서 저희가 가장 중요하게 본 질문은 단순히 "loss를 낮출 수 있는가"가 아니었습니다. 저희의 최종 목표는 영화 리뷰가 긍정인지 부정인지 분류하는 downstream task의 정확도를 높이는 것이었기 때문에, 사전 학습 loss와 감성 분류 validation accuracy가 함께 안정적으로 나오는 설정을 찾는 것을 목표로 잡았습니다.

발표는 다음 순서로 진행하겠습니다.

먼저 전체 구현 구조와 데이터 처리 방식을 설명하고, 그다음 Light 실험에서 model size, learning rate, dropout을 비교한 과정을 설명하겠습니다. 이후 Light에서 고른 설정을 Basic 설정으로 확장했을 때 어떤 성능 변화가 있었는지 보고, 마지막으로 실험 중 발생한 decode 오류와 후속 실험인 `emb_dim=192`와 `emb_dim=256` 비교까지 정리하겠습니다.

---

## 1. 프로젝트 전체 구조

먼저 프로젝트 전체 구조입니다.

저희 구현은 크게 여섯 부분으로 나눌 수 있습니다.

첫 번째는 `src/bpe.py`의 UTF-8 byte-level BPE tokenizer입니다. 원본 리뷰 문장을 바로 모델에 넣을 수 없기 때문에, 문장을 token ID 배열로 바꾸는 역할을 합니다.

두 번째는 `src/dataset.py`와 `src/embeddings.py`입니다. `GPTDataset`은 token ID를 다음 토큰 예측용 input과 target으로 잘라 주고, `InputEmbedding`은 token ID를 token embedding과 position embedding의 합으로 바꿔 Transformer에 넣을 수 있는 벡터로 만듭니다.

세 번째는 `src/attention.py`입니다. 여기서는 Multi-Head Self-Attention과 causal mask를 구현했습니다. GPT는 현재 위치에서 미래 토큰을 보면 안 되기 때문에, attention score에서 미래 위치를 `-inf`로 가려 주는 causal mask가 필요합니다.

네 번째는 `src/model.py`입니다. 이 파일에는 LayerNorm, GELU, FeedForward, TransformerBlock, GPTModel이 들어 있습니다. 전체 모델 흐름은 `InputEmbedding -> N개의 TransformerBlock -> final LayerNorm -> LM head`입니다.

다섯 번째는 `src/train.py`입니다. 여기서는 사전 학습 loss 계산, 평가, checkpoint 저장, text generation, 학습 결과 JSON 저장을 담당합니다.

마지막은 `src/finetune.py`입니다. 사전 학습된 GPT backbone 위에 감성 분류용 classifier head를 붙여서 NSMC 리뷰를 부정 0, 긍정 1로 분류하도록 미세 조정합니다.

이렇게 파일을 나눈 이유는 각 단계의 책임이 다르기 때문입니다. tokenizer는 문장을 token으로 바꾸는 문제를 담당하고, dataset은 학습 샘플을 만드는 문제를 담당하고, model은 신경망 계산을 담당하고, train과 finetune은 학습 절차를 담당합니다. 이렇게 나누면 어느 단계에서 문제가 생겼는지 추적하기 쉽고, tokenizer나 모델 구조를 바꾸더라도 학습 루프 전체를 다시 설계하지 않아도 됩니다.

전체 테스트는 `.venv` 환경에서 `pytest`로 실행했고, 결과는 `34 passed, 1 warning`이었습니다.

---

## 2. 데이터와 BPE

다음은 데이터와 BPE입니다.

저희는 NSMC 영화 리뷰 데이터를 사용했습니다. 사전 학습에는 리뷰 텍스트 자체를 사용했고, 감성 분류에는 리뷰 텍스트와 긍정/부정 label을 사용했습니다.

전처리에서는 빈 리뷰를 제거하고 공백을 정리한 뒤 train, validation, test 데이터를 나누었습니다. 보고서 기준으로 감성 분류 train 데이터는 137,996개, validation 데이터는 11,999개, test 데이터는 49,997개입니다.

Tokenizer는 UTF-8 byte-level BPE로 구현했습니다.

이 방식을 선택한 이유는 한국어 데이터의 특성 때문입니다. 한국어는 한 글자가 UTF-8에서 여러 byte로 표현됩니다. 만약 글자 단위나 공백 단위로만 자르면, 처음 보는 조사나 어미, 오타, 특수문자가 나왔을 때 `<unk>`가 많이 생길 수 있습니다. 반면 byte-level 방식은 모든 문자를 최소한 byte 단위로 표현할 수 있기 때문에, 처음 보는 한국어 문자열도 완전히 표현할 수 있습니다.

BPE는 여기서 한 단계 더 나아갑니다. 처음에는 byte 단위로 시작하지만, corpus에서 자주 붙어 나오는 byte pair를 merge해서 더 긴 token으로 만듭니다. 즉 모든 문자를 표현할 수 있는 안정성과, 자주 등장하는 패턴을 더 짧은 token sequence로 압축하는 효율성을 같이 가져가기 위해 byte-level BPE를 사용했습니다.

특수 토큰 ID는 고정했습니다. `<pad>`는 0, `<unk>`는 1, `<bos>`는 2, `<eos>`는 3이고, 원본 byte token은 4부터 259까지 사용합니다. 260 이상은 BPE merge로 새로 생긴 token입니다.

실험 규모는 Light와 Basic으로 나누었습니다.

Light 설정에서는 `corpus[:500000]`, `vocab_size=2000`, `context_length=64`를 사용했습니다. 이 설정의 목적은 최종 성능을 확정하는 것이 아니라, 여러 하이퍼파라미터 후보를 빠르게 비교하는 것이었습니다.

Basic 설정에서는 `corpus[:1500000]`, `vocab_size=3000`, `context_length=128`을 사용했습니다. 이 설정은 Light에서 고른 조합이 더 큰 데이터와 더 긴 문맥에서도 확장 가능한지 확인하기 위한 설정입니다.

---

## 3. 모델 구조

[화면] `diagrams/000_gpt_pretrain_vs_finetune_flow.svg`

이 그림은 전체 파이프라인을 보여줍니다.

왼쪽에서 NSMC 리뷰 텍스트가 BPE tokenizer를 거쳐 token ID 배열로 바뀝니다. 이 token ID는 두 가지 학습에 사용됩니다.

첫 번째는 위쪽의 사전 학습입니다. 여기서는 label 없이 리뷰 문장 자체를 사용해서 다음 token을 예측합니다. `GPTDataset`이 input과 target을 만듭니다. 예를 들어 input이 `[10, 11, 12]`라면 target은 한 칸 밀린 `[11, 12, 13]`이 됩니다. GPTModel은 각 위치마다 다음 token 후보 점수인 logits를 출력하고, 실제 다음 token과 비교해서 cross entropy loss를 계산합니다.

두 번째는 아래쪽의 감성 분류 fine-tuning입니다. 여기서는 리뷰 텍스트와 긍정/부정 label을 사용합니다. 사전 학습된 GPT backbone으로 문장 표현을 만들고, 그 위에 classification head를 붙여 부정과 긍정 두 클래스 점수를 예측합니다.

핵심은 GPT backbone은 공유하지만, 사전 학습에서는 LM head를 쓰고 감성 분류에서는 classification head를 쓴다는 점입니다. 즉 먼저 리뷰 문장을 읽고 다음 token을 예측하는 능력을 학습시킨 뒤, 그 표현을 이용해서 긍정/부정 분류를 수행하는 구조입니다.

모델 내부에서는 token embedding과 position embedding을 더합니다. token embedding은 "이 token이 무엇인가"를 표현하고, position embedding은 "이 token이 문장 안에서 몇 번째 위치인가"를 표현합니다. Transformer는 순서 정보를 자동으로 알 수 없기 때문에 position embedding이 필요합니다.

그다음 여러 개의 TransformerBlock을 통과합니다. 각 block은 causal self-attention과 feed-forward network로 구성됩니다. causal self-attention은 현재 token이 이전 token들을 얼마나 참고할지 계산합니다. 이때 미래 token을 보면 다음 token 예측 문제가 너무 쉬워지기 때문에 causal mask를 사용해서 미래 위치를 가립니다.

또한 각 block에는 LayerNorm과 residual connection을 넣었습니다. 모델이 여러 층을 통과하면 값의 크기가 불안정해질 수 있고 gradient 흐름도 약해질 수 있습니다. LayerNorm은 각 token vector의 분포를 안정화하고, residual connection은 입력 정보를 다음 층으로 직접 전달해서 깊은 모델에서도 학습이 끊기지 않도록 돕습니다.

---

## 4. 실험 설계와 선택 기준

저희는 실험을 한 번에 모든 값을 바꾸는 방식으로 하지 않았습니다. 가능한 한 하나의 축만 바꿔서 비교했습니다. 그래야 어떤 변화가 성능에 영향을 주었는지 설명할 수 있기 때문입니다.

최종 선택 기준은 세 가지로 고정했습니다.

첫 번째, sentiment validation accuracy가 높은 조합을 우선합니다. 저희 최종 목표가 감성 분류이기 때문입니다.

두 번째, validation accuracy가 비슷하면 sentiment validation loss가 낮은 조합을 선택합니다. accuracy가 같아도 loss가 낮으면 모델이 정답에 더 확신 있게 가까워졌다고 볼 수 있기 때문입니다.

세 번째, 둘 다 비슷하면 더 작은 모델을 선택합니다. 작은 모델은 학습 시간이 짧고 반복 실험이 쉬우며, 같은 성능이면 계산 비용이 더 낮기 때문입니다.

Light 실험의 공통 설정은 `corpus_chars=500000`, `vocab_size=2000`, `context_length=64`, `batch_size=8`, `n_heads=4`, `num_epochs=3`, `learning rate=3e-4`를 기준으로 했습니다. 이후 model size, learning rate, dropout을 각각 비교했습니다.

---

## 5. Model Size 가설

첫 번째 실험 축은 model size입니다.

저희가 세운 가설은 "모델이 너무 작으면 한국어 리뷰의 표현과 문맥을 충분히 담지 못하지만, 모델이 너무 크면 Light 데이터와 짧은 epoch 안에서 충분히 수렴하지 못할 수 있다"였습니다.

여기서 `emb_dim`은 token 하나를 표현하는 벡터의 크기입니다. 값이 작으면 표현 공간이 좁아서 복잡한 문맥을 담기 어렵습니다. `n_layers`는 TransformerBlock을 몇 번 쌓을지 정하는 값입니다. layer가 너무 적으면 문맥을 여러 단계로 조합하기 어렵습니다.

하지만 모델을 키우는 것이 항상 좋은 것은 아닙니다. 모델이 커지면 파라미터 수가 늘어나고, 같은 데이터와 같은 epoch에서는 충분히 학습되지 못할 수 있습니다. 그래서 저희는 Small, Base, Large를 비교했고, 최종적으로 중간 크기인 Model-Base가 가장 안정적일 것이라고 예상했습니다.

실험 결과도 이 방향과 맞았습니다. Model-Small은 구조가 작아서 리뷰 어휘는 어느 정도 나오지만 문장 연결이 불안정했고, sentiment validation accuracy도 0.634였습니다. Model-Large는 더 큰 모델이지만 짧은 학습 조건에서는 충분히 수렴하지 못해 validation accuracy가 낮게 나왔습니다. 반면 Model-Base는 validation accuracy 0.642, test accuracy 0.645로 Light 후보 중 가장 안정적인 결과를 보였습니다.

이 결과의 핵심은 "모델을 크게 만들면 무조건 좋아진다"가 아니라는 점입니다. 데이터 크기, epoch 수, 학습 시간까지 같이 고려했을 때, 제한된 조건에서는 적당한 크기의 모델이 더 좋은 선택이 될 수 있습니다.

---

## 6. Learning Rate 가설

두 번째 실험 축은 learning rate입니다.

저희 가설은 "learning rate가 낮으면 안정적이지만 loss가 천천히 줄고, 높으면 빠르게 움직이지만 validation 성능이 불안정해질 수 있다"였습니다. 그래서 `1e-4`, `3e-4`, `5e-4`를 비교했고, 중간값인 `3e-4`가 가장 안정적일 것이라고 보았습니다.

`1e-4`는 update 폭이 작습니다. 안정적일 수는 있지만, Light처럼 epoch가 제한된 실험에서는 충분히 학습하지 못할 수 있습니다. 실제로 LR-Low는 LM train loss와 validation loss가 높았고, sentiment validation accuracy도 0.609로 낮았습니다.

반대로 `5e-4`는 update 폭이 큽니다. 빠른 수렴을 기대할 수 있지만, 작은 mini GPT에서는 parameter가 너무 크게 움직여 validation 성능이 흔들릴 수 있습니다. 실제 LR-High도 validation accuracy가 0.594로 낮게 나왔습니다.

결국 이 실험에서 learning rate의 목적은 train loss만 빠르게 낮추는 것이 아니라, 제한된 학습 시간 안에서 validation 성능을 안정적으로 만드는 것이었습니다. 그 기준에서는 `3e-4`가 가장 적절했습니다.

---

## 7. Dropout 가설

세 번째 실험 축은 dropout입니다.

저희 가설은 "dropout이 없으면 학습 데이터에 너무 맞춰질 수 있고, dropout이 너무 크면 작은 모델과 작은 데이터에서 학습 신호가 약해질 수 있다"였습니다. 그래서 `0.0`, `0.1`, `0.2`를 비교했습니다.

Dropout은 일부 hidden unit을 무작위로 끄면서 특정 feature에 과하게 의존하는 것을 줄여 줍니다. 그래서 과적합을 줄이는 데 도움이 될 수 있습니다.

하지만 dropout이 너무 크면 문제가 생깁니다. 특히 이번처럼 mini GPT이고 Light 데이터로 빠르게 비교하는 상황에서는, 필요한 정보까지 자주 사라져서 underfitting처럼 보일 수 있습니다.

결과를 보면 `dropout=0.2`는 sentiment validation accuracy가 0.533까지 떨어졌습니다. 반면 `dropout=0.1`을 사용한 Model-Base가 가장 안정적이었습니다.

다만 이 dropout 실험은 해석할 때 주의해야 합니다. 보고서에도 적었듯이 dropout 실험의 JSON은 `emb_dim=128`, `n_layers=4`로 저장되어 있어서 Model-Base와 완전히 같은 통제 조건은 아니었습니다. 따라서 이 결과를 "dropout 0.2가 절대 나쁘다"라고 일반화하면 안 됩니다. 더 정확한 해석은 "이번 Light 실험 조건에서는 `drop_rate=0.1`을 유지하는 것이 가장 안전했다"입니다.

---

## 8. Light 실험 결과

[화면] `diagrams/report_graphs/01_light_sentiment_val_accuracy.png`, `diagrams/report_graphs/02_light_loss_comparison.png`

이제 Light 실험 전체 결과입니다.

그래프에서 Model-Base가 Light 후보 중 가장 높은 sentiment validation accuracy를 보였습니다. Model-Base의 설정은 `emb_dim=192`, `n_layers=4`, `n_heads=4`, `drop_rate=0.1`, `lr=3e-4`입니다.

Model-Base를 선택한 이유는 단순히 사전 학습 loss 하나만 낮았기 때문이 아닙니다. 저희 기준은 downstream task인 감성 분류 validation accuracy였고, Model-Base는 sentiment validation accuracy 0.642, validation loss 0.6251로 가장 안정적이었습니다. 또한 Model-Large보다 작기 때문에 같은 Colab 학습 환경에서 반복 실험하기도 더 좋았습니다.

여기서 중요한 점은 Light 단계의 목적입니다. Light 단계는 최종 성능을 확정하는 단계가 아니라, 큰 Basic 실험으로 가져갈 후보를 합리적으로 좁히는 단계였습니다. 전체 데이터와 긴 epoch로 모든 조합을 다 실험하기에는 시간이 많이 들기 때문에, Light에서 빠르게 후보를 비교하고 가장 안정적인 조합을 선택했습니다.

손실 그래프도 함께 봐야 합니다. 사전 학습의 다음 token 예측 loss가 낮아지는 것과 감성 분류 accuracy가 좋아지는 것은 완전히 같은 문제가 아닙니다. 사전 학습은 다음 token을 맞히는 문제이고, 감성 분류는 문장 전체의 긍정/부정을 맞히는 문제입니다. 그래서 저희는 LM loss뿐 아니라 sentiment validation loss와 validation accuracy를 함께 보았습니다.

---

## 9. Basic 설정 적용 결과

[화면] `diagrams/report_graphs/03_basic_sentiment_accuracy.png`

다음은 Light에서 고른 Model-Base 조합을 Basic 설정에 적용한 결과입니다.

Basic에서는 corpus를 더 크게 사용하고, vocabulary size를 2,000에서 3,000으로 늘리고, context length를 64에서 128로 늘렸습니다. 또한 사전 학습 epoch를 3에서 10으로 늘렸습니다.

이렇게 확장한 이유는 Light 실험만으로는 모델이 충분히 수렴했다고 보기 어렵기 때문입니다. Light는 빠른 비교를 위한 설정이었고, 실제로 더 큰 데이터와 더 긴 학습을 적용했을 때 downstream 성능이 좋아지는지 확인할 필요가 있었습니다.

Basic 사전 학습 결과는 마지막 epoch 평균 train loss 5.1752, 마지막 eval train loss 4.8838, 마지막 eval validation loss 5.4538이었습니다. 이후 sentiment fine-tuning 결과는 validation accuracy 0.8080, test accuracy 0.8047이었습니다.

Light 결과와 비교하면 downstream 성능이 크게 올라갔습니다. Light Model-Base의 test accuracy는 0.645였고, 별도 마지막 재실행 기록에서는 0.618까지 내려간 결과도 있었습니다. 반면 Basic에서는 test accuracy가 0.8047까지 올라갔습니다.

다만 여기서 주의할 점이 있습니다. Light와 Basic은 corpus size, vocab size, context length, epoch가 함께 바뀌었습니다. 따라서 이 상승을 특정 파라미터 하나의 효과라고 단정할 수는 없습니다. 더 정확한 해석은 "Light에서 후보를 선별하고, Basic에서 그 조합의 확장 가능성을 확인했다"입니다.

또 하나 주의할 점은 Basic loss가 Light loss보다 높게 보일 수 있다는 것입니다. 이것을 성능이 나빠졌다고 바로 해석하면 안 됩니다. Basic은 vocab size가 2,000에서 3,000으로 커졌고 context length도 64에서 128로 늘었습니다. 즉 모델이 맞혀야 하는 다음 token 후보가 더 많아졌고, 더 긴 문맥을 처리해야 합니다. 그래서 loss의 절대값은 Light와 직접 비교하기 어렵습니다.

중요한 결론은, 더 큰 설정을 사용할 때는 그에 맞는 충분한 학습 epoch가 함께 필요하다는 것입니다. 이번 Basic 결과는 더 많은 데이터와 더 긴 사전 학습이 downstream sentiment accuracy 개선에 도움이 되었을 가능성을 보여줍니다.

---

## 10. 디버깅 기록: 생성 샘플 decode 오류

[화면] `diagrams/decode_error_traceback.svg`

다음은 실험 중 발생한 문제입니다.

Basic 사전 학습 중 학습 자체는 정상적으로 진행되었습니다. 예를 들어 epoch 1 step 200, step 400, step 600, epoch 2 step 800까지 진행되었습니다.

그런데 오류는 학습 loss 계산이 아니라, 생성 샘플을 문자열로 바꾸는 decode 단계에서 발생했습니다.

실행 흐름을 보면, 먼저 `model.generate()`가 generated token IDs를 만듭니다. 그다음 `tokenizer.decode(generated_ids)`가 token ID를 byte로 다시 펼치고, 마지막에 `bytes.decode("utf-8")`로 문자열 복원을 시도합니다. 그런데 학습 초반 GPT는 아직 byte-level BPE의 규칙을 잘 학습하지 못했기 때문에, UTF-8 문법상 불가능한 byte sequence를 생성할 수 있습니다. 이때 `UnicodeDecodeError`가 발생했습니다.

이 오류를 해석할 때 중요한 점은, 이것이 학습 실패가 아니라는 것입니다. vocab load 실패도 아니고, train/validation loss 계산 실패도 아니고, optimizer가 깨진 것도 아닙니다. 문제는 오직 "생성 샘플을 출력하려고 decode하는 순간" 발생했습니다.

그래서 해결 방향도 학습 루프 전체를 바꾸는 것이 아니라, sample logging 단계에서 decode를 안전하게 만드는 것이었습니다. 최종적으로 tokenizer의 `decode()`가 `errors="replace"` 옵션을 받을 수 있게 하고, 학습 중 생성 샘플 출력에서는 `tokenizer.decode(generated_ids, errors="replace")`로 호출하도록 했습니다.

이렇게 하면 학습 초반에 생성 문장 일부가 `�`로 깨져 보일 수는 있지만, 학습과 loss 저장은 계속 진행됩니다. 발표에서 이 문제는 "모델 학습 실패"가 아니라 "byte-level 생성 샘플의 UTF-8 복원 문제"로 설명하는 것이 정확합니다.

---

## 11. corpus_chars 해석 주의점

[화면] `diagrams/report_graphs/06_corpus_chars_limit.png`

또 하나 실험 해석에서 주의한 부분은 `corpus_chars`입니다.

`corpus_chars`는 학습에 사용할 문자열을 앞에서 몇 글자까지 자를 것인가를 정하는 값입니다. 하지만 원본 train corpus보다 큰 값을 넣는다고 해서 실제 데이터가 자동으로 늘어나는 것은 아닙니다.

현재 LM train text는 약 1,379,486자입니다. 따라서 `corpus[:5_000_000]`처럼 더 큰 값을 넣어도 실제로는 전체 train corpus까지만 사용됩니다. 이 경우는 "500만 자 학습"이 아니라 "가지고 있는 전체 train corpus 학습"입니다.

이 점이 중요한 이유는 실험 해석 때문입니다. 실제 데이터가 138만 자 정도인데 `1,500,000`과 `5,000,000`을 비교한다고 해도, 두 실험은 거의 같은 데이터를 보는 실험이 됩니다. 더 큰 corpus 실험을 하려면 추가 리뷰 데이터를 붙이거나 별도 말뭉치를 추가해야 합니다.

또 train, validation, test를 섞어서 글자 수를 늘리는 것은 피해야 합니다. 평가 데이터가 학습에 들어가면 leakage가 생겨서 성능 평가가 왜곡되기 때문입니다.

---

## 12. 감성 분류 미세 조정

다음은 fine-tuning 구조입니다.

감성 분류는 NSMC 리뷰를 부정 0, 긍정 1로 분류하는 task입니다. 여기서는 GPT의 LM head를 그대로 사용하지 않았습니다. LM head는 다음 token을 예측하기 위한 출력층이기 때문에, 문장 전체의 감정을 분류하는 문제에는 맞지 않습니다.

그래서 저희는 GPT backbone의 hidden state를 사용하고, 그 위에 Linear classifier를 붙였습니다.

문장 대표 벡터로는 마지막 유효 token의 hidden state를 사용했습니다. 이 선택에는 이유가 있습니다. GPT는 causal model이기 때문에 뒤쪽 token일수록 앞쪽 문맥을 더 많이 본 상태입니다. 즉 마지막 유효 token은 문장 앞부분의 정보를 가장 많이 반영한 위치입니다.

다만 padding token은 실제 문장이 아니기 때문에 대표 벡터로 쓰면 안 됩니다. 그래서 input에서 `<pad>`가 아닌 마지막 위치를 찾고, 그 위치의 hidden state를 classifier에 넣었습니다.

Basic fine-tuning의 마지막 결과는 train accuracy 0.7517, validation accuracy 0.8080, test accuracy 0.8047이었습니다. 이 결과를 통해 Light에서 고른 backbone 설정이 Basic 규모에서도 감성 분류 성능으로 이어질 수 있음을 확인했습니다.

---

## 13. 후속 실험: emb_dim 192와 256 비교

[화면] `diagrams/report_graphs/04_basic_pretrain_loss_curves.png`, `diagrams/report_graphs/05_emb_dim_generalization_gap.png`

마지막으로 후속 실험입니다.

Basic 설정에서 모델의 표현 차원인 `emb_dim`을 192에서 256으로 키운 실험도 비교했습니다.

직관적으로는 `emb_dim=256`이 더 큰 모델이기 때문에 더 좋은 성능을 낼 것처럼 보일 수 있습니다. 실제로 pretraining train loss는 `emb_dim=256`이 더 빠르게 낮췄습니다. 마지막 train loss도 4.4891로, `emb_dim=192`의 4.8838보다 낮았습니다.

하지만 validation loss는 거의 좋아지지 않았습니다. `emb_dim=192`의 validation loss는 5.4538이고, `emb_dim=256`은 5.4473으로 차이가 매우 작았습니다. 반면 train-validation gap은 `emb_dim=192`가 0.5700, `emb_dim=256`이 0.9582였습니다.

이 gap이 커졌다는 것은, 큰 모델이 train corpus에는 더 잘 맞았지만 validation corpus에는 그만큼 좋아지지 않았다는 뜻입니다. 즉 일반화 성능이 좋아졌다기보다 과적합 신호가 커졌다고 해석할 수 있습니다.

downstream sentiment 결과도 같은 방향이었습니다. `emb_dim=192`는 validation accuracy 80.80%, test accuracy 80.47%였고, `emb_dim=256`은 validation accuracy 78.22%, test accuracy 78.05%였습니다.

이 결과의 결론은 "`emb_dim=256`이 절대 나쁘다"가 아닙니다. 더 정확한 결론은 "모델 용량을 키우면 train loss는 더 내려갈 수 있지만, 그 용량을 일반화 성능으로 바꾸려면 learning rate, dropout, early stopping, checkpoint 선택도 함께 조정해야 한다"입니다.

특히 `emb_dim=256`은 파라미터 수가 2.95M에서 4.73M으로 늘었습니다. 그런데 corpus, dropout, learning rate, epoch는 그대로였기 때문에, 큰 모델에 맞는 정규화나 학습률 조정이 부족했을 수 있습니다. 다음 실험에서는 `drop_rate`를 0.15에서 0.2 정도로 높이거나, learning rate를 낮추고, 마지막 checkpoint가 아니라 validation loss가 가장 낮은 checkpoint를 선택하는 방식이 더 적절합니다.

---

## 14. 최종 결론

정리하겠습니다.

이번 프로젝트에서 저희는 PyTorch만 사용해서 byte-level BPE tokenizer, GPT dataset, causal multi-head attention, TransformerBlock, GPTModel, 사전 학습 루프, 감성 분류 fine-tuning까지 직접 구현했습니다.

실험적으로는 Light 설정에서 빠르게 여러 후보를 비교하고, sentiment validation accuracy와 validation loss를 기준으로 Model-Base를 선택했습니다. 선택된 설정은 `emb_dim=192`, `n_layers=4`, `n_heads=4`, `drop_rate=0.1`, `lr=3e-4`였습니다.

이후 이 조합을 Basic 설정으로 확장했습니다. Basic에서는 더 큰 corpus, 더 큰 vocabulary, 더 긴 context length, 더 긴 epoch를 사용했고, 최종적으로 sentiment validation accuracy 0.8080, test accuracy 0.8047을 얻었습니다.

이번 실험에서 가장 중요한 배운 점은 세 가지입니다.

첫 번째, 사전 학습 loss만 보고 모델을 고르면 안 됩니다. 최종 목표가 감성 분류라면 downstream validation accuracy를 함께 봐야 합니다.

두 번째, 모델을 크게 만든다고 무조건 성능이 좋아지지는 않습니다. 데이터 크기, epoch, dropout, learning rate, checkpoint 선택이 같이 맞아야 큰 모델의 용량이 일반화 성능으로 이어집니다.

세 번째, 실험 결과를 해석할 때는 설정 차이를 조심해야 합니다. Light와 Basic은 여러 조건이 동시에 바뀌었기 때문에, 성능 상승을 특정 파라미터 하나의 효과라고 단정하기보다는, Light에서 후보를 선별하고 Basic에서 확장 가능성을 확인한 결과로 보는 것이 더 정확합니다.

이상으로 발표를 마치겠습니다. 감사합니다.
