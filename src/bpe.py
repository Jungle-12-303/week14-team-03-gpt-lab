# -*- coding: utf-8 -*-
"""
UTF-8 byte-level BPE 토크나이저 과제 템플릿.

외부 tokenizer 라이브러리 없이 BPE(Byte Pair Encoding)를 직접 구현합니다.
한국어 NSMC 리뷰를 다루므로 문자열을 글자/공백 단위로 먼저 자르지 말고,
항상 `text.encode("utf-8")`로 byte ID 시퀀스를 만든 뒤 merge를 적용하세요.
"""

from pathlib import Path
import json # json 저장을 위한 라이브러리

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
SPECIAL_IDS = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
BYTE_OFFSET = len(SPECIAL_TOKENS)
NUM_BYTES = 256


class BPETokenizer:
    """
    UTF-8 byte-level BPE 토크나이저.

    권장 ID 배치:
    - 0~3: <pad>, <unk>, <bos>, <eos>
    - 4~259: 원본 byte 0~255
    - 260 이상: BPE merge로 생성한 토큰
    """

    def __init__(self, vocab_size: int = 3000):
        self.vocab_size = vocab_size
        self.id_to_token = {} # 값(value)의 타입이 섞여있을 수 있음 (e.g. "<pad>", b"\x00", (101, 101) => 순서대로 문자열, 바이트, 튜플 타입)
        self.token_to_id = {} # key의 타입이 섞여있을 수 있음
        self.merges = [] # token ID 쌍(튜플 형태)을 순서대로 저장하는 배열 

    def _init_special_tokens(self):
        """
        DONE: 특수 토큰 및 바이트를 단어사전에 저장
        1. 특수 토큰 4개를 고정 ID 0~3에 등록합니다.
        2. byte 0~255를 ID 4~259에 bytes([byte_value]) 형태로 등록합니다.
        """
        # 1. 특수 토큰 4개를 고정 ID 0~3에 등록합니다.
        #   권장 ID 배치:
        #     - 0~3: <pad>, <unk>, <bos>, <eos>
        
        self.id_to_token[0] = PAD_TOKEN
        self.id_to_token[1] = UNK_TOKEN
        self.id_to_token[2] = BOS_TOKEN
        self.id_to_token[3] = EOS_TOKEN
        
        # 2. byte 0~255를 ID 4~259에 bytes([byte_value]) 형태로 등록합니다.
        for i in range (0, NUM_BYTES):
            self.id_to_token[i + BYTE_OFFSET] = bytes([i])
        
        # +) token_to_id 에도 저장
        self.token_to_id = { s:i for i, s in self.id_to_token.items() }

    def get_pad_id(self):
        """padding 토큰 ID."""
        return SPECIAL_IDS[PAD_TOKEN]

    def get_unk_id(self):
        """unknown 토큰 ID."""
        return SPECIAL_IDS[UNK_TOKEN]

    def get_bos_id(self):
        """문장 시작 토큰 ID."""
        return SPECIAL_IDS[BOS_TOKEN]

    def get_eos_id(self):
        """문장 끝 토큰 ID."""
        return SPECIAL_IDS[EOS_TOKEN]

    def train(self, corpus: str):
        """
        DONE: 코퍼스에서 BPE merge rule과 vocabulary를 학습합니다.

        구현 힌트:
        - `corpus.encode("utf-8")`로 byte ID 시퀀스를 만듭니다.
        - 가장 자주 등장하는 이웃 token pair를 찾습니다.
        - 새 token ID를 만들고, 시퀀스의 해당 pair를 새 ID로 치환합니다.
        - `self.merges`, `self.id_to_token`, `self.token_to_id`를 갱신합니다.
        """
        self._init_special_tokens()
        
        ids = [encoded + BYTE_OFFSET for encoded in corpus.encode("utf-8")]
        
        while (len(self.id_to_token) < self.vocab_size): # vocab_size 만큼 토큰 머지
            if len(ids) < 2: # 이웃 pair(token ID 쌍)를 만들 수 없으면 더 이상 merge 할 것이 없다는 의미이므로 반복문 중단
                break
            
            pair_counts = {} # token ID 쌍의 개수를 카운트 할 딕셔너리 생성
                        
            for i in range (len(ids)-1):
                pair = (ids[i], ids[i+1]) # token ID 쌍을 생성
                pair_counts[pair] = pair_counts.get(pair, 0) + 1 # pair_counts.get(pair, 0) == pair_counts 딕셔너리에 pair가 키인 값이 있다면 그 값을, 없다면 0을 가져와라.는 뜻
            
            if not pair_counts:
                break

            best_pair = max(pair_counts, key=pair_counts.get) # pair_counts 중 가장 큰 value를 가진 튜플(키)을 찾아서 반환
            
            new_idx = len(self.id_to_token)
            
            self.merges.append(best_pair)
            self.id_to_token[new_idx] = best_pair
            self.token_to_id[best_pair] = new_idx
            
            # 현재 corpus도 best pair로 바꿔주기
            i = 0
            new_ids = []
            while (i < len(ids)):
                if (i < len(ids) - 1) and (best_pair == (ids[i], ids[i+1])):
                    new_ids.append(new_idx)
                    i+=2
                else:
                    new_ids.append(ids[i])
                    i+=1
            ids = new_ids # 새로운 ids를 ids에 대입

    def save(self, path: str | Path):
        """
        TODO: vocabulary와 merge rule을 JSON 파일로 저장합니다.

        bytes와 tuple은 JSON에 바로 저장할 수 없으므로 type 정보를 함께 저장하세요.
        """
          
        # merge rule? BPE가 학습한 “어떤 token 쌍을 하나의 새 token으로 합칠지”에 대한 규칙
            # 예를 들어 merges = [(101, 101), (260, 102), (103, 104)] 라면,
            # 1번째 merge: (101, 101)을 합친다
            # 2번째 merge: (260, 102)를 합친다
            # 3번째 merge: (103, 104)를 합친다 는 의미
        
        merge_rules = [[a, b] for a, b in self.merges]
        data = {
            "vocab_size": self.vocab_size,
            "merges": merge_rules,
            "id_to_token": { },
            # token_to_id는 굳이 넣지 않는다! > id_to_token에서 다시 만들 수 있는 역방향 인덱스이기 때문   
        }
        
        # JSON에 넣을 토큰을 타입별로 저장 방식 다르게 진행
        for token_id, token in self.id_to_token.items():
            if isinstance(token, bytes):
                data["id_to_token"][str(token_id)] = {
                    "type": "bytes",
                    "value": list(token)
                }
            elif isinstance(token, str):
                data["id_to_token"][str(token_id)] = {
                    "type": "str",
                    "value": token
                }
            elif isinstance(token, tuple):
                data["id_to_token"][str(token_id)] = {
                    "type": "tuple",
                    "value": list(token)
                }
        
        # path에 JSON 파일로 저장
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        

    def load(self, path: str | Path):
        """
        TODO: save()로 저장한 JSON 파일을 읽어 vocabulary와 merge rule을 복원합니다.
        """
        # 초기화
        self.id_to_token = {} 
        self.token_to_id = {}
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f) # 파일 열기
        
        # 저장된 json에서 
        self.vocab_size = data["vocab_size"]
        merge_rules = data["merges"]
        self.merges = [(a, b) for a, b in merge_rules]
        
        # 토큰 복원
        # 바이트
        token_dict = data["id_to_token"] # 바이트, 문자, 튜플 등 형태 섞여있음
        for token_id, token_data in token_dict.items():
            token = token_data["value"]
            if token["type"] == "bytes":
                self.id_to_token[int(token_id)] = bytes(token)
            
            elif token["type"] == "str":
                self.id_to_token[int(token_id)] = token
            
            elif token["type"] == "tuple":
                self.id_to_token[int(token_id)] = tuple(token)

        # 토큰 > ID 역변환
        self.token_to_id = { s:i for i, s in self.id_to_token.items() }

    def encode(self, text: str, add_bos_eos: bool = False) -> list[int]:
        """
        TODO: 문자열을 token ID 리스트로 변환합니다.

        구현 힌트:
        - 먼저 UTF-8 byte ID 리스트를 만듭니다.
        - train/load에서 얻은 merge rule을 학습 순서대로 적용합니다.
        - add_bos_eos=True이면 앞뒤에 bos/eos ID를 붙입니다.
        """
        raise NotImplementedError("BPETokenizer.encode를 구현하세요.")

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """
        TODO: token ID 리스트를 문자열로 복원합니다.

        주의:
        - merge token은 원본 byte token까지 재귀적으로 펼칩니다.
        - byte를 하나씩 decode하지 말고, 마지막에 `bytes(...).decode("utf-8")`를 한 번만 호출합니다.
        """
        raise NotImplementedError("BPETokenizer.decode를 구현하세요.")
