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
        for token, idx in SPECIAL_IDS.items():
            self.id_to_token[idx] = token
            self.token_to_id[token] = idx

        # NUM_BYTES = 256 
        # BYTE_OFFSET = 4
        for byte_value in range(NUM_BYTES):
            token_id = BYTE_OFFSET + byte_value
            byte_token = bytes([byte_value])

            self.id_to_token[token_id] = byte_token
            self.token_to_id[byte_token] = token_id

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

    def _merge_rule_helper(self, ids : list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        new_ids = []
        i = 0

        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        
        return new_ids

    def expand_token(self, token_id: int) -> bytes:
        token = self.id_to_token[token_id]

        if isinstance(token, bytes):
            return token

        if isinstance(token, tuple):
            left_id, right_id = token
            return self.expand_token(left_id) + self.expand_token(right_id)

        if isinstance(token, str):
            return b""

        raise ValueError(f"Unknown token type: {type(token)}")

    def train(self, corpus: str):
        """
        DONE: 코퍼스에서 BPE merge rule과 vocabulary를 학습합니다.

        구현 힌트:
        - `corpus.encode("utf-8")`로 byte ID 시퀀스를 만듭니다.
        - 가장 자주 등장하는 이웃 token pair를 찾습니다.
        - 새 token ID를 만들고, 시퀀스의 해당 pair를 새 ID로 치환합니다.
        - `self.merges`, `self.id_to_token`, `self.token_to_id`를 갱신합니다.
        """
        # 0. 기존 vocab/merge 초기화
        self.id_to_token = {}
        self.token_to_id = {}
        self._init_special_tokens()
        self.merges = []

        # 1. corpus를 UTF-8 byte로 바꾸고, byte token ID sequence로 변환
        ids = []

        for byte in corpus.encode("utf-8"):
            token_id = BYTE_OFFSET + byte
            ids.append(token_id)

        # 2. vocab_size에 도달할 때까지 merge 반복
        while len(self.id_to_token) < self.vocab_size:
        # sequence 길이가 2보다 작으면 이웃 pair가 없음
            if len(ids) < 2:
                break

        # 3. 이웃 token pair 빈도 세기
            pair_counts = {}

            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i + 1])

                if pair not in pair_counts:
                    pair_counts[pair] = 0

                pair_counts[pair] += 1

        # pair가 없으면 중단
            if not pair_counts:
                break

            # 4. 가장 자주 나온 pair 선택
            best_pair = max(pair_counts, key=pair_counts.get)

            # 5. 새 token ID 만들기
            new_id = len(self.id_to_token)

            # 6. vocab과 merge rule 갱신
            self.id_to_token[new_id] = best_pair
            self.token_to_id[best_pair] = new_id
            self.merges.append(best_pair)

            # 7. sequence에서 best_pair를 new_id로 치환
            ids = self._merge_rule_helper(ids, best_pair, new_id)
        
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
        # 기존 vocab을 사용해서 ID로 변환
        """
        TODO: 문자열을 token ID 리스트로 변환합니다.
        구현 힌트:
        - 먼저 UTF-8 byte ID 리스트를 만듭니다.
        - train/load에서 얻은 merge rule을 학습 순서대로 적용합니다.
        - add_bos_eos=True이면 앞뒤에 bos/eos ID를 붙입니다.
        """
        # 1. base vocab이 비어 있으면 초기화
        if not self.id_to_token:
            self._init_special_tokens()

        # 2. 문자열을 UTF-8 byte token ID로 변환
        ids = []

        for byte_value in text.encode("utf-8"):
            token_id = BYTE_OFFSET + byte_value
            ids.append(token_id)

        # 3. train/load에서 저장된 merge rule을 학습 순서대로 적용
        for pair in self.merges:
            new_token_id = self.token_to_id[pair]

            ids = self._merge_rule_helper(ids, pair, new_token_id)
        # 4. 필요하면 bos/eos 추가 
        if add_bos_eos:
            ids = [self.get_bos_id()] + ids + [self.get_eos_id()]

        return ids 


    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """
        TODO: token ID 리스트를 문자열로 복원합니다.

        주의:
        - merge token은 원본 byte token까지 재귀적으로 펼칩니다.
        - byte를 하나씩 decode하지 말고, 마지막에 `bytes(...).decode("utf-8")`를 한 번만 호출합니다.
        """
        byte_chunks = []

        for token_id in ids: 
            token = self.id_to_token[token_id]

            if isinstance(token, str):
                if skip_special:
                    continue
                byte_chunks.append(token.encode("utf-8"))
                continue     
                
            byte_chunks.append(self.expand_token(token_id))
        return b"".join(byte_chunks).decode("utf-8")
