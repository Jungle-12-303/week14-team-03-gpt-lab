# -*- coding: utf-8 -*-
"""
UTF-8 byte-level BPE 토크나이저 과제 템플릿.

외부 tokenizer 라이브러리 없이 BPE(Byte Pair Encoding)를 직접 구현합니다.
한국어 NSMC 리뷰를 다루므로 문자열을 글자/공백 단위로 먼저 자르지 말고,
항상 `text.encode("utf-8")`로 byte ID 시퀀스를 만든 뒤 merge를 적용하세요.
"""

from pathlib import Path
import json


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
        self.id_to_token = {}
        self.token_to_id = {}
        self.merges = []

    def _init_special_tokens(self):
        """
        DONE: 특수 토큰 및 바이트를 단어사전에 저장
        1. 특수 토큰 4개를 고정 ID 0~3에 등록합니다.
        2. byte 0~255를 ID 4~259에 bytes([byte_value]) 형태로 등록합니다.
        """
        for token, idx in SPECIAL_IDS.items():
            self.id_to_token[idx] = token
            self.token_to_id[token] = idx

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

    def _merge_rule_helper(self, ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        new_ids = []
        i = 0

        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
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
        self.id_to_token = {}
        self.token_to_id = {}
        self._init_special_tokens()
        self.merges = []

        ids = []
        for byte in corpus.encode("utf-8"):
            token_id = BYTE_OFFSET + byte
            ids.append(token_id)

        while len(self.id_to_token) < self.vocab_size:
            if len(ids) < 2:
                break

            pair_counts = {}
            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i + 1])

                if pair not in pair_counts:
                    pair_counts[pair] = 0

                pair_counts[pair] += 1

            if not pair_counts:
                break

            best_pair = max(pair_counts, key=pair_counts.get)
            best_count = pair_counts[best_pair]

            # 2번 이상 등장하는 pair만 merge해야 실제로 시퀀스가 압축됩니다.
            # 한 번만 나온 pair까지 새 토큰으로 만들면 vocab만 늘고 일반화에 도움이 되지 않으므로,
            # 요구서의 "더 이상 자주 등장하는 pair가 없을 때 멈춤" 조건으로 보고 중단합니다.
            if best_count < 2:
                break

            new_id = len(self.id_to_token)

            self.id_to_token[new_id] = best_pair
            self.token_to_id[best_pair] = new_id
            self.merges.append(best_pair)

            ids = self._merge_rule_helper(ids, best_pair, new_id)

    def save(self, path: str | Path):
        """
        TODO: vocabulary와 merge rule을 JSON 파일로 저장합니다.

        bytes와 tuple은 JSON에 바로 저장할 수 없으므로 type 정보를 함께 저장하세요.
        """
        merge_rules = [[a, b] for a, b in self.merges]
        data = {
            "vocab_size": self.vocab_size,
            "merges": merge_rules,
            "id_to_token": {},
        }

        for token_id, token in self.id_to_token.items():
            if isinstance(token, bytes):
                data["id_to_token"][str(token_id)] = {
                    "type": "bytes",
                    "value": list(token),
                }
            elif isinstance(token, str):
                data["id_to_token"][str(token_id)] = {
                    "type": "str",
                    "value": token,
                }
            elif isinstance(token, tuple):
                data["id_to_token"][str(token_id)] = {
                    "type": "tuple",
                    "value": list(token),
                }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str | Path):
        """
        TODO: save()로 저장한 JSON 파일을 읽어 vocabulary와 merge rule을 복원합니다.
        """
        self.id_to_token = {}
        self.token_to_id = {}

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.vocab_size = data["vocab_size"]
        merge_rules = data["merges"]
        self.merges = [(a, b) for a, b in merge_rules]

        token_dict = data["id_to_token"]
        for token_id, token_data in token_dict.items():
            token_type = token_data["type"]
            token = token_data["value"]

            if token_type == "bytes":
                self.id_to_token[int(token_id)] = bytes(token)
            elif token_type == "str":
                self.id_to_token[int(token_id)] = token
            elif token_type == "tuple":
                self.id_to_token[int(token_id)] = tuple(token)

        self.token_to_id = {token: token_id for token_id, token in self.id_to_token.items()}

    def encode(self, text: str, add_bos_eos: bool = False) -> list[int]:
        """
        TODO: 문자열을 token ID 리스트로 변환합니다.

        구현 힌트:
        - 먼저 UTF-8 byte ID 리스트를 만듭니다.
        - train/load에서 얻은 merge rule을 학습 순서대로 적용합니다.
        - add_bos_eos=True이면 앞뒤에 bos/eos ID를 붙입니다.
        """
        if not self.id_to_token:
            self._init_special_tokens()

        ids = []
        for byte_value in text.encode("utf-8"):
            token_id = BYTE_OFFSET + byte_value
            ids.append(token_id)

        for pair in self.merges:
            new_token_id = self.token_to_id[pair]
            ids = self._merge_rule_helper(ids, pair, new_token_id)
        if add_bos_eos:
            ids = [self.get_bos_id()] + ids + [self.get_eos_id()]

        return ids

    def decode(
        self,
        ids: list[int],
        skip_special: bool = True,
        errors: str = "strict",
    ) -> str:
        """
        TODO: token ID 리스트를 문자열로 복원합니다.

        주의:
        - merge token은 원본 byte token까지 재귀적으로 펼칩니다.
        - byte를 하나씩 decode하지 말고, 마지막에 `bytes(...).decode("utf-8")`를 한 번만 호출합니다.
        - errors="strict"는 잘못된 UTF-8 byte 조합을 오류로 잡고,
          errors="replace"는 학습 중 생성 샘플의 깨진 byte를 �로 대체합니다.
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

        return b"".join(byte_chunks).decode("utf-8", errors=errors)
