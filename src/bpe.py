# -*- coding: utf-8 -*-
"""
UTF-8 byte-level BPE 토크나이저 과제 템플릿.

외부 tokenizer 라이브러리 없이 BPE(Byte Pair Encoding)를 직접 구현합니다.
한국어 NSMC 리뷰를 다루므로 문자열을 글자/공백 단위로 먼저 자르지 말고,
항상 `text.encode("utf-8")`로 byte ID 시퀀스를 만든 뒤 merge를 적용하세요.
"""

from pathlib import Path


PAD_TOKEN = "<pad>" #길이 맞추기 위한 padding
UNK_TOKEN = "<unk>" #모르는 토큰
BOS_TOKEN = "<bos>" #문장 시작
EOS_TOKEN = "<eos>" #문장 끝

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
        self.vocab_size = vocab_size #최종 vocabulary 크기
        self.id_to_token = {} #token ID -> token 내용
        self.token_to_id = {} #token 내용 -> token ID
        self.merges = [] #학습한 BPE merge rule 목록

    #초기 vocabulary 만드는 함수
    def _init_special_tokens(self):
        """
        TODO:
        1. 특수 토큰 4개를 고정 ID 0~3에 등록합니다.
        2. byte 0~255를 ID 4~259에 bytes([byte_value]) 형태로 등록합니다.
        """
        #어휘사전 만들기

        # 단어 -> id 인덱스로 바꾸는거

        # 특수 토큰 등록
        for i in range(len(SPECIAL_TOKENS)):
            self.id_to_token[i] = SPECIAL_TOKENS[i] #id에 토큰 등록
            self.token_to_id[SPECIAL_TOKENS[i]] = i #토큰 인덱스에 id 등록

        for byte_value in range(0, 255 + 1):
            token_id = byte_value + BYTE_OFFSET # i에 4를 더해준다
            byte_token = bytes([byte_value]) #숫자 하나를 1바이트 데이터로 바꾼다

            self.id_to_token[token_id] = byte_token #id 인덱스에 byte 문자 저장
            self.token_to_id[byte_token] = token_id #byte 문자에 id 저장


        # raise NotImplementedError("_init_special_tokens를 구현하세요.")

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

    def get_pair_counts(self, ids: list[int]) -> dict[tuple[int, int], int]:
        pair_counts = {}

        for i in range(len(ids) - 1):
            pair = (ids[i], ids[i + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

        return pair_counts

    def replace_pair(self, ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
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

    def train(self, corpus: str):
        """
        TODO: 코퍼스에서 BPE merge rule과 vocabulary를 학습합니다.

        구현 힌트:
        - `corpus.encode("utf-8")`로 byte ID 시퀀스를 만듭니다.
        - 가장 자주 등장하는 이웃 token pair를 찾습니다.
        - 새 token ID를 만들고, 시퀀스의 해당 pair를 새 ID로 치환합니다.
        - `self.merges`, `self.id_to_token`, `self.token_to_id`를 갱신합니다.
        """
        
        #1. 기본 vocabulary 초기화
        # 0~3 : <pad>, <unk>, <bos>, <eos>
        # 4~259 : byte 0 ~ 255
        self._init_special_tokens()

        #이전에 학습한 merge rule이 있을 수 있으므로 초기화
        self.merges = []

        #2. 문자열을 UTF-8 byte ID 리스트로 변환
        #ex) : 'AB' -> bytes 65, 66 -> token ID 69, 70
        #ex) : "가" -> bytes [234, 176, 128] -> IDs [238, 180, 132]
        ids = [BYTE_OFFSET + byte_value for byte_value in corpus.encode("utf-8")]

        # corpus가 너무 짧으면 pair가 없으므로 학습할 게 없음
        if len(ids) < 2:
            return
        
        #BPE merge로 새로 만들어지는 토큰 ID는 260부터 시작
        next_id = BYTE_OFFSET + NUM_BYTES

        #vocab_size에 도달할 때까지 반복
        while next_id < self.vocab_size:
            #현재 ids에서 이웃 pair 등장 횟수 세기
            # ex){(101,102) : 3, (102,101) : 2}
            pair_counts = self.get_pair_counts(ids)

            # 더 이상 pair가 없으면 종료
            if not pair_counts:
                break
            
            #가장 자주 등장한 pair 찾기
            #ex) best_pair = (101,102)
            best_pair = max(pair_counts, key=pair_counts.get)

            #ex) best_count = 3
            best_count = pair_counts[best_pair]

            #한 번만 나온 pair는 합쳐도 압축 효과가 거의 없으므로 중단
            if best_count < 2:
                break
            
            # 새 token ID 등록
            # ex) 260 = (238, 180)
            self.id_to_token[next_id] = best_pair
            self.token_to_id[best_pair] = next_id

            # merge rule 저장
            # encode()에서 이 순서대로 merge를 적용해야 함
            # ex) merges = [(101,102)]
            self.merges.append(best_pair)

            # 실제 ids 안의 best_pair를 next_id로 치환
            # ex) 원래 ids = 101 102 101 102 101 102
            # 260 260 260
            ids = self.replace_pair(ids, best_pair, next_id)

            # 다음 새 토큰 id 준비
            next_id += 1


        raise NotImplementedError("BPETokenizer.train을 구현하세요.")

    def save(self, path: str | Path):
        """
        TODO: vocabulary와 merge rule을 JSON 파일로 저장합니다.

        bytes와 tuple은 JSON에 바로 저장할 수 없으므로 type 정보를 함께 저장하세요.
        """
        raise NotImplementedError("BPETokenizer.save를 구현하세요.")

    def load(self, path: str | Path):
        """
        TODO: save()로 저장한 JSON 파일을 읽어 vocabulary와 merge rule을 복원합니다.
        """
        raise NotImplementedError("BPETokenizer.load를 구현하세요.")

    def encode(self, text: str, add_bos_eos: bool = False) -> list[int]:
        """
        TODO: 문자열을 token ID 리스트로 변환합니다.

        구현 힌트:
        - 먼저 UTF-8 byte ID 리스트를 만듭니다.
        - train/load에서 얻은 merge rule을 학습 순서대로 적용합니다.
        - add_bos_eos=True이면 앞뒤에 bos/eos ID를 붙입니다.
        """

        #text를 UTF-8 byte ID 리스트로 바꾼다.
        #ex) low  [108 + 4, 111 + 4, 119 + 4]
        byte_id_list =  [BYTE_OFFSET + byte_value for byte_value in text.encode("utf-8")]

        #next_id를 260으로 둔다.
        # next_id = BYTE_OFFSET + NUM_BYTES

        #self.merges를 순서대로 돈다.
        for merge_pair in self.merges:

            new_id = self.token_to_id[merge_pair]
            #byte_id_list에서 merge_pair을 next_id로 치환
            #ex) (112,115) , (260, 123), (261, 36)
            byte_id_list = self.replace_pair(byte_id_list, merge_pair, new_id)

            new_id += 1

        #add_bos_eos가 True면 앞뒤에 bos/eos를 붙인다.
        if(add_bos_eos):
            byte_id_list = [self.get_bos_id()] + byte_id_list + [self.get_eos_id()]

        return byte_id_list

        raise NotImplementedError("BPETokenizer.encode를 구현하세요.")

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """
        TODO: token ID 리스트를 문자열로 복원합니다.

        주의:
        - merge token은 원본 byte token까지 재귀적으로 펼칩니다.
        - byte를 하나씩 decode하지 말고, 마지막에 `bytes(...).decode("utf-8")`를 한 번만 호출합니다.
        """
        def expand_token(token_id : int) -> list[int]:
            #각 id를 확인한다
            for id in ids:
                token = self.id_to_token[id]

                #id가 SPECIAL_ID이면
                if (isinstance(token, str)) and (skip_special is True):
                    if skip_special:
                        result = []
                    else:
                        result = list(token.encode("utf-8"))

                elif isinstance(token, bytes):
                    result = list(token)

                elif isinstance(token, tuple):
                    left_id, right_id = token
                    result = expand_token(left_id) + expand_token(right_id)

                else:
                    raise

        #special token이면 skip_special 옵션에 따라 건너뛴다

        #byte token이면 byte 값으로 바꾼다

        #merge token이면 내부 token들을 계속 펼친다

        #모든 결과를 하나의 byte 리스트로 모인다

        #마지막에 bytes(byte_list).decode('utf-8) 한 번만 한다

        raise NotImplementedError("BPETokenizer.decode를 구현하세요.")
