from vllm.vllm import SamplingParams
from vllm.vllm import LLM
from transformers import AutoTokenizer


class LanguageModel:
    """LLM
    vllm을 활용한 LLM 세부 구현체
    completion 형태의 메시지를 구성하고 발화한다.
    prompt 엔지니어링과 history 관리를 수행한다.
    """

    def __init__(
        self,
        model_name: str,
        max_model_len: int = 16384,
        max_new_tokens: int = 48,
    ):
        self.max_new_tokens = max_new_tokens
        self.model = LLM(
            model=model_name,
            dtype="float32",
            max_model_len=max_model_len,
            # device="mps",
        )

        # sampling_params = SamplingParams(max_tokens=max_new_tokens)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.base_prompt = """
        You are designed to answer user questions accurately and effectively.
        Reference documents will be provided and if the references contain useful information then use it.
        But using them is not mendatory, if they are not relevant you could ignore them.
        Also Your response should be short, compact, clear, well-structured, compact and informative.
        """

        self.cites_prompt = """
        Now, here are some hint Documents.
        1. Doc1: Israel Adesanya fights in the middleweight division and was the middleweight champion from 2019 to 2022.
        2. Doc: 2024 UFC middleweight champion is Dricus du Plessis. and he got his next fight at UFC 312 fight card with his contender Sean Strickland.
        2. Doc3: In the wake of Yoel Romero’s stunning knock out victory against Luke Rockholdat UFC 221, there are a number of possible matchups at 2021

        Please answer the question.
        """

        self.base_system_message = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

    def chat(self, conversation: list):
        """챗
        채팅 형식으로 대화를 받아 입력으로 넣어 발화를 생성해 반화한다.

        Args:
            conversation (list): (system | user | assistant) role을 가지는 (role, content) 구조의 대화 리스트

        Examples::
            conversation = [
                {
                    "role": "system",
                    "content": base_prompt
                },
                {
                    "role": "user",
                    "content": prompt,
                },
                {
                    "role": "system",
                    "content": cites,
                },
            ]
        """
        sampling_params = SamplingParams(max_tokens=self.max_new_tokens)
        output = self.model.chat(conversation, sampling_params=sampling_params)

        return output
