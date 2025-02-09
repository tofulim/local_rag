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
    ):
        self.model = LLM(
            model=model_name,
            dtype="float32",
            max_model_len=max_model_len,
            # device="mps",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)


# ---
from vllm.vllm import SamplingParams
from vllm.vllm import LLM
from transformers import AutoTokenizer

llm = LLM(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    dtype="float32",
    max_model_len=49028,
    # device="mps",
)


# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

# Prepare your prompts
base_prompt = """
You are an AI assistant designed to answer user questions accurately and effectively.
Reference documents are provided, but you are **not required** to use them if they are not relevant.
If the references contain useful information, cite them appropriately (e.g., **(Doc 1)**).
Otherwise, answer based on your general knowledge.
Your response should be **clear, well-structured, and informative**.
"""
cites = """
Now, here are some Documents and summaries of those you could refer.
1. Document 1 Summary: Israel Adesanya fights in the middleweight division and was the middleweight champion from 2019 to 2022.
2. Document 2 Summary: 2024 UFC middleweight champion is Dricus du Plessis. and he got his next fight at UFC 312 fight card with his contender Sean Strickland.
2. Document 3 Summary: In the wake of Yoel Romero’s stunning knock out victory against Luke Rockholdat UFC 221, there are a number of possible matchups at 2021

Please answer the question.
"""
prompt = f"who is the 2024 UFC middleweight champion?{cites}"
messages = [
    {"role": "system", "content": f"You are Qwen, created by Alibaba Cloud. You are a helpful assistant. {base_prompt}"},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False
)



query = "who is the 2024 UFC middleweight champion?"

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

sampling_params = SamplingParams(max_tokens=256)
output = llm.chat(conversation, sampling_params=sampling_params)
# text='</think>\n\nThe 2024 UFC middleweight champion is Dricus'
print(output)
