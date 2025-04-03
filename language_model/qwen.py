from language_model.llm import BaseLanguageModel


class DeepSeekQwen(BaseLanguageModel):
    """LLM
    vllm을 활용한 LLM 세부 구현체
    https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
    """

    def __init__(
        self,
        model_name: str,
        max_model_len: int = 16384,
        max_new_tokens: int = 64,
    ):
        super().__init__(model_name, max_model_len, max_new_tokens)

        self.base_system_message = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        self.base_direction = """
        You are designed to answer user questions with short, concise and simple.

        - You may be provided with reference documents. Use them if they contain relevant information.
        - If the references are not useful, feel free to ignore them. referring them are not mendatory.
        """
        self.base_document_request_message = "Do you have any reference documents for me to consider?"
        self.base_user_reference_message = "Here are some reference documents."
