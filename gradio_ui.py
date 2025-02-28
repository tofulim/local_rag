import gradio as gr
from local_rag import LocalRAG
from crawler.medium_crawler import MediumCrawler
from db.vector_db import VectorDB
from embedding.text_embedding import Vectorizer
from language_model.llm import LanguageModel
from summary.summarize import Summarizer


crawler = MediumCrawler()
vector_db = VectorDB()
vectorizer = Vectorizer()
summarizer = Summarizer()
llm = LanguageModel(model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", max_new_tokens=128)

local_rag = LocalRAG(
    crawler=crawler,
    vecotr_db=vector_db,
    vectorizer=vectorizer,
    summarizer=summarizer,
    llm=llm,
)


def chatbot(question: str, history: list = []):
    # 주제 선택
    local_rag.set_rag_background(topic=question)
    # 쿼리
    response = local_rag(
        query=question,
        num_docs=3,
    )
    model_output = response[0].outputs[0].text

    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": model_output})
    return history

iface = gr.Interface(
    fn=chatbot,
    inputs=gr.Textbox(placeholder="질문을 입력하세요"),
    outputs=gr.Chatbot(type="messages")  # type="messages" 추가
)

if __name__ == "__main__":
    iface.launch()
