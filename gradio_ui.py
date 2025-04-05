import gradio as gr
from local_rag import LocalRAG
from embedding.text_embedding import Vectorizer
from crawler.crawler_factory import CrawlerFactory
from language_model.model_factory import ModelFactory
from summary.summarize import Summarizer
from db.vector_db import VectorDB


vectorizer = Vectorizer(
    model_name="intfloat/multilingual-e5-small",
)
vector_db = VectorDB(dim=vectorizer.embedding_dim)

summarizer = Summarizer(model_name="digit82/kobart-summarization")
crawler = CrawlerFactory().get("tistory")()
llm = ModelFactory().get("kanana")(
    model_name="kakaocorp/kanana-nano-2.1b-instruct",
    max_new_tokens=256,
    max_model_len=4096,
)

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
    response, ref_docs = local_rag(
        query=question,
        num_docs=3,
    )
    model_output = response[0].outputs[0].text

    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": model_output})
    return history, '\n'.join(ref_docs)

iface = gr.Interface(
    fn=chatbot,
    inputs=gr.Textbox(label="질문", placeholder="질문을 입력하세요"),
    outputs=[
        gr.Chatbot(label="채팅", type="messages"),
        gr.Textbox(label="참고 자료", interactive=False),
    ],  # type="messages" 추가
)

if __name__ == "__main__":
    iface.launch()
