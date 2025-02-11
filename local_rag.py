"""
LocalRAG는 특정 주제에 대해 데이터를 모으고 요약해 LLM에 제공하는 RAG 테크닉을 사용하는 클래스이다.
Mac OS 로컬 환경에서 이에 필요한 LLM, 요약 모델, 임베딩 모델들을 모두 직접 호스팅한다.
"""

import requests
from local_rag.db.vector_db import VectorDB
from local_rag.embedding.text_embedding import Vectorizer
from local_rag.summary.summarize import Summarizer


class LocalRAG:
    """로컬 RAG
    주제에 대한 크롤링, 글 요약, 벡터 임베딩을 수행하고
    사용자가 해당 주제에 대해 쿼리를 요청하면 LLM모델에 관련있는 크롤링 데이터의 요약본을 함께 제공하여
    보다 신뢰도 있는 답변을 반환한다.
    """
    def __init__(
        self,
        crawler: object,
        vecotr_db: VectorDB,
        vectorizer: Vectorizer,
        summarizer: Summarizer,
    ):
        self.crawler = crawler
        self.vdb = vecotr_db
        self.vectorizer = vectorizer
        self.summarizer = summarizer

    def __call__(self, query: str):
        # 사용자의 질문에 RAG 테크닉을 이용하여 답변한다.
        pass

    def get_topic_articles(
        self,
        topic: str,
    ):
        """주제 기사 수집
        특정 주제에 대해 기술한 글을 인터넷에서 N개 수집한다.
        (제목, 내용) 형태의 글 객체를 반환한다.

        Args:
            topic (str): 유저가 구축하기 원하는 RAG 시스템의 주제

        Returns:
            articles (list[dict]): 제목, 내용을 포함한 글 객체 리스트
        """
