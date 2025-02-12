"""
LocalRAG는 특정 주제에 대해 데이터를 모으고 요약해 LLM에 제공하는 RAG 테크닉을 사용하는 클래스이다.
Mac OS 로컬 환경에서 이에 필요한 LLM, 요약 모델, 임베딩 모델들을 모두 직접 호스팅한다.
"""

from db.vector_db import VectorDB
from embedding.text_embedding import Vectorizer
from crawler.medium_crawler import MediumCrawler
from language_model.llm import LanguageModel
from summary.summarize import Summarizer


class LocalRAG:
    """로컬 RAG
    주제에 대한 크롤링, 글 요약, 벡터 임베딩을 수행하고
    사용자가 해당 주제에 대해 쿼리를 요청하면 LLM모델에 관련있는 크롤링 데이터의 요약본을 함께 제공하여
    보다 신뢰도 있는 답변을 반환한다.
    """
    def __init__(
        self,
        crawler: MediumCrawler,
        vecotr_db: VectorDB,
        vectorizer: Vectorizer,
        summarizer: Summarizer,
        llm: LanguageModel,
    ):
        self.crawler = crawler
        self.vector_db = vecotr_db
        self.vectorizer = vectorizer
        self.summarizer = summarizer
        self.llm = llm

        self.base_system_message = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        self.base_direction = """
        You are designed to answer user questions with precision and efficiency.

        - Keep your responses concise and to the point.
        - You may be provided with reference documents. Use them if they contain relevant information.
        - If the references are not useful, feel free to ignore them.
        """

    def __call__(self, query: str, num_docs: int):
        """질문 답변
        유저의 쿼리에 대해 답변한다.
        RAG를 위한 사전 준비작업은 이미 완료된 것을 전제로 한다. (`set_rag_background` method)

        유저의 쿼리를 포함해 messages 형태의 대화 프롬프트를 구성하여 모델에 생성 요청한다.
        1. db 에서 쿼리와 관련있는 벡터들을 검색해 관련 문서 N개를 추출한다.
        2. 문서들의 요약문을 활용해 prompt를 구성한다.
        3. 모델에 답변 생성 요청

        Args:
            query (str): 유저의 입력 질문
            num_docs (int): 참조 관련 문서 개수

        Returns:
            answer (str): 모델 답변

        """
        # 주제 선정과 이후 해당 주제에 대한 크롤링, 문서 추출 그리고 요약과 벡터화까지 마친 상태에서 수행된다.
        # 사용자의 질문에 RAG 테크닉을 이용하여 답변한다.
        query_vector = self.vectorizor([query])[0]["embeddings"].reshape(1, -1)
        distances, indicies = self.vector_db.search(
            query=query_vector,
            k=num_docs,
        )

        print(f"검색 결과는 다음과 같습니다.\ndistance: {distances}\nindicies: {indicies}")

        conversation = self._get_conversation(
            query=query,
            doc_indicies=indicies[0],
        )

        res = self.llm.chat(conversation=conversation)

        return res


    def _get_conversation(self, query: str, doc_indicies: list):
        """대화 구성
        모델에 입력할 대화를 구성한다.
        (user/assistant/system) 중 하나의 role을 가지며 앞서 구한 참고 문서들을 활용해 입력 prompt를 만든다.

        Args:
            query (str): 유저 입력
            doc_indicies (list): 연관 문서 인덱스 리스트
        """
        docs = [self.summarized_articles[index] for index in doc_indicies]

        cite_strings = []
        for idx, doc in enumerate(docs):
            cite_strings.append(f"Doc {idx}: {doc}")
        cite_prompt = "\n".join(cite_strings)


        conversation = [
            {
                "role": "system",
                "content": f"{self.base_system_message}\n{self.base_direction}"
            },
            {
                "role": "user",
                "content": query,
            },
            {
                "role": "assistant",
                "content": "Do you have any reference documents for me to consider?",
            },
            {
                "role": "user",
                "content": f"Here are some reference documents.\n{cite_prompt}",
            },
        ]

        return conversation


    def set_rag_background(self, topic: str):
        # 1. 문서들을 추출한다. [(제목, 내용), ...]
        articles = self._get_topic_articles(topic=topic)
        # "제목: 내용" 형태로 변환한다.
        formed_articles = list(map(lambda article: f"{article['title']: article['content']}", articles))

        # 2. 문서들을 요약한다.
        # (summary, elapsed_time) 구조를 변환한다.
        summarize_results = self.summarizer(texts = formed_articles)
        self.summarized_articles = []
        for summarize_result in summarize_results:
            self.summarized_articles.extend(summarize_result)

        # 3. 벡터화하고 벡터 db를 만든다.
        vectorized_results = self.vectorizer(texts=self.summarized_articles)
        vectors = []
        for vectorized_result in vectorized_results:
            vectors.extend(vectorized_result["embeddings"])

        self.vector_db.add(vectors=vectors)

        print("rag background settings done !")



    def _get_topic_articles(
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
        articles = []
        urls = self.crawler.get_topic_urls(topic=topic)

        for url in urls:
            parsed_bs_obj = self.crawler.parse(url=url)
            title = self.crawler.get_title(parsed_bs_obj)
            content = self.crawler.get_content(parsed_bs_obj)
            # (제목, 본문) 형태로 추가
            articles.append({
                "title": title,
                "content": content,
            })

        return articles


if __name__ == "__main__":
    crawler = MediumCrawler()
    vector_db = VectorDB()
    vectorizer = Vectorizer()
    summarizer = Summarizer()
    # llm = LanguageModel(model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

    # local_rag = LocalRAG(
    #     crawler=crawler,
    #     vecotr_db=vector_db,
    #     vectorizer=vectorizer,
    #     summarizer=summarizer,
    #     llm=llm,
    # )

    # # 주제 선택
    # local_rag.set_rag_background(topic="beverage")
    # # 쿼리
    # res = local_rag(query="what's unique beverage nowdays")
    # print(res)
