"""
LocalRAG는 특정 주제에 대해 데이터를 모으고 요약해 LLM에 제공하는 RAG 테크닉을 사용하는 클래스이다.
Mac OS 로컬 환경에서 이에 필요한 LLM, 요약 모델, 임베딩 모델들을 모두 직접 호스팅한다.
"""
import numpy as np

from embedding.text_embedding import Vectorizer
from crawler.medium_crawler import MediumCrawler
from language_model.llm import LanguageModel
from summary.summarize import Summarizer
from db.vector_db import VectorDB

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
        query_vector = self.vectorizer([query])[0]["embeddings"].reshape(1, -1)
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
        formed_articles = list(map(lambda article: f"{article['title']}: {article['content']}", articles))

        # 2. 문서들을 요약한다.
        # (summary, elapsed_time) 구조를 변환한다.
        summarize_results = self.summarizer(texts = formed_articles)
        self.summarized_articles = []
        for summarize_result in summarize_results:
            self.summarized_articles.extend(summarize_result)

        # 3. 벡터화하고 벡터 db를 만든다.
        vectorized_results = self.vectorizer(texts=self.summarized_articles)
        vectors = np.vstack(list(map(lambda vectorized_result: vectorized_result["embeddings"], vectorized_results)))

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
    llm = LanguageModel(model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

    local_rag = LocalRAG(
        crawler=crawler,
        vecotr_db=vector_db,
        vectorizer=vectorizer,
        summarizer=summarizer,
        llm=llm,
    )

    # 주제 선택
    local_rag.set_rag_background(topic="beverage")
    # 쿼리
    res = local_rag(
        query="what's unique beverage nowdays",
        num_docs=3,
    )
    print(res)

    # 프롬프트 (docs가 좀 이상하게 들어갔음) 그래도 파이프라인은 대략 됨
    # RequestOutput(request_id=0, prompt="<｜begin▁of▁sentence｜>You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\n        You are designed to answer user questions with precision and efficiency.\n\n        - Keep your responses concise and to the point.\n        - You may be provided with reference documents. Use them if they contain relevant information.\n        - If the references are not useful, feel free to ignore them.\n        <｜User｜>what's unique beverage nowdays<｜Assistant｜>Do you have any reference documents for me to consider?<｜end▁of▁sentence｜><｜User｜>Here are some reference documents.\nDoc 0: time\nDoc 1: time\nDoc 2: summary_texts<｜Assistant｜><think>\n", prompt_token_ids=[151646, 151646, 2610, 525, 1207, 16948, 11, 3465, 553, 54364, 14817, 13, 1446, 525, 264, 10950, 17847, 382, 286, 1446, 525, 6188, 311, 4226, 1196, 4755, 448, 16052, 323, 15024, 382, 286, 481, 13655, 697, 14507, 63594, 323, 311, 279, 1459, 624, 286, 481, 1446, 1231, 387, 3897, 448, 5785, 9293, 13, 5443, 1105, 421, 807, 6644, 9760, 1995, 624, 286, 481, 1416, 279, 15057, 525, 537, 5390, 11, 2666, 1910, 311, 10034, 1105, 624, 260, 151644, 12555, 594, 4911, 42350, 1431, 13778, 151645, 5404, 498, 614, 894, 5785, 9293, 369, 752, 311, 2908, 30, 151643, 151644, 8420, 525, 1045, 5785, 9293, 624, 9550, 220, 15, 25, 882, 198, 9550, 220, 16, 25, 882, 198, 9550, 220, 17, 25, 12126, 79646, 151645, 151648, 198], encoder_prompt=None, encoder_prompt_token_ids=None, prompt_logprobs=None, outputs=[CompletionOutput(index=0, text='Okay, the user is asking about "what\'s unique beverage right now." They\'ve also provided some reference documents, but they won\'t go into details. Maybe they\'re looking for a quick answer without getting too bogged down.\n\nI should explain that the field is pretty broad, so I\'ll give examples of popular Chinese', token_ids=(32313, 11, 279, 1196, 374, 10161, 911, 330, 12555, 594, 4911, 42350, 1290, 1431, 1189, 2379, 3003, 1083, 3897, 1045, 5785, 9293, 11, 714, 807, 2765, 944, 728, 1119, 3565, 13, 10696, 807, 2299, 3330, 369, 264, 3974, 4226, 2041, 3709, 2238, 34419, 3556, 1495, 382, 40, 1265, 10339, 429, 279, 2070, 374, 5020, 7205, 11, 773, 358, 3278, 2968, 10295, 315, 5411, 8453), cumulative_logprob=None, logprobs=None, finish_reason=length, stop_reason=None)], finished=True, metrics=RequestMetrics(arrival_time=1739367192.726567, last_token_time=1739367225.29746, first_scheduled_time=1739367192.7344131, first_token_time=1739367211.25861, time_in_queue=0.00784611701965332, finished_time=1739367225.299223, scheduler_time=0.01094104199995627, model_forward_time=None, model_execute_time=None), lora_request=None, num_cached_tokens=0, multi_modal_placeholders={})]
