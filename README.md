# 🔮 Local RAG
**내 맥에서 호스팅하는 RAG 테크닉**

Local RAG는 Apple 실리콘칩 MPS를 이용해 온전히 모든 것을 호스팅하면서도 쓸만한 답변을 얻어낼 수 있음을 테스트하는 프로젝트입니다.

프로젝트를 시작하게된 배경에는 다음과 같은 요소들이 있었습니다.
- deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B 등장과 오픈소스 공개 (25.01.22) - [link](https://arxiv.org/abs/2501.12948)
- vLLM에서의 부분적 MPS 지원 (25.01.08) - [link](https://github.com/vllm-project/vllm/pull/11696)
- 나는 M2 Air 16G 장비를 가지고 있음 💻

맥북을 가진 개인 누구나 간단히 로컬에서 LLM, 요약, 임베딩을 포함한 모델들을 호스팅하는 것. 

나만의 검색 증강으로 LLM으로부터 원하는 답변을 얻어낼 수 있는 것을 확인하고자 하였습니다.

## Composition
레포는 크게 5개의 디렉토리로 구성되어 있습니다.

- 모델 파트
    - /language_model
    - vllm에 있는 LLM 클래스를 활용한 언어 모델이 담긴 곳입니다.
    - 파이프라인의 최종 단계인 LLM 발화 생성에 사용됩니다.
- 크롤링 파트
    - /crawler
    - LLM에 제공할 참조 문서를 찾아내는 파트입니다.
    - 현재는 medium 블로그를 기본으로 사용하고 있습니다.
- 요약 파트
	- /summary
	- 크롤링해 추출한 문서를 요약합니다.
- 문서 임베딩 파트
	- /embedding
	- 요약한 문서를 벡터 공간에 임베딩합니다.
- 데이터 베이스
	- /db
	- 임베딩 벡터를 보관/검색하는 db입니다.

## Workflow
![image](https://github.com/user-attachments/assets/34b83966-d2e4-436e-8eb2-fbb4428539d4)

## Installation
Open a terminal and run:

```
$ conda create -n local_rag python=3.9
$ conda activate local_rag

$ pip3 install -r requirements.txt

$ sh install_vllm.sh
```
## Run
Open a terminal and run:

### just deploy and try
```
$ python3 -m local_rag --question "Who is the best boxer in the world"
```
<img width="2541" alt="image" src="https://github.com/user-attachments/assets/2c372516-b41e-4746-aac6-11b252231661" />


## Limitation of this project
Qwen 1.5B 모델의 성능이 생각보다 좋지 않았습니다. 원인을 생각해보자면.. R1은 reasoning 모델입니다. 1.5B의 작은 크기 모델에서는 스스로의 사전학습 지식으로 알아서 추론하는 경우보다 주어진 데이터를 근거로 생각하고 추론하는 경향이 느껴졌습니다. 따라서 좋은 문서를 넣어야 좋은 답변을 얻을 수 있을 것이라 생각합니다. (좀 더 질문 의도에 맞는 명확한 정보를 크롤링하여 전달하는 것이 필요함.)

또한 1.5B는 한국말을잘 못하는 것 같아 영어를 전제로 하고 있습니다.

언어 설정이 영어에 제한되는 점 또한 Local RAG를 피부로 느끼기에 아쉬운 부분입니다.
