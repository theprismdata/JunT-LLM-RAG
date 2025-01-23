# 윈도우 노트북에서 RAG 실행해 보기!
동영상 데모
https://www.youtube.com/@HONGJOONG-SHIN

### 사전 준비
#### OpenSearch and Dashboard Docker Compose
Opensearch는 Single Node로 작동합니다.
Dashboard와 Opensearch 계정은 admin / juntPass123! 로 설정하였습니다.

```
set OPENSEARCH_INITIAL_ADMIN_PASSWORD=juntPass123!
docker-compose -f docker\docker-compose.yml up
```

접속 테스트
```
curl https://localhost:9200 -ku admin:juntPass123!
{
  "name" : "1543b25120d3",
  "cluster_name" : "docker-cluster",
  "cluster_uuid" : "7qHMHfNIRQqIf9BjxLVl7Q",
  "version" : {
    "distribution" : "opensearch",
    "number" : "2.18.0",
    "build_type" : "tar",
    "build_hash" : "99a9a81da366173b0c2b963b26ea92e15ef34547",
    "build_date" : "2024-10-31T19:08:39.157471098Z",
    "build_snapshot" : false,
    "lucene_version" : "9.12.0",
    "minimum_wire_compatibility_version" : "7.10.0",
    "minimum_index_compatibility_version" : "7.0.0"
  },
  "tagline" : "The OpenSearch Project: https://opensearch.org/"
}
```
참조.
https://opensearch.org/docs/latest/install-and-configure/install-opensearch/docker/

### Multiturn Hisotry DB
```
docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
```

##### Python 3.10 가상 환경 생성
```
python -m venv .venv 
```
가상 환경에 라이브러리 설치(Windows)
```
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

### 데이터 추출
추출할 데이터를 source_doc에 복사합니다.

데이터 추출 (현재는 PDF 포맷만 대상)
Preprocess로 이동하여 source_doc 폴더에 있는 파일들에 대해 추출을 시작합니다.

document_extract.py 스크립트를 실행
```commandline
cd .\Preprocess\
python .\document_extract.py -input ../source_doc
python .\document_extract.py -input F:\DC_NextCloud\03_DC_RnD_LAB

``` 

###  데이터 임베딩
doc_embedding 폴더로 이동합니다.<BR>
추출된 데이터의 경로를 설정합니다. <BR>
예) store.embedding_doc(file_path="../metadump.json")

embedding.py 스크립트를 실행합니다.
```commandline
cd .\doc_embedding\
python .\embedding.py
```
임베딩 모델은 open model인 snunlp/KR-SBERT-V40K-klueNLI-augSTS를 사용합니다. <br>
opensearch index는 "doc_embedding"를 기본 설정으로 합니다. <br>
index를 바꾸고 싶으면 store = OpenSearchEmbeddingStore()를<BR>
store = OpenSearchEmbeddingStore(index_name="new_name") 와 같은 방법으로 변경합니다.

임베딩이 완료되면 이제 검색이 예제와 같이 검색이 가능합니다.<BR>
query = "AI와 신경망에 대해 알려주세요"<BR>
results = store.search_similar(query)

예제를 실행시키면 아래의 결과를 얻을 수 있습니다.
```commandline
원본 파일: ..\source_doc\[KIPA 한국행정연구원] (이슈페이퍼)\2023-12-[KIPA 한국행정연구원] (이슈페이퍼) (2023-131호) 초격차 신산업 분야 정부기능 수행 문제점 .pdf
위치 페이지: 3
위치 줄번호: 22
점수: 0.78504634
텍스트: 로봇전문인재 플랫폼 운영
원본 파일: E:\1.Developing\opds\junt-llm-rag\source_doc\[KIPA 한국행정연구원] (이슈페이퍼)\2023-12-[KIPA 한국행정연구원] (이슈페이퍼) (2023-131호) 초격차 신산업 분야 정부기능 수행 문제점 .pdf
위치 페이지: 10
위치 줄번호: 2
점수: 0.78348976
텍스트: 로봇전문인재 플랫폼 운영
원본 파일: ..\source_doc\[KIPA 한국행정연구원] (이슈페이퍼)\2023-12-[KIPA 한국행정연구원] (이슈페이퍼) (2023-131호) 초격차 신산업 분야 정부기능 수행 문제점 .pdf
위치 페이지: 10
위치 줄번호: 2
점수: 0.78348976

```

### RAG 확장을 이용한 질의
우선 ollama을 PC에 설치 후 아래의 명령을 실행하여 PC에 모델을 다운로드 받습니다.
```commandline
ollama pull gemma2:2b
```

다음 명령을 수행하여 LLM을 통해 질문에 대한 답변을 정리합니다.
```
python .\RAGSearch.py -query "한국행정연구원의 신산업 정보를 요약해줘."
```
```
python .\RAGSearch.py -query "디지털트윈의 목적 및 구축･활용･효과 차원에서 견지해야 할 주요 원칙은"

질의 결과:
## 디지털트윈 주요 원칙:

디지털트윈은 과학적 예측을 기반으로 정책결정을 지원하는 도구로서 중요한 역할을 할 수 있습니다.  이는 여러 범주를 포함하 여 꾸준히 연구과 개발해야 합니다. 디지털 트윈의 성공적인 활용을 위한 주요 원칙은 다음과 같습니다:

**1. 과학적이고 예측적인 정책 결정 도구로서의 디지털트윈:**

* 디지털트윈은 과학적 기반으로 정책 수립 및 모니터링 시스템을 구축하여 정책결정에 대한 효율성을 높이는 데 중점을 둡니다.
*  사실적인 예측 모델과 데이터 기반 분석을 통해 정부 정책의 효율성을 평가하고 개선할 수 있습니다.
* **핵심:** 과학적 근거를 바탕으로 정책 결정 및 실행에 활용하는 디지털트윈은 정치적 편견이나 예측 불확실성을 최소화합니 다.

**2. 전문적인 디지털트윈 운영 체계 구축:**

*  전담 조직 설치와 관련된 전략(예: 5-2) 및 표준화(5-2)가 필요하며, 이를 통해 디지털트윈 사업의 성공 가능성과 지속성을  높이는데 중점을 둡니다.
*   변화에 대한 적응력과 미래 유망성을 고려하여 지속 가능한 발전 방향을 설정합니다.
* **핵심:** 전담 조직 설치와 다양한 표준화를 통해 디지털 트윈 활용의 효율성과 안정성을 확보합니다.

**3.  디지털트윈 사업의 지속 가능성을 위한 핵심 원칙:**

*   **인프라 및 시스템 구축:** 정확한 데이터 분석을 위한 고도화된 기술 기반 정보 시스템(infra)과 통합된 데이터베이스(system)를 구축해야 합니다.
* **디지털트윈의 차별성:**  다양한 분야에서 차별화된 기능 및 적용이 가능하도록 디지털 트윈 사업을 위한 정확하고 효율적인 시스템 구축이 필요합니다.

**4. 공통의 이해와 목적 지향적인 트윈 구축:**

* 디지털트윈의 실제 활용을 위해서는 공통된 이해와 원칙 기반의 목표 설정이 중요합니다.
*  다양한 분야의 전문가들의 의견과 협력을 통해 다층적인 관점에서 정책 결정에 대한 과학적 근거를 확립해야 합니다.

**결론:**

디지털트윈은 과학적 예측을 기반으로 정책 결정 지원 도구로서,  국가의 안보 및 경제 발전에 큰 역할을 할 수 있습니다. 끊임 없는 연구와 개발을 통해 디지털 트윈 활성화를 위한 전략과 원칙을 확립해야 합니다.



****세부 자료****
~~~~
텍스트: - 과학적･예측적 정책결정 도구로서의 디지털트윈의 목적 및 구축･활용･효과 차원에서 견지해야 할 주요 원칙은 다음과 같음
원본 파일: ..\source_doc\[KIPA 한국행정연구원] (이슈페이퍼)\2023-12-[KIPA 한국행정연구원] (이슈페이퍼) (2023-134호)디지 털트윈을 활용한 과학적·예측적 정책의사결정 활성화 방안(조세현+차남준+나보리+우하린+김상숙).pdf
위치 페이지: 9
위치 줄번호: 3
점수: 0.93782365
텍스트: 과학적･예측적 정책의사결정을 위한 디지털트윈 활성화 전략의 필요성
원본 파일: ..\source_doc\[KIPA 한국행정연구원] (이슈페이퍼)\2023-12-[KIPA 한국행정연구원] (이슈페이퍼) (2023-134호)디지 털트윈을 활용한 과학적·예측적 정책의사결정 활성화 방안(조세현+차남준+나보리+우하린+김상숙).pdf
위치 페이지: 3
위치 줄번호: 7
점수: 0.8927963
텍스트: 강화와 전담조직 설치를 통한 디지털트윈 사업의 성공 가능성 및 지속성
원본 파일: ..\source_doc\[KIPA 한국행정연구원] (이슈페이퍼)\2023-12-[KIPA 한국행정연구원] (이슈페이퍼) (2023-134호)디지 털트윈을 활용한 과학적·예측적 정책의사결정 활성화 방안(조세현+차남준+나보리+우하린+김상숙).pdf
위치 페이지: 12
위치 줄번호: 20
점수: 0.8887669
~~~~
```

혹은 다음 명령을 수행하여 LLM을 통해 과거의 질의 내용을 기반으로 하여 새로운 질문을 추가할 수도 있습니다.
```
python .\RAGSearch_MultiTurn.py -query "한국행정연구원의 신산업 정보를 요약해줘."
```

로컬 ollama 서버와의 연결 문제로 일시적 에러가 발생할 수 있습니다.
