import argparse
import os
import sys

from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from typing import Iterator
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnableWithMessageHistory
from sentence_transformers import SentenceTransformer
from vectorstore.opensearchengine import OpenSearchEmbeddingStore

# Redis 설정
session_id = "user_2"
REDIS_URL = "redis://localhost:6379"
chat_history = RedisChatMessageHistory(session_id=session_id, url=REDIS_URL)

def process_stream_output(stream_output: Iterator[str]) -> None:
    """Stream output processor"""
    try:
        for chunk in stream_output:
            print(chunk, end='', flush=True)
        print()  # New line after streaming completes
    except Exception as e:
        print(f"\nError in streaming: {e}")

def initialize_models():
    """모델과 프롬프트 초기화"""
    # 임베딩 모델 설정
    model_name = 'KR-SBERT-V40K-klueNLI-augSTS'
    # model_name = 'BAAI/bge-m3'
    # if not os.path.exists('./embeddingmodel/KR-SBERT-V40K-klueNLI-augSTS'):
    #     hugging_cmd = 'huggingface-cli download snunlp/KR-SBERT-V40K-klueNLI-augSTS --local-dir ./embeddingmodel/KR-SBERT-V40K-klueNLI-augSTS/'
    #     os.system(hugging_cmd)

    if not os.path.exists(f'./embeddingmodel/{model_name}'):
        hugging_cmd = f'huggingface-cli download {model_name} --local-dir ./embeddingmodel/{model_name}'
        os.system(hugging_cmd)

    embedding_model = SentenceTransformer(f'./embeddingmodel/{model_name}/')

    # LLM 설정
    try:
        llm = Ollama(
            model="gemma2:2b",
            base_url="http://localhost:11434",
            callbacks=[],
            temperature=0.7,
        )
    except Exception as e:
        print(f"Error connecting to Ollama server: {e}")
        print("Please make sure the Ollama server is running (ollama serve)")
        sys.exit(1)

    # 프롬프트 템플릿 설정 - MessagesPlaceholder 사용
    system_message = """너는 보고서를 제공하는 챗봇이야.
나의 질문에 대해 Relevant Information을 이용하여 보고서를 작성해줘.
이전 대화 내용을 잘 참고해서 일관성 있게 답변해줘.
답변은 한국어로 해야 돼."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
        ("system", "Relevant Information: {context}")
    ])

    # Chain 설정
    chain = prompt | llm | StrOutputParser()
    chain_with_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: chat_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    return embedding_model, chain_with_history

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-query', help='질문을 해주세요')
    args = parser.parse_args()

    # 벡터 스토어 초기화
    store = OpenSearchEmbeddingStore()

    # 모델 초기화
    embedding_model, chain_with_history = initialize_models()

    print(f'질문 : ', args.query)
    query = args.query

    # 임베딩 및 검색
    embedding_vector = embedding_model.encode(query)
    results = store.search_similar(index_name="doc_embedding", vector=embedding_vector)

    if results is not None:
        search_text = ''
        for hit in results:
            search_text += hit['_source']['text']
        print("\n질의 결과:")
        config = {"configurable": {"session_id": session_id}}
        response = chain_with_history.invoke(
            {
                "input": query,
                "context": search_text
            },
            config=config
        )
        print(response)

        # 디버깅용 대화 히스토리 출력
        # print("\n현재 대화 히스토리:")
        # for msg in chat_history.messages:
        #     print(f"Type: {msg.type}, Content: {msg.content}")

if __name__ == '__main__':
    main()