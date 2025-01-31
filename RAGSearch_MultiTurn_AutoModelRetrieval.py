import argparse
import os
import sys
import torch
from typing import Iterator

from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from sentence_transformers import SentenceTransformer
from vectorstore.opensearchengine import OpenSearchEmbeddingStore
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
# Redis 설정
session_id = "aicode_digger_chat1"
REDIS_URL = "redis://localhost:6379"
chat_history = RedisChatMessageHistory(session_id=session_id, url=REDIS_URL)
chat_history.clear()

# Model ID 설정
model_id = "google/gemma-2b-it"
model_path = f"./HF_Models/{model_id}"

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
    if not os.path.exists(f'./embeddingmodel/{model_name}'):
        hugging_cmd = f'huggingface-cli download {model_name} --local-dir ./embeddingmodel/{model_name}'
        os.system(hugging_cmd)

    embedding_model = HuggingFaceEmbeddings(
        model_name=f'./embeddingmodel/{model_name}/',
        model_kwargs={'device': 'cpu'}
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=model_path)
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.9,
            device_map="auto"
        )
        llm = HuggingFacePipeline(pipeline=pipe)

    except Exception as e:
        print(f"Error loading HuggingFace model: {e}")
        sys.exit(1)

    return embedding_model, llm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-query', help='질문을 해주세요')
    args = parser.parse_args()

    # 모델 초기화
    embedding_model, llm = initialize_models()
    print(f'질문 : ', args.query)
    query = args.query

    # Langchain Wrapping
    store = OpenSearchVectorSearch(index_name="doc_embedding",
                                   embedding_function=embedding_model,
                                   opensearch_url="http://localhost:9200",
                                   verify_certs=False,
                                   ssl_show_warn=False,
                                   use_ssl=True,
                                   text_field="text",
                                   http_auth=('admin', 'juntPass123!'))

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
    chain = prompt | llm | StrOutputParser()
    chain_with_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: chat_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    # 검색 방방법1
    # docs = store.similarity_search(query, k=4, search_type="approximate_search",
    #                                vector_field="embedding_vector")
    # 검색 방법 2 : 다른 함수와 연동시 사용.
    # try:
    #     retriever = store.as_retriever(search_kwargs={'score_threshold': 0.3},
    #                                    search_type="similarity_score_threshold")
    #     results = retriever.invoke(query, vector_field="embedding_vector")
    #     print(results)
    # except Exception as e:
    #     print(e)
    try:
        retriever = store.as_retriever(search_kwargs={'k': 3},
                                       search_type="similarity")
        result_docs = retriever.invoke(query, vector_field="embedding_vector")
        print(result_docs)
    except Exception as e:
        print(e)


    if result_docs is not None:
        search_text = ''
        for docs in result_docs:
            print(docs.metadata['source_path'])
            print(docs.metadata['text'])
            print(docs.metadata['meta'])
            search_text += docs.metadata['text']
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

if __name__ == '__main__':
    main()