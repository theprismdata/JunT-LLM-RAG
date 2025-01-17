import argparse
import sys

from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from typing import Iterator
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnableWithMessageHistory
from opensearchengine import OpenSearchEmbeddingStore

session_id = "user_123"
REDIS_URL = "redis://localhost:6379"
redischathistory = RedisChatMessageHistory(session_id=session_id, url=REDIS_URL)

# 과거 대화 로드.
history = []
if len(redischathistory.messages) > 0:
    for session_time in redischathistory.messages:
        print(f"Type : {session_time.type} Messge : {session_time.content}")
        history.append({session_time.type:session_time.content})

store = OpenSearchEmbeddingStore()



def process_stream_output(stream_output: Iterator[str]) -> None:
    """Stream output processor"""
    try:
        for chunk in stream_output:
            print(chunk, end='', flush=True)
        print()  # New line after streaming completes
    except Exception as e:
        print(f"\nError in streaming: {e}")

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('-query', help='질문을 해주세요')
    args = parser.parse_args()
    store = OpenSearchEmbeddingStore()

    print(f'질문 : ', args.query)
    query = args.query

    results = store.search_similar(query)
    search_text = ''
    for hit in results:
        search_text += hit['_source']['content']

    print("Call LLM")
    system_message = """너는 보고서를 제공하는 챗봇이야. \n나의 질문에 대해 Relevant Information을 이용하여 보고서를 작성해줘. \n 답변은 한국어로 해야되"""

    # Create streaming-enabled prompt chain
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "{question}"),
        ("assistant", "Relevant Information: {related_info}\n")
    ])

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

    # Create and execute streaming chain
    chain = prompt | llm | StrOutputParser()
    chain_with_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: redischathistory,
        input_messages_key="question",
        history_messages_key="history",
    )

    print("\n질의 결과:")
    config = {"configurable": {"session_id": session_id}}
    response = chain_with_history.invoke({"question": query, "related_info": search_text}, config = config)
    print(response)
    # process_stream_output(chain.stream({
    #     "user_input": query,
    #     "related_info": search_text
    # }))

    print("\n****세부 자료****")
    for hit in results[:3]:
        print(f"텍스트: {hit['_source']['content']}")
        print(f"원본 파일: {hit['_source']['meta']['source']}")
        print(f"위치 페이지: {hit['_source']['meta']['page']}")
        print(f"위치 줄번호: {hit['_source']['meta']['ln']}")
        print(f"점수: {hit['_score']}")