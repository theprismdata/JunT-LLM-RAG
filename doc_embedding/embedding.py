import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from opensearchengine import OpenSearchEmbeddingStore


if __name__ == "__main__":
    store = OpenSearchEmbeddingStore()

    # 인덱스 생성
    store.create_index(recreate=True)

    # Preprocess에서 추출된 정보를 vectordb에 저장.
    # 파일명이 이미 존재한다면 해당 파일에 대한 정보를 삭제하고 합니다.
    store.embedding_doc(file_path="../metadump.json")

    # 데이터를 검색합니다.
    query = "AI와 신경망에 대해 알려주세요"
    results = store.search_similar(query)

    print("\n검색 결과:")
    for hit in results:
        print(f"텍스트: {hit['_source']['content']}")
        print(f"원본 파일: {hit['_source']['source_path']}")
        print(f"위치 페이지: {hit['_source']['meta']['page']}")
        print(f"위치 줄번호: {hit['_source']['meta']['ln']}")
        print(f"점수: {hit['_score']}")
