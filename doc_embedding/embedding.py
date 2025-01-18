import json
import os
import sys
from sentence_transformers import SentenceTransformer
from opensearchengine import OpenSearchEmbeddingStore

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

if os.path.exists('./embeddingmodel/KR-SBERT-V40K-klueNLI-augSTS') == False:
    hugging_cmd = 'huggingface-cli download snunlp/KR-SBERT-V40K-klueNLI-augSTS --local-dir ./embeddingmodel/KR-SBERT-V40K-klueNLI-augSTS/'
    os.system(hugging_cmd)
embedding_model = SentenceTransformer('./embeddingmodel/KR-SBERT-V40K-klueNLI-augSTS/')

if __name__ == "__main__":
    # store = OpenSearchEmbeddingStore(id='admin', password='juntPass123!')
    #
    # # 인덱스 생성
    # store.create_index(index_name='doc_embedding', recreate=True)

    with open("../metadump.json", "r", encoding="utf-8") as file:
        file_info_list = json.load(file)
        for file_info in file_info_list:
            print(file_info['origin_path'])
            doc_meta = file_info['doc_meta']
            for meta_info in doc_meta:
                meta_type = meta_info['type']
                pn = meta_info['page']
                if meta_type == 'text':
                    start_line = meta_info['line_pos']['start_line']
                    end_line = meta_info['line_pos']['end_line']
                    context =  meta_info['context']
                    print(context)
                elif  meta_type == 'table':
                    context = meta_info['context']
                    print(context)

    # # 데이터를 검색합니다.
    # query = "AI와 신경망에 대해 알려주세요"
    # results = store.search_similar(query)
    #
    # print("\n검색 결과:")
    # for hit in results:
    #     print(f"텍스트: {hit['_source']['content']}")
    #     print(f"원본 파일: {hit['_source']['source_path']}")
    #     print(f"위치 페이지: {hit['_source']['meta']['page']}")
    #     print(f"위치 줄번호: {hit['_source']['meta']['ln']}")
    #     print(f"점수: {hit['_score']}")

