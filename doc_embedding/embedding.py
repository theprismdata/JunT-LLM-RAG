import json
import os
import pathlib
import sys
from sentence_transformers import SentenceTransformer
from vectorstore.opensearchengine import OpenSearchEmbeddingStore
from langchain.text_splitter import RecursiveCharacterTextSplitter

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

if os.path.exists('../embeddingmodel/KR-SBERT-V40K-klueNLI-augSTS') == False:
    hugging_cmd = 'huggingface-cli download snunlp/KR-SBERT-V40K-klueNLI-augSTS --local-dir ./embeddingmodel/KR-SBERT-V40K-klueNLI-augSTS/'
    os.system(hugging_cmd)
embedding_model = SentenceTransformer('../embeddingmodel/KR-SBERT-V40K-klueNLI-augSTS/')


if __name__ == "__main__":

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=50,
        chunk_overlap=4,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    store = OpenSearchEmbeddingStore(id='admin', password='juntPass123!')

    index_name = "doc_embedding"
    store.create_index(index_name=index_name, reindex=True)
    meta_path = "../meta_dumps"

    for input_file in pathlib.Path(meta_path).rglob("*.json"):
        with open(input_file, "r", encoding="utf-8") as file:
            file_info = json.load(file)
            print(file_info['origin_path'])
            sourcefile = file_info['origin_path']
            doc_meta = file_info['doc_meta']
            for meta_info in doc_meta:
                meta_type = meta_info['type']
                pn = meta_info['page']
                print(f"Page {pn}")
                if meta_type == 'text':
                    start_line = meta_info['line_pos']['start_line']
                    # end_line = meta_info['line_pos']['end_line']
                    context =  meta_info['context']
                    chunks = text_splitter.split_text(context)
                    print(f"Chunk Len {len(chunks)}")
                    for chunk in chunks:
                        print("\n", chunk)
                        embedding_vector = embedding_model.encode(chunk)
                        document = {
                            'text': context,
                            'source_path': sourcefile,
                            'meta': {
                                    "page_number": pn,
                                    "start_line": start_line,
                                    # "end_line": end_line
                                    },
                            'embedding_vector': embedding_vector
                        }
                elif  meta_type == 'table':
                    context = meta_info['context']
                    embedding_vector = embedding_model.encode(context)
                    document = {
                        'text': context,
                        'source_path': sourcefile,
                        'meta': {
                            "page_number": pn,
                            "start_line": start_line,
                            # "end_line": end_line
                        },
                        'embedding_vector': embedding_vector
                    }
                store.store_doc(document)

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

