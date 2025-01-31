import json
import os
import pathlib
import sys

from sentence_transformers import SentenceTransformer
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from vectorstore.opensearchengine import OpenSearchEmbeddingStore
from langchain.text_splitter import RecursiveCharacterTextSplitter


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
    store.create_index(index_name=index_name, reindex=False)

    rlist = []
    rlog_fp = open("write.log.txt", 'r')
    while True:
        rfilename = rlog_fp.readline()
        if not rfilename: break
        rfilename = rfilename.strip()
        if len(rfilename) == 0: continue
        rlist.append(rfilename)
    rlog_fp.close()
    print("LOAD HISTORY")
    print(f"{len(rlist)} files")

    meta_path = "../meta_dumps"
    for input_file in pathlib.Path(meta_path).rglob("*.json"):
        if str(input_file) in rlist:
            continue
        wlog_fp = open("write.log.txt", 'a')
        with open(input_file, "r", encoding="utf-8") as fp:
            file_info = json.load(fp)
            # print(file_info['origin_path'])
            sourcefile = file_info['origin_path']
            if 'doc_meta' not in file_info:
                print(f"doc meta missing {input_file}")
                continue
            doc_meta = file_info['doc_meta']
            for meta_info in doc_meta:
                meta_type = meta_info['type']
                pn = meta_info['page']
                # print(f"Page {pn}")
                if meta_type == 'text':
                    start_line = meta_info['line_pos']['start_line']
                    context =  meta_info['context']
                    chunks = text_splitter.split_text(context)
                    for chunk in chunks:
                        # print("\n", chunk)
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
                        },
                        'embedding_vector': embedding_vector
                    }
                try:
                    if store.check_duplicate(sourcefile=sourcefile, text=context) == False:
                        store.store_doc(document)
                except Exception as e:
                    print(str(e))
        wlog_fp.write(f"{input_file}\n")
        wlog_fp.close()

