from opensearchpy import OpenSearch, RequestsHttpConnection
import os
import json

class OpenSearchEmbeddingStore:
    def __init__(self, host='localhost', port=9200, id='admin', password='juntPass123!'):
        self.client = OpenSearch(
            hosts=[{'host': host, 'port': port}],
            http_auth=(id, password),  # 기본 자격증명 (필요에 따라 수정)
            use_ssl=True,
            verify_certs=False,
            ssl_show_warn=False,
            connection_class=RequestsHttpConnection
        )

    def create_index(self, index_name='doc_embedding', recreate=False):
        self.index_name = index_name

        try:
            # 인덱스 설정
            settings = {
                "settings": {
                    "index": {
                        "knn": True,
                        "knn.algo_param.ef_search": 100
                    }
                },
                "mappings": {
                    "properties": {
                        "content": {"type": "text"},
                        "source_path": {"type": "keyword"},
                         "embedding_vector": {
                            "type": "knn_vector",
                            "dimension": 768,
                            "method": {
                                "name": "hnsw",
                                "space_type": "cosinesimil",
                                "engine": "nmslib"
                            }
                        }
                    }
                }
            }

            # 인덱스 삭제하고 생성
            if recreate == True:
                if self.client.indices.exists(self.index_name):
                    self.client.indices.delete(self.index_name)
                response = self.client.indices.create(
                    index=self.index_name,
                    body=settings
                )
                print(f"Index created: {response}")
                return 0
            else:
                return 0
        except Exception as e:
            print(f"Error creating index: {e}")
            return -1

    def get_embedding(self, text):
        embedding_vct = self.embedding_model.encode(text)
        return embedding_vct

    def drop_doc(self, sourcefile):
        query = {
            "query": {
                "term": {
                    "source_path": sourcefile
                }
            }
        }
        response = self.client.delete_by_query(
            index=self.index_name,
            body=query
        )

        print(f"삭제된 문서 수: {response['deleted']}")
        return response['deleted']


    def store_embedding(self, sourcefile, text, metadata=None):
        embedding = self.get_embedding(text)
        document = {
            'content': text,
            'embedding_vector': embedding,
            'source_path': sourcefile,
            'meta': metadata
        }
        response = self.client.index(
            index=self.index_name,
            body=document
        )
        return response

    def store_doc(self, index: str, document: dict):
        response = self.client.index(
            index=self.index,
            body=document
        )
        return response

    def search_similar(self, query_text, top_k=5):
        """유사한 텍스트 검색"""
        query_embedding = self.get_embedding(query_text)
        try:
            query = {
                "query": {
                    "knn": {
                        "embedding_vector": {
                            "vector": query_embedding,
                            "k": top_k
                        }
                    }
                }
            }
            response = self.client.search(
                index=self.index_name,
                body=query
            )
            return response['hits']['hits']

        except Exception as e:
            print(f"Error during vector search: {e}")
        return response['hits']['hits']

