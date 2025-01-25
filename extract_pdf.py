import argparse
import os
import pathlib
import pprint
import re
import pdfplumber
import json
# from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
import nltk
from langchain.text_splitter import NLTKTextSplitter

def detect_dot_pattern(text):
        pattern = r'[·]+'
        matches = re.findall(pattern, text)
        return len(matches) > 0
    
def specialchar_replace_patttern(text):
        pattern = r'[,•]+'
        replace = ' '
        text = re.sub(pattern, replace, text)
        return text
    
class TextExtract:
    """
    Document extration main class
    """
    def __init__(self, bucket_name=None):
        self.del_table = True
        self.cvt_image = False
        self.preprocess_meta = []

    def is_inside_any_table(self, word_bbox, tables):
        """단어가 테이블 내부에 있는지 확인"""
        word_x0, word_y0, word_x1, word_y1 = word_bbox
        word_center_y = (word_y0 + word_y1) / 2
        word_center_x = (word_x0 + word_x1) / 2

        for table in tables:
            table_x0, table_y0, table_x1, table_y1 = table.bbox
            if (table_x0 <= word_center_x <= table_x1 and
                    table_y0 <= word_center_y <= table_y1):
                return True
        return False

    def detect_dot_pattern(self, text):
        pattern = r'[\w\s]+\.{3,}\s*\d+'
        matches = re.findall(pattern, text)
        return len(matches) > 0

    def get_contexttable_pdffile_by_plumber(self, source_file_name: str) -> list:
        result = {
            "doc_meta": [],
        }
        index_keywords = ['목차', '차례', 'contents', 'index', '그림목차']
        temp_all_contents = "" #전체 콘텐츠 일관성 확인용
        with pdfplumber.open(source_file_name) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_elements = []
                try:
                    tables = page.find_tables()

                    for table_num, table in enumerate(tables, 1):
                        df = pd.DataFrame(table.extract())
                        if len(df) > 0:  # 빈 테이블 제외
                            df.columns = df.iloc[0]
                            df = df.iloc[1:]
                            markdown_table = df.to_markdown(index=False)

                            # 테이블의 중심 y 좌표 계산
                            y_center = (table.bbox[1] + table.bbox[3]) / 2

                            page_elements.append({
                                'type': 'table',
                                'context': markdown_table,
                                'y_center': y_center,
                                'page': page_num,
                                'table_number': table_num,
                                'bbox': table.bbox
                            })
                except Exception as e:
                    print(str(e))
                    continue
                 
                # 텍스트 추출 (테이블 영역 제외)
                words = page.extract_words(keep_blank_chars=True, x_tolerance=3, y_tolerance=3)

                # 테이블 영역 외의 단어들만 모으기
                current_line = []
                current_line_number = 1
                current_y = None

                #index page detection(try....)
                bIndex_page = False
                for word in words:
                    target_word = word['text'].strip()
                    if target_word in index_keywords:
                        bIndex_page = True
                        break
                    if detect_dot_pattern(target_word):
                        bIndex_page = True
                        break
                if bIndex_page == True:
                    continue
                
                for word in words:
                    word_bbox = (word['x0'], word['top'], word['x1'], word['bottom'])
                    if self.is_inside_any_table(word_bbox, tables):
                        continue
                    
                    word_y_center = (word['top'] + word['bottom']) / 2
                    if current_y is None:
                        current_y = word_y_center

                    if abs(word_y_center - current_y) < 5:
                        current_line.append(word['text'])
                    else:
                        # 새로운 라인 시작
                        if current_line:
                            line_text = " ".join(current_line)                            
                            if line_text.strip():  # 빈 라인 제외
                                
                                page_elements.append({
                                    'type': 'text',
                                    'context': line_text,
                                    'y_center': current_y,
                                    'page': page_num,
                                    'line': current_line_number
                                })
                                temp_all_contents += line_text +'\n'
                                current_line_number += 1
                        current_line = [word['text']]
                        current_y = word_y_center
                    
                # 마지막 라인 처리
                if current_line:
                    line_text = " ".join(current_line)
                    if line_text.strip():
                        page_elements.append({
                            'type': 'text',
                            'context': line_text,
                            'y_center': current_y,
                            'page': page_num,
                            'line': current_line_number
                        })
                        temp_all_contents += line_text +'\n'
                
                # y 위치에 따라 요소들을 정렬
                page_elements.sort(key=lambda x: x['y_center'])

                # 텍스트 요소들을 청크로 분할하며 결과에 추가
                current_text = ""
                current_text_elements = []

                for element in page_elements:
                    if element['type'] == 'text':
                        current_text += element['context'] + "\n"
                        current_text_elements.append({
                            'page': element['page'],
                            'line': element['line']
                        })
                        start_ln = current_text_elements[0]['line']
                        fist_end_page_line_num = {'start_line': start_ln}
                        result["doc_meta"].append({
                            "type": "text",
                            "page": page_num,
                            "context": current_text,
                            "line_pos": fist_end_page_line_num
                        })
                        temp_all_contents += current_text+'\n'
                        current_text = ""
                        current_text_elements = []
                    else:  # 테이블인 경우
                        if current_text:
                            result["doc_meta"].append({
                                "type": "text",
                                "page": page_num,
                                "context": current_text,
                                "line_pos": fist_end_page_line_num
                            })
                            temp_all_contents += current_text
                            current_text = ""
                            current_text_elements = []

                        # 테이블 추가
                        result["doc_meta"].append({
                            "type": "table",
                            "page": element['page'],
                            "table_number": element['table_number'],
                            "context": element['context']
                        })
                        temp_all_contents += element['context']+'\n'

                # 페이지의 마지막 텍스트 처리
                if current_text:
                    start_ln = current_text_elements[0]['line']
                    fist_end_page_line_num = {'start_line': start_ln}
                    result["context"].append({
                        "type": "text",
                        "page": page_num,
                        "context": current_text,
                        "line_pos": fist_end_page_line_num
                    })
                    temp_all_contents += current_text +'\n'
                    
            nltk_text_splitter = NLTKTextSplitter(
                                chunk_size=200,
                                length_function=len
                            )
            chunks = nltk_text_splitter.split_text(temp_all_contents)
            print("총 청크 수:", len(chunks))
        return None, result, chunks

    def extract_from_file(self, idx: int, filepath: str):
        with open(filepath, 'rb') as f:
            header = f.read(4)
            if header[:4] == b'%PDF':
                _, infor_meta, chunks = self.get_contexttable_pdffile_by_plumber(filepath)
                
                if len(chunks) < 2:
                    infor_meta = None
                    
                if infor_meta is not None:
                    preprocess_meta = {'origin_path':filepath,
                                        'doc_meta':infor_meta['doc_meta'],
                                        'lenchunk': len(chunks)}
                else:
                    preprocess_meta = None
            
                with open(f'meta_dumps/{idx}_metadump.json', "w", encoding='utf-8') as fp:
                    if preprocess_meta is not None:
                        fp.write(json.dumps(preprocess_meta, ensure_ascii=False,indent=2))
                    else:
                        fp.write(json.dumps({"origin_path": filepath,
                                                "status":"Error"},
                                                ensure_ascii=False))
            else:
                print("No pdf format")
            
                            
    def extract_from_folder(self, folder_path: str, ext: str):
        """
        폴더 경로에서 ext에 설정된 확장자를 가진 파일 목록을 가져옴
        텍스트를 추출하여 json파일로 저장합니다.
        :return:
        """
        src_meta_path = 'meta_dumps'
        if os.path.exists(src_meta_path) == False:
            os.makedirs(src_meta_path, exist_ok=True)

        origin_file_list = []
        for metafile in pathlib.Path(src_meta_path).rglob("*.json"):
            with open(metafile, "r", encoding="utf-8") as file:
                print(metafile, "check")
                file_info = json.load(file)
                orgin_file = file_info['origin_path']
                origin_file_list.append(orgin_file)
                
        idx = 0
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(ext):
                    filepath = os.path.abspath(os.path.join(root, file))
                    print(f"In process {filepath}")
                    
                    with open(filepath, 'rb') as f:
                        header = f.read(4)
                        if header[:4] == b'%PDF':
                            self.extract_from_file(idx=idx, filepath=filepath)
                            
                            # if len(chunks) < 2:
                            #     infor_meta = None
                                
                            # if infor_meta is not None:
                            #     preprocess_meta = {'origin_path':filepath,
                            #                         'doc_meta':infor_meta['doc_meta'],
                            #                         'lenchunk': len(chunks)}
                            # else:
                            #     preprocess_meta = None
                        
                            # with open(f'meta_dumps/{idx}_metadump.json', "w", encoding='utf-8') as fp:
                            #     if preprocess_meta is not None:
                            #         fp.write(json.dumps(preprocess_meta, ensure_ascii=False,indent=2))
                            #     else:
                            #         fp.write(json.dumps({"origin_path": filepath,
                            #                                 "status":"Error"},
                            #                                 ensure_ascii=False))
                            # idx += 1
                        else:
                            print("No pdf format")
                            
    def do_aggregate_folder(self, folder_path: str):
        for metafile in pathlib.Path(folder_path).rglob("*.json"):
            self.do_aggregate_file(metafile)
    
    def do_aggregate_file(self, metafile: str):
        pages_contents = {}
        
        with open(metafile, "r", encoding="utf-8") as fp:
            print(metafile, "check")
            file_info = json.load(fp)
            print(file_info['origin_path'])
            if 'status' in list(file_info.keys()):
                return
            cur_page_contents = ''
            
            pages_contents['filename'] = os.path.splitext(os.path.basename(file_info['origin_path']))[0]
                        
            for meta_infos in file_info['doc_meta']:
                print(f"Page {meta_infos['page']}")
                if meta_infos['type'] == 'text':
                    print(f"Page:{meta_infos['page']}")
                                            
                    if detect_dot_pattern(meta_infos['context']):
                        continue                    
                    if "표 " in meta_infos['context']:
                        continue
                    if "그림 " in meta_infos['context']:
                        continue
                    cur_page_contents = meta_infos['context']
                                        
                    pattern = r'(?:첫째|둘째|셋째|넷째|다섯째|여섯째|일곱째|여덟째|아홉째|열째),\s*'
                    cur_page_contents = re.sub(pattern, '', cur_page_contents)

                if meta_infos['page'] in pages_contents.keys():
                    pages_contents[meta_infos['page']] += cur_page_contents
                    
                    #전처리: 미완 단어위 \n을 제거함.
                    pattern = r'([가-힣a-zA-Z0-9\s]+)\n([가-힣a-zA-Z0-9]+)'
                    replacement = r'\1\2'
                    pages_contents[meta_infos['page']] = re.sub(pattern, replacement, pages_contents[meta_infos['page']])
                    print(f"{pages_contents[meta_infos['page']]}")
                else:
                    pages_contents[meta_infos['page']] = cur_page_contents
                cur_page_contents = ''
                
        
        agg_dump_filename = os.path.splitext(os.path.basename(metafile))[0] 
        with open(f"meta_aggregate/{agg_dump_filename}.json", "w", encoding='utf-8') as fp:
            fp.write(json.dumps(pages_contents, ensure_ascii=False,indent=2))
                                

            
        
parser = argparse.ArgumentParser()
parser.add_argument('-input', help='추출할 파일이 있는 경로를 넣어주세요')
args = parser.parse_args()
if __name__ == '__main__' :
    print(f'추출 대상 경로: ', args.input)
    te = TextExtract()
    extensions = (".pdf")
    te.extract_from_folder(folder_path=args.input, ext=extensions)
        
    # te.extract_from_file(filepath=file_name, ext=extensions)
    te.do_aggregate_file(metafile="./meta_dumps/0_metadump.json")
    te.do_aggregate_folder(folder_path="./meta_dumps")