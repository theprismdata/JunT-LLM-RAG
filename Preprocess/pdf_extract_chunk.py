import os
import pathlib
import pdfplumber
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd

def is_inside_any_table(word_bbox, tables):
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

def extract_content(pdf_path):
    result = {
        "doc_meta": [], 
    }
    print(os.path.abspath(__file__))
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            # 페이지의 모든 요소를 담을 리스트
            page_elements = []
            
            # 테이블 찾기
            tables = page.find_tables()
            
            # 테이블 처리
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
            
            # 텍스트 추출 (테이블 영역 제외)
            words = page.extract_words(keep_blank_chars=True, x_tolerance=3, y_tolerance=3)
            
            # 테이블 영역 외의 단어들만 모으기
            current_line = []
            current_line_number = 1
            current_y = None
            
            for word in words:
                word_bbox = (word['x0'], word['top'], word['x1'], word['bottom'])
                
                # 테이블 내부의 텍스트는 건너뛰기
                if is_inside_any_table(word_bbox, tables):
                    continue
                
                word_y_center = (word['top'] + word['bottom']) / 2
                
                if current_y is None:
                    current_y = word_y_center
                
                # y 위치가 비슷하면 같은 라인으로 간주
                if abs(word_y_center - current_y) < 5:
                    current_line.append(word['text'])
                else:
                    # 새로운 라인 시작
                    if current_line:
                        line_text = " ".join(current_line)
                        if line_text.strip():  # 빈 라인 제외
                            page_elements.append({
                                'type': 'text',
                                'context': f"{line_text}",
                                'y_center': current_y,
                                'page': page_num,
                                'line': current_line_number
                            })
                            current_line_number += 1
                    current_line = [word['text']]
                    current_y = word_y_center
            
            # 마지막 라인 처리
            if current_line:
                line_text = " ".join(current_line)
                if line_text.strip():
                    page_elements.append({
                        'type': 'text',
                        'context': f"{line_text}",
                        'y_center': current_y,
                        'page': page_num,
                        'line': current_line_number
                    })
            
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
                    
                    page_number = current_text_elements[0]['page']
                    start_ln = current_text_elements[0]['line']
                    end_ln = current_text_elements[-1]['line']
                    
                    fist_end_page_line_num = {'start_line': start_ln,'end_line': end_ln}
                    # 청크 크기가 되면 분할
                    # if len(current_text) >= 100:
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=50,
                        chunk_overlap=4,
                        length_function=len,
                        # separators=["\n\n", "\n", ".", " ", ""]
                        separators=[""]
                    )
                    chunks = text_splitter.split_text(current_text)
                    
                    for chunk in chunks:
                        print("\n", chunk)
                        result["doc_meta"].append({
                            "type": "text",
                            "page": page_num,
                            "context": chunk,
                            "line_pos": fist_end_page_line_num
                        })
                    
                    current_text = ""
                    current_text_elements = []
                else:  # 테이블인 경우
                    # 남은 텍스트가 있으면 먼저 처리
                    if current_text:
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1000,
                            chunk_overlap=200,
                            length_function=len,
                            separators=["\n\n", "\n", ".", " ", ""]
                        )
                        chunks = text_splitter.split_text(current_text)
                        
                        for chunk in chunks:
                            result["doc_meta"].append({
                                "type": "text",
                                "page": page_num,
                                "context": chunk,
                                "line_pos": fist_end_page_line_num
                            })
                        
                        current_text = ""
                        current_text_elements = []
                    
                    # 테이블 추가
                    result["doc_meta"].append({
                        "type": "table",
                        "page": element['page'],
                        "table_number": element['table_number'],
                        "context": element['context']
                    })
            
            # 페이지의 마지막 텍스트 처리
            if current_text:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len,
                    separators=["\n\n", "\n", ".", " ", ""]
                )
                chunks = text_splitter.split_text(current_text)
                # page_number = current_text_elements[0]['page']
                start_ln = current_text_elements[0]['line']
                end_ln = current_text_elements[-1]['line']
            
                fist_end_page_line_num = {'start_line': start_ln, 'end_line': end_ln}
                for chunk in chunks:
                    result["context"].append({
                        "type": "text",
                        "page": page_num,
                        "context": chunk,
                        "line_pos": fist_end_page_line_num
                    })
    
    return result

def save_to_json(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# 사용 예시
if __name__ == "__main__":
    # pdf_path = os.path.join(*["..", "source_doc", "[KIPA 한국행정연구원] (이슈페이퍼)", "2023-12-[KIPA 한국행정연구원] (이슈페이퍼) (2023-134호)디지털트윈을 활용한 과학적·예측적 정책의사결정 활성화 방안(조세현+차남준+나보리+우하린+김상숙).pdf"])
    # print(pdf_path)
    # print(os.path.exists(pdf_path))
    files_meta = []
    for filepath in pathlib.Path("../source_doc/").rglob('*.pdf'):
        print(filepath)
        if filepath.is_file():
            doc_meta = extract_content(filepath)
            doc_meta = doc_meta['doc_meta']
            files_meta.append({"origin_path": str(filepath),
                                "doc_meta": doc_meta})

    output_path = "../metadump_chunk.json"  # 결과 저장할 JSON 파일 경로
    save_to_json(files_meta, output_path)