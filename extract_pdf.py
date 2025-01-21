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


def get_contexttable_pdffile_by_plumber(self, source_file_name: str) -> list:
    result = {
        "doc_meta": [],
    }

    with pdfplumber.open(source_file_name) as pdf:
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
                    start_ln = current_text_elements[0]['line']
                    fist_end_page_line_num = {'start_line': start_ln}
                    result["doc_meta"].append({
                        "type": "text",
                        "page": page_num,
                        "context": current_text,
                        "line_pos": fist_end_page_line_num
                    })
                    current_text = ""
                    current_text_elements = []
                else:  # 테이블인 경우
                    # 남은 텍스트가 있으면 먼저 처리
                    if current_text:
                        result["doc_meta"].append({
                            "type": "text",
                            "page": page_num,
                            "context": current_text,
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
                start_ln = current_text_elements[0]['line']
                fist_end_page_line_num = {'start_line': start_ln}
                result["context"].append({
                    "type": "text",
                    "page": page_num,
                    "context": current_text,
                    "line_pos": fist_end_page_line_num
                })
    return None, result

def save_to_json(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# 사용 예시
if __name__ == "__main__":
    files_meta = []
    for filepath in pathlib.Path("source_doc/").rglob('*.pdf'):
        print(filepath)
        if filepath.is_file():
            doc_meta = get_contexttable_pdffile_by_plumber(filepath)
            doc_meta = doc_meta['doc_meta']
            files_meta.append({"origin_path": str(filepath),
                                "doc_meta": doc_meta})

    output_path = "metadump.json"  # 결과 저장할 JSON 파일 경로
    save_to_json(files_meta, output_path)