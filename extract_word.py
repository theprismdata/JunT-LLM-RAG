import json
import pathlib
import pprint

from docx import Document
import os


def get_msword_content(file_path):
    """
    MS Word 문서에서 텍스트를 추출하고 페이지 번호와 줄 번호를 포함하여 반환합니다.

    Args:
        file_path (str): Word 문서의 경로

    Returns:
        list: 각 줄의 정보를 담은 딕셔너리 리스트
            - page_number: 페이지 번호
            - line_number: 줄 번호
            - content: 텍스트 내용
    """
    try:
        doc = Document(file_path)
    except Exception as e:
        raise Exception(f"문서를 열 수 없습니다: {str(e)}")
    current_page = 1
    line_number = 1
    doc_meta = []
    whole_page_extractinfo = ''
    for paragraph in doc.paragraphs:
        # 페이지 나누기 확인
        if paragraph._element.xpath('.//w:lastRenderedPageBreak'):
            current_page += 1
        line_meta = []
        text_list = paragraph.text.split("\n")
        if len(text_list) > 0:  # 빈 줄 제외
            for text in text_list:
                whole_page_extractinfo += text + '\n'
                line_meta.append({"ln": line_number, "content": text})
                line_number += 1
        doc_meta.append({"page": current_page,
                            "line_meta": line_meta})
    return whole_page_extractinfo, doc_meta


def save_to_text(extracted_content, output_file):
    """
    추출된 내용을 텍스트 파일로 저장합니다.

    Args:
        extracted_content (list): extract_word_content 함수의 결과
        output_file (str): 출력 파일 경로
    """
    with open(output_file, "w", encoding='utf-8') as fp:
        fp.write(json.dumps(extracted_content, ensure_ascii=False))


if __name__ == "__main__":
    # 사용 예시
    output_file = './metadump.json'

    try:
        files_meta = []
        for filepath in pathlib.Path("source_doc/").rglob('*.docx'):
            print(filepath)
            if filepath.is_file():
                formed_clear_contents, doc_meta  = get_msword_content(filepath)
                files_meta.append({"origin_path": str(filepath),
                                   "doc_meta": doc_meta})

        save_to_text(files_meta, output_file)
        print(f"문서 내용이 성공적으로 추출되어 {output_file}에 저장되었습니다.")

    except Exception as e:
        print(f"오류 발생: {str(e)}")