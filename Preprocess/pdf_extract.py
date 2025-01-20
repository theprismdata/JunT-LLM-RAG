import json
import pathlib
import pprint
import pdfplumber
import os
import re
import pandas as pd

def get_context_pdffile_by_plumber(file_path, del_table=True):
    """
    get context from pdf file
    :param source_file_name: source pdf file path
    :return: extracted context text
    """
    print(f"Source path {file_path}")
    try:
        pdfplumb = pdfplumber.open(file_path)
        whole_page_extractinfo = ""
        page_exist_tbl = False
    except IOError as e:
        print(f"I/O error({e.errno}): {e.strerror}")
        return None

    isSkeep = False
    cvt_image = False
    doc_meta = []

    for page_num, _ in enumerate(pdfplumb.pages):
        page_plumb_contents = {}
        table_list = []
        last_line = 0
        if cvt_image == True:
            pil_img = pdfplumb.pages[page_num].to_image(resolution=1200)
            pil_img.save(f'{file_path}-{page_num}.png',"PNG", quantize=False)

        try:
            for table_info in pdfplumb.pages[page_num].find_tables():
                x0 = table_info.bbox[0]
                y0 = table_info.bbox[1]
                x1 = table_info.bbox[2]
                y1 = table_info.bbox[3]
                table_list.append((x0, y0, x1, y1))
                table = table_info.extract()
                df = pd.DataFrame(table[1::], columns=table[0])
                df.replace('\x00', '', inplace=True)
                df.replace('Ÿ', '*', inplace=True)
                page_plumb_contents[int(y0)] = {"type":"table",
                                                "value": df.to_markdown()}
                line_meta.append({"ln": last_line,
                                  "type": "table",
                                  "context": df.to_markdown()})
        except Exception as e:
            print(e)
            continue

        line_meta = []
        try:
            for ln, content in enumerate(pdfplumb.pages[page_num].extract_text_lines()):
                txt_content = content['text']
                x0 = content['x0']
                y0 = content['top']
                x1 = content['x1']
                y1 = content['bottom']
                try:
                    if len(table_list) > 0:
                        if (table_list[0][0] < x0 and table_list[0][1] < y0 and
                                table_list[0][2] > x1 and table_list[0][3] > y1):#Filter context in outbound detected table contents
                            pass
                        else:
                            page_plumb_contents[int(y0)] = {"type": "text", "value": txt_content}
                            line_meta.append({"ln": ln+1,
                                              "type": "text",
                                                "context": txt_content})
                    else:
                        page_plumb_contents[int(y0)] = {"type": "text", "value": txt_content}
                        line_meta.append({"ln": ln + 1,
                                          "type": "text",
                                          "context": txt_content})
                    last_line = ln
                except Exception as e:
                    print(str(e))
        except Exception as e:
            print(e)
            continue
        if len(line_meta) > 0:
            page_meta = {"page": page_num + 1, "line_meta":line_meta}

        if len(page_plumb_contents) > 0:
            #각 페이지 단위 콘텐츠 결합
            pos_list = list(page_plumb_contents.keys())
            pos_list = sorted(pos_list)
            page_exist_tbl = False
            page_textonly_filtering = ""
            for position in pos_list:
                if page_plumb_contents[position]["type"] == "table":
                    if del_table == False:
                        page_textonly_filtering = re.sub(r"(?<![\.\?\!])\n", " ", page_textonly_filtering)
                        whole_page_extractinfo += page_textonly_filtering + "\n" + page_plumb_contents[position]["value"] + "\n"

                    page_textonly_filtering = ""
                    page_exist_tbl = True
                else:
                    page_textonly_filtering += page_plumb_contents[position]["value"] + "\n"

            if page_exist_tbl is False:
                page_textonly_filtering = re.sub(r"(?<![\.\?\!])\n", " ", page_textonly_filtering)
                whole_page_extractinfo += page_textonly_filtering

        if len(line_meta) > 0:
            doc_meta.append(page_meta)

    whole_page_extractinfo = re.sub(r"\(cid:[0-9]+\)", "", whole_page_extractinfo)
    print('Go Next document')
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
    output_file = '../metadump.json'

    try:
        files_meta = []
        for filepath in pathlib.Path("../source_doc/").rglob('*.pdf'):
            print(filepath)
            if filepath.is_file():
                whole_page_extractinfo, doc_meta = get_context_pdffile_by_plumber(filepath)
                files_meta.append({"origin_path": str(filepath),
                                   "doc_meta": doc_meta})

        save_to_text(files_meta, output_file)
        print(f"문서 내용이 성공적으로 추출되어 {output_file}에 저장되었습니다.")

    except Exception as e:
        print(f"오류 발생: {str(e)}")