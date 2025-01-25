#!/usr/bin/env python
# coding: utf-8

# In[10]:


from langchain_core.prompts import PromptTemplate
from datasets import load_dataset,DatasetDict
from tqdm import tqdm
import pathlib
import json
import yaml
from langchain_ollama import OllamaLLM, ChatOllama


# In[11]:


prompt = PromptTemplate.from_template(
        """Context information is below. You are only aware of this context and nothing else.
        ---------------------        
        {context}        
        ---------------------
        Given this context, generate only questions based on the below query.
        Your task is to provide exactly **{num_questions}** question(s) for an upcoming quiz/examination. 
        You are not to provide more or less than this number of questions. 
        The question(s) should be diverse in nature across the document. 
        The purpose of question(s) is to test the understanding of the students on the context information provided.
        You must also provide the answer to each question. The answer should be based on the context information provided only.
        
        Restrict the question(s) to the context information provided only.
        QUESTION and ANSWER should be written in Korean. response in JSON format which contains the `question` and `answer`.
        DO NOT USE List in JSON format.
        ANSWER should be a complete sentence.
        
        #Format:
        ```json
        {{
            "QUESTION": "PMDU(prime minister’s delivery unit)가 어떤 역할을 하는 조직인가요?",
            "ANSWER": "PMDU(Prime Minister’s Delivery Unit)는 일반적으로 국가 주요 우선 과제의 진행 상황을 감독하고 개선하기 위해 설립됩니다. "
        }},
        {{
            "QUESTION": "조직 형태로서의 네트워크는 계층제와 시장이라는 조직형태에서 어떠한 특성을 가지는가?",
            "ANSWER": "계층제적 지배구조는 수평적⋅수직적으로 분화되어 있고 지시⋅명령과 같은 행정적 수단에 의해 통제된다."    
        }},
        {{
            "QUESTION": "향후 발전을 위해 정부 역할은 어떻게 설정되어야 할까?",
            "ANSWER": "정부역할에 대한 새로운 관심과 개혁 노력이 뒤따를 필요가 있다."    
        }}
        ```
        """
        )


# In[12]:


def custom_json_parser(response):
    json_string = response.content.strip().removeprefix("```json\n").removesuffix("\n```").strip()
    json_string = f'[{json_string}]' #1개의 json이 나올때도 있고 여러 json이 발생할 경우도 있으니 list.extends를 감안해서.
    try:
        json_fmt = json.loads(json_string)
        return json_fmt
    except Exception as e:
        print(json_string)
        print(str(e))
        return None


# In[13]:


model_id = "gemma2:2b"
model = ChatOllama(model=model_id, temperature=0, format='json')


# In[14]:


chain =  prompt | model| custom_json_parser


# ##### 질문과 답변 생성 예제 2

# In[ ]:


folder_path = 'meta_aggregate'
idx = 0
for metafile in pathlib.Path(folder_path).rglob("*.json"):
    qaset_list = []
    with open(metafile, "r", encoding="utf-8") as fp:
        print(metafile)
        json_info = json.load(fp)
        title = json_info['filename']
        json_key = list(json_info.keys())
        json_key.remove('filename')
        for pi, page_num in enumerate(json_key):
            contents = json_info[page_num]
            print(f"page index {pi}")
            llm_rtn = chain.invoke({"context": f"{contents}", "num_questions": "3"})        
            if llm_rtn is not None:
                qa_modified = {"question":'','answer': '', "source":title}
                for qa_pair in llm_rtn:
                    key_list = list(qa_pair.keys())
                    for item in key_list:
                        if "q" in item.lower():
                            qa_modified['question'] = f"{title}에서 {qa_pair[item]}"
                        elif "a" in item.lower():
                            qa_modified['answer'] =qa_pair[item]
                    qaset_list.append(qa_modified)
            else:
                continue
    with open(f"qaset/{idx}.json", "w", encoding='utf-8') as fp:
            fp.write(json.dumps(qaset_list, ensure_ascii=False,indent=2))
    idx += 1
print("finish")


# In[ ]:




