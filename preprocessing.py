import os
import re
import pandas as pd
import json
import tqdm

# Functions to preprocess each file 
# The title and text of the file follow the format of the Naver web novel


def link_to_id(text):
    """
    (ko) 네이버 웹소설에서 등장인물을 표시하는 사진 링크를 가져와, id를 추출하는 전처리를 수행하는 함수.
    사진 링크가 있는 웹소설만을 데이터로 사용할 때 쓸 수 있는 함수로, 네이버 웹소설 형식에 맞춰져 있습니다.

    (en) It's a function that takes a photo link which displays a character in a Naver web novel, 
    and performs a preprocessing process to extract ids used. 

    논항 Args:
        text: 사진 링크가 포함된 네이버 웹소설 각 회차 컨텐츠(text).
              each episode of Naver Web Novel content (text) with photo link.

    리턴값 Returns:
        mod_text: 추출한 id로 링크를 대체한 수정된 텍스트
                  modified text that replaces the link with the extracted id
        ids: 추출한 id 리스트 (중복 있음). 
             the extracted id list (with duplicate).
    """
    pattern = re.compile(r'\.net/(\d+_\d+)/') # id extraction pattern

    matches = re.finditer(pattern, text)
    mod_text = text  # copy text to modify
    ids = list()

    for match in matches:
        if match:
            id_ = match.group(1)
            mod_text = re.sub(fr'https://.+?/{id_}.+=w80_2', f'{id_}: ', mod_text) #링크를 아이디로 변환
            ids.append(id_) 

    return mod_text, ids  # string tuple


def preprocessing(text):
    return re.sub('\n+', '\n', text.strip())    


# %%
# 데이터프레임으로 파일 정보 저장
# | 소설 이름 | 회차 | 회차 이름 | 회차 내용 (id는 전처리됨) | 등장인물 id |

# Save file information as a data frame
# | Novel Name | Round | Round Name | Round Content (id preprocessed) | Character ids |

file_path = './Novels_text/Labeled/'  
novdf = pd.DataFrame({'novtitle':[], 'r':[], 'rcontent':[], 'ids':[]})

for filename in os.listdir(file_path):
    filepath = os.path.join(file_path, filename)

    if os.path.isfile(filepath):
        with open(filepath, 'r', encoding='utf8') as f: 
            r = re.search(r'\d+', filename.split('_')[1]).group()  
            novtitle = re.sub('네이버웹소설.txt','', filename.split(',')[-1]).strip()         
            content, id_ = link_to_id(preprocessing(f.read())) # 정의해둔 전처리함수로 읽어들인 파일 전처리를 수행합니다.
            
            # Create a new DataFrame for the current row
            df = pd.DataFrame({'novtitle': [novtitle], 'r': [r], 'rcontent': [content], 'ids': [id_]})
            
            # Concatenate the new DataFrame with the existing DataFrame
            novdf = pd.concat([novdf, df], ignore_index=True)
            novdf['r'] = novdf['r'].astype(int)

# 소설 제목과 회차 순서대로 정렬 Sort in order of fiction titles and rounds
novdf = novdf.sort_values(by=['novtitle', 'r'], ascending=[True, True]).reset_index(drop=True)
novlist = list(set(novdf['novtitle']))

# print(novlist)
# novdf

# %%
def nov_concat(novdf):
    '''
    (ko) 소설 별 인물과 회차를 누적해서 데이터프레임으로 저장합니다.
    각 회차의 구분을 위해 회차 간 구분자를 '\n***\n'으로 설정합니다. '***'은 하나의 회차 내에서도 장면 구분을 위해 쓰이는 구분자이므로, 이후 scene을 나눌 때 일괄적으로 처리할 수 있습니다. 
    작가가 한 회차의 마지막 문장과 다음 회차의 첫 문장을 동일하게 반복하는 경우를 드물지 않게 볼 수 있습니다. 후작업에서 참고 바랍니다. 
    
    (en) It accumulates characters and episodes by novel and stores them as data frames.
    Note that the delimiter between rounds to '\n***\n' to separate each round. 

    Arg:
        novdf: | 소설 이름 | 회차 | 회차 이름 | 회차 내용 (id는 전처리됨) | 등장인물 id | 으로 저장한 각 파일의 정보 데이터프레임
               | Novel Name | Round | Round Name | Round Content (id preprocessed) | Character ids | (dataframe)

    Return:
        nov_cont: | 소설 이름 | 회차 별 내용을 누적한 전체 내용 | 누적된 id | 으로 저장한 각 소설의 정보 데이터프레임
                  | Title | Content | ID | (dataframe)
    '''
    novs = novdf.groupby('novtitle').agg({'rcontent':'\n***\n'.join, 'ids': lambda x: list(set(sum(x, [])))}).reset_index()
    novs.columns = ['Title', 'Content', 'ID']
    return novs


# print(nov_concat(novdf)['rcontent'][0])
accum_novdf = nov_concat(novdf)
accum_novdf

# %%
# Check Special Characters: To check quotes types
# 특수 문자 확인: 인용문의 종류 체크


def extract_special_characters(text):
    pattern = re.compile(r'[^\w\s|_|:|!|\.|?|…]')
    special_characters = list(set(re.findall(pattern, text)))

    return special_characters


print(*extract_special_characters(accum_novdf['Content'][1]), sep='\t')

# %%


def sum_all_text(PATH='', save=True, Label=False):
    """
    This function processes and organizes text data from Naver Web Novels.
    
    Parameters:
        PATH (str): Path to the directory containing the Naver Web Novel text files.
        save (bool): Optional flag to indicate whether to save the processed data as a JSON file.
        Label (bool): Optional flag to indicate whether to separate the dialogue from the narrative text.
    
    Returns:
        list_in_dic (list): List of dictionaries containing processed novel data.
    """
    os.chdir(PATH)
    # txt 확장자를 가진 파일만 불러오기
    file_list = [file for file in os.listdir() if file.endswith(".txt")]
    list_in_dic = []

    for file in tqdm(file_list):
        novel = {}
        f = open(file, 'r')
        text = f.read()

        name = file.split('.')
        novel['id'] = name[0].split('_')[0][1:]

        # title, epi_title 분리
        novel['epi_title'] = name[1].split('네이버웹소설')[0].strip()
        title = ', '.join(novel['epi_title'].split(',')[:-1])
        novel['title'] = novel['epi_title'].split(',')[-1]
        novel['epi_title'] = title

        novel['epi'] = re.sub(r'[^0-9]', '', name[0].split('_')[1])
        novel['contents'] = text

        # contents 분리
        new_con = list(re.sub('\n+', '\n', novel['contents']).split('\n'))
        novel['contents'] = new_con

        if Label is True:
            con = list()
            for i in novel['contents']:
                try:
                    if ':' in i:
                        speakers = i.split(':')[1].split('/')[3]
                        texts = ':'.join(i.split(':')[2:]).strip()
                        con.append(f'{speakers} : {texts}')
                    else:
                        con.append(i.strip())
                except:
                    con.append(i.strip())
            novel['contents'] = con

        list_in_dic.append(novel)

    if save is True:
        with open("navers.json", "w", encoding='utf-8') as f:
                        json.dump(list_in_dic, f, ensure_ascii=False, default=str, indent=4)

    return list_in_dic

