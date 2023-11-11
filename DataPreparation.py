from bs4 import BeautifulSoup
import requests
import re
import json
import os


class WebNovelIndexing:
    def get_novel_info(self, base_url="https://novel.naver.com", genres={'로맨스': 101, '로판': 109, '판타지': 102, '현판': 110, '무협': 103}, save='Y'):
        """
        네이버 웹소설의 정보를 가져옵니다.

        Args:
            base_url: 네이버 웹소설의 기본 URL입니다.
            genres: 네이버 웹소설의 장르 목록입니다.

        Returns:
            네이버 웹소설의 정보 목록입니다.
        """

        self.novel_info_list = []

        # 장르별 URL 가져오기
        self.links_per_genre = {}
        for genre in genres:
            url = f"{base_url}/webnovel/genre?genre={genres[genre]}"

            # 페이지네이션 처리
            temp = [url]
            response = requests.get(url)
            if response.status_code == 200:
                dom = BeautifulSoup(response.text, 'html.parser')
                pages = dom.select('.default_paging a')
                for page in pages:
                    temp.append(f"{base_url}{page.attrs['href']}")
                self.links_per_genre[genre] = temp

        # 소설별 URL 가져오기
        self.links_per_novel = {}
        for genre, links in self.links_per_genre.items():
            temp = []
            for link in links:
                response = requests.get(link)
                if response.status_code == 200:
                    dom = BeautifulSoup(response.text, 'html.parser')
                    alist = dom.select('.card_list li a')
                    for a in alist:
                        temp.append(f"{base_url}{a.attrs['href']}")
            self.links_per_novel[genre] = temp

        # 소설 정보 가져오기
        for genre, urls in self.links_per_novel.items():
            for url in urls:
                novel = {}
                response = requests.get(url)
                if response.status_code == 200:
                    dom = BeautifulSoup(response.text, 'html.parser')

                    # id 추가
                    novel['id'] = re.search(r'\d{6,}', url).group(0)

                    # 제목 추가
                    novel['title'] = dom.select_one('title').text

                    # 작가 추가
                    novel['writer'] = dom.select_one(".info_area a").text

                    # genre 추가
                    novel['genre'] = genre

                    # 전체 회차 수 추가
                    novel['total_epi'] = re.search(
                        r'\d+', dom.select_one('.past_number').text).group(0)

                    self.novel_info_list.append(novel)
        
        if save == 'Y':
            path = os.getcwd()
            folder_path = path + '/data'
            
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            
            # 파일에 저장
            with open(f"{folder_path}/novel_info_list.json", "w", encoding='utf-8') as f:
                json.dump(self.novel_info_list, f, ensure_ascii=False, default=str, indent=4)

        return self.novel_info_list
    
    def get_download(self, novel_info_list, only_label=False):
        '''
        웹소설을 회차별로 다운로드합니다. 현재 작업중인 디렉토리에 txt파일 형태로 다운로드됩니다.
        
        Args:
            novel_info_list: json형식으로 된 novel_info_list입니다. (get_novel_info 메서드 통해서 획득 가능)
            only_label: True일 경우, 발화자가 표시된 웹소설만 다운로드 합니다. (default=False)
        '''
        
        # 전체 작품 통합 리스트 생성
        self.total_url = []
        for novel in novel_info_list:
            novel_id = novel['id']
            total_epi = novel['total_epi']
            for epi in range(1, 1+int(total_epi)):
                self.total_url.append(f"https://novel.naver.com/webnovel/detail?novelId={novel_id}&volumeNo={epi}")
        
        # 각 링크에 대해 다운로드 진행
            for link_per_epi in self.total_url: 
                full_text = ""
                response = requests.get(link_per_epi)
                if response.status_code == 200:
                    dom = BeautifulSoup(response.text, 'html.parser')
                    plist = dom.select('.detail_view_content p')
                    filename = re.search(r'\d{6,}', link_per_epi)[0] + "_" + re.sub("[\/#:?]", "", dom.select_one('title').text) # 파일명 = id_특문 제거된 제목

                if only_label: # only_label 옵션을 줬을 때 동작부 정의   
                    # 경로 생성  
                    path = os.getcwd()
                    webnovel = path + "/WebNovel"
                    if not os.path.exists(webnovel):
                        os.makedirs(webnovel)
                    
                    # 다운로드
                    for tag in plist:
                        if tag.select('a'):
                            img_link = tag.select_one('a img').attrs['src']
                            full_text += "\n" + img_link + " : " + tag.text
                        else:
                            full_text += "\n" + tag.text
                    if dom.select('.talk'): # 화자 이미지가 존재한다면
                        with open(f'{webnovel}/L{filename}.txt', 'w', encoding='utf-8') as fp:
                            fp.write(full_text)
                            
                else: # only_label 옵션 껐을 때
                    # 경로 생성
                    path = os.getcwd()
                    webnovel = path + "/WebNovel"
                    labeled_path = webnovel + "/Labeled"
                    unlabeled_path = webnovel + "/Unlabeled"
                    if not os.path.exists(labeled_path):
                        os.makedirs(labeled_path)
                    if not os.path.exists(unlabeled_path):
                        os.makedirs(unlabeled_path)
                        
                    # 다운로드
                    for tag in plist:
                        if tag.select('a'): # 
                            img_link = tag.select_one('a img').attrs['src']
                            full_text += "\n" + img_link + " : " + tag.text
                        else:
                            full_text += "\n" + tag.text 
                    if dom.select('.talk'): # 화자 이미지가 존재한다면
                        with open(f'{labeled_path}/L{filename}.txt', 'w', encoding='utf-8') as fp:
                            fp.write(full_text)
                    else:
                        with open(f'{unlabeled_path}/U{filename}.txt', 'w', encoding='utf-8') as fp:
                            fp.write(full_text)