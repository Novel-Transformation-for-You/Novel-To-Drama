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

        novel_info_list = []

        # 장르별 URL 가져오기
        links_per_genre = {}
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
                links_per_genre[genre] = temp

        # 소설별 URL 가져오기
        links_per_novel = {}
        for genre, links in links_per_genre.items():
            temp = []
            for link in links:
                response = requests.get(link)
                if response.status_code == 200:
                    dom = BeautifulSoup(response.text, 'html.parser')
                    alist = dom.select('.card_list li a')
                    for a in alist:
                        temp.append(f"{base_url}{a.attrs['href']}")
            links_per_novel[genre] = temp

        # 소설 정보 가져오기
        for genre, urls in links_per_novel.items():
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

                    novel_info_list.append(novel)
        
        if save == 'Y':
            path = os.getcwd()
            folder_path = path + '/data'
            
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            
            # 파일에 저장
            with open(f"{folder_path}/novel_info_list.json", "w", encoding='utf-8') as f:
                json.dump(novel_info_list, f, ensure_ascii=False, default=str, indent=4)

        return novel_info_list