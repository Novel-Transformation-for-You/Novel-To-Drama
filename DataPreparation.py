from bs4 import BeautifulSoup
import requests
import re
from tqdm import tqdm


class WebNovelIndexing:
    def __init__(self):
        self.base_url = "https://novel.naver.com"
        self.genres = {'로맨스': 101, '로판': 109, '판타지': 102, '현판': 110, '무협': 103}
        self.links_per_gen = {}
        self.novel_info_list = []

    def url_by_genre(self):
        for g, i in self.genres.items():
            url = f"{self.base_url}/webnovel/genre?genre={i}"

            # Page pagination handling
            temp = [url]
            response = requests.get(url)
            if response.status_code == 200:
                dom = BeautifulSoup(response.text, 'html.parser')
                pages = dom.select('.default_paging a')
                for page in pages:
                    temp.append(f"{self.base_url}{page.attrs['href']}")
                self.links_per_gen[g] = temp

    def links_per_novel(self):
        links_per_novel = {}
        for genre, links in self.links_per_gen.items():
            temp = []
            for link in links:
                response = requests.get(link)
                if response.status_code == 200:
                    dom = BeautifulSoup(response.text, 'html.parser')
                    alist = dom.select('.card_list li a')
                    for a in alist:
                        temp.append(f"{self.base_url}{a.attrs['href']}")
            links_per_novel[genre] = temp
        return links_per_novel

    def novel_info(self):
        for genre, urls in tqdm(self.links_per_novel().items()):
            for url in tqdm(urls):
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
                    novel['total_epi'] = re.search(r'\d+', dom.select_one('.past_number').text).group(0)
                    self.novel_info_list.append(novel)
        return self.novel_info_list

    def sum_url(self):
        title_url = []
        for links in self.links_per_novel().values():
            title_url.extend(links)
        return title_url


if __name__ == "__main__":
    novel_crawler = WebNovelIndexing()
    novel_crawler.url_by_genre()
    novel_info_list = novel_crawler.novel_info()
    urls = novel_crawler.sum_url()
    print(novel_info_list)
    print(urls)