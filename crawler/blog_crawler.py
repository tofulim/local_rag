import urllib3
from bs4 import BeautifulSoup


class BlogCrawler:
    def __init__(self, url):
        headers = {"User-Agent": "Mozilla/5.0"}
        http = urllib3.PoolManager()  # PoolManager 객체 생성
        response = http.request("GET", url, headers=headers)  # GET 요청 보내기

        # 응답 데이터를 파싱
        self._soup = BeautifulSoup(response.data, "html.parser")

    @property
    def info(self):
        return {
            "title": self.title,
            "description": self.description,
            "image": self.image,
        }

    @property
    def title(self):
        return self._soup.find("meta", property="og:title")["content"]

    @property
    def description(self):
        return self._soup.find("meta", property="og:description")["content"]

    @property
    def image(self):
        return self._soup.find("meta", property="og:image")["content"]
