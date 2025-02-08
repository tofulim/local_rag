import urllib3
from bs4 import BeautifulSoup


class BlogCrawler:
    """
    블로그 크롤러
    주어진 url을 bs4로 html 파싱해 meta 데이터들을 가져온다.
    """
    def __init__(self):
        self.headers = {"User-Agent": "Mozilla/5.0"}
        self.pool_manager = urllib3.PoolManager()  # PoolManager 객체 생성

    def parse(self, url: str):
        response = self.pool_manager.request("GET", url, headers=self.headers)  # GET 요청 보내기

        # 응답 데이터를 파싱
        _soup = BeautifulSoup(response.data, "html.parser")

        return _soup

    def get_info(self, _soup: object):
        return {
            "title": self.get_title(_soup),
            "description": self.get_description(_soup),
            "image": self.get_image(_soup),
        }

    def get_title(self, _soup: object):
        try:
            return _soup.find("meta", property="og:title")["content"]
        except Exception:
            return None

    def get_description(self, _soup: object):
        try:
            return _soup.find("meta", property="og:description")["content"]
        except Exception:
            return None

    def get_image(self, _soup: object):
        try:
            return _soup.find("meta", property="og:image")["content"]
        except Exception:
            return None
