import re
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
            return self.normalize_newlines(_soup.find("meta", property="og:title")["content"])
        except Exception:
            return None

    def get_description(self, _soup: object):
        try:
            return self.normalize_newlines(_soup.find("meta", property="og:description")["content"])
        except Exception:
            return None

    def get_image(self, _soup: object):
        try:
            return _soup.find("meta", property="og:image")["content"]
        except Exception:
            return None

    def get_content(self, _soup: object):
        return self.normalize_newlines(_soup.select_one("article").get_text())

    def normalize_newlines(self, text: str) -> str:
        # HTML에서 오는 비표준 공백 문자들 제거
        text = re.sub(r'[\u00A0\u2000-\u200B\u202F\u205F\u3000]', ' ', text)  # 유니코드 공백
        text = re.sub(r'\t', ' ', text)  # 탭 제거

        # 여러 줄 개행을 하나로
        text = re.sub(r'\n{2,}', '<DUMMY-NL>', text)

        # 여러 개의 공백을 하나로
        text = re.sub(r'[ ]{2,}', ' ', text)
        text = re.sub(r'\s+', ' ', text)

        # 각 줄 앞뒤 공백 제거
        lines = [line.strip() for line in text.splitlines()]
        text = '\n'.join(lines)
        text = text.replace("\xa0", "")  # 특수 공백 제거

        text = re.sub('<DUMMY-NL>', '\n', text)
        # 전체 앞뒤 공백 제거
        return text.strip()


if __name__ == "__main__":
    bc = BlogCrawler()

    soup = bc.parse(url="https://best.godtuza.net/1814")
    # print(bc.get_info(soup))
    # print(bc.get_title(soup))
    print(bc.get_content3(soup))
