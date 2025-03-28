import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup

from utils.utils import remove_escape
from crawler.blog_crawler import BlogCrawler


class TistoryCrawler(BlogCrawler):
    """티스토리 크롤러
    티스토리는 static backend crawling이 되지 않고 Dynamic Rendering이 필요하므로 Selenium을 사용한다.
    1. 키워드 검색
    2. 검색 결과에서 블로그 포스트들의 url 추출
    """
    def __init__(self):
        BlogCrawler.__init__(self)

        self.tistory_url = "https://www.tistory.com"

    def get_topic_urls(self, topic: str, max_pages: int = 1):
        """topic search
        주제에 대해 검색한 url 10개 반환

        Args:
            topic (str): 검색할 주제

        Returns:
            urls (list): 검색 결과 url
        """
        # Headless 옵션 (브라우저 안 띄움)
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")

        # WebDriver 실행
        driver = webdriver.Chrome(options=options)

        # 티스토리 검색 URL
        search_url = f"{self.tistory_url}/search?keyword={topic}"

        driver.get(search_url)
        time.sleep(2)

        urls = []
        for page in range(1, max_pages + 1):
            # 현재 페이지의 HTML 가져오기
            soup = BeautifulSoup(driver.page_source, "html.parser")

            # 블로그 포스트 목록 추출
            posts = soup.select("div[class='item_group']")

            for post in posts:
                p = post.select("a.link_cont.zoom_cont")
                url = p[0].get("href")
                # title = post.select_one("strong.tit_cont").get_text()

                urls.append(url)

            # 다음 페이지 클릭
            try:
                next_btn = driver.find_element(By.XPATH, f'//*[@id="mArticle"]/div/div[4]/div[2]/div/a[{page + 1}]')

                next_btn.click()
                time.sleep(5)
            except Exception as e:
                print(f"[!] 다음 페이지 없음 또는 에러 발생: {e}")
                break

        driver.quit()
        return urls


    def get_content(self, _soup: object):
        return remove_escape((_soup.select_one("article")).get_text())


if __name__ == "__main__":
    tc = TistoryCrawler()

    # soup = mc.parse(url=url)

    # print(mc.get_info(soup))
    # print(mc.get_content(soup))
    # print(mc.get_topic_urls("startup"))

    urls = tc.get_topic_urls("ufc")
    print(f"urls: {urls}")

    for url in urls:
        soup = tc.parse(url)
        info = tc.get_info(soup)
        print(info)
