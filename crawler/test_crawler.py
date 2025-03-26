from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
import time
from bs4 import BeautifulSoup

def crawl_tistory_search(keyword: str, max_pages: int = 3):
    # Headless 옵션 (브라우저 안 띄움)
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")

    # WebDriver 실행
    driver = webdriver.Chrome(options=options)

    # 티스토리 검색 URL
    search_url = f"https://www.tistory.com/search?keyword={keyword}"
    driver.get(search_url)
    time.sleep(2)

    results = []

    for page in range(max_pages):
        # 현재 페이지의 HTML 가져오기
        soup = BeautifulSoup(driver.page_source, "html.parser")

        # 블로그 포스트 목록 추출
        posts = soup.select("div[class='list_tistory_top']")

        for post in posts:
            # title = post.get_text(strip=True)
            p = post.select("a.link_cont.zoom_cont")
            print(f"p: {p[0]}")
            link = p[0].get("href")
            # results.append((title, link))
            results.append(link)

        print(f"results: {results}")



        # 다음 페이지 클릭
        try:
            next_btn = driver.find_element(By.CLASS_NAME, "btn_next")
            next_btn.click()
            time.sleep(2)
        except Exception as e:
            print(f"[!] 다음 페이지 없음 또는 에러 발생: {e}")
            break

    driver.quit()
    return results


# ✅ 실행 예제
if __name__ == "__main__":
    keyword = "파이썬"
    data = crawl_tistory_search(keyword)

    for i, (title, link) in enumerate(data, 1):
        print(f"{i:02d}. {title} - {link}")
