from crawler.blog_crawler import BlogCrawler
from crawler.medium_crawler import MediumCrawler
from crawler.tistory_crawler import TistoryCrawler


class CrawlerFactory:
    """크롤러 팩토리
    주어진 문자열에 해당하는 크롤러 클래스 객체를 반환하는 팩토리 클래스
    """

    def __init__(self):
        self.platform2crawler = {
            "blog": BlogCrawler,
            "tistory": TistoryCrawler,
            "medium": MediumCrawler,
        }

    def get(self, platform_name: str):
        crawler = self.platform2crawler.get(platform_name, None)

        assert crawler is not None, f"Invalid platform name: {platform_name}. now {self.platform2crawler.keys()} available"

        return crawler
