from utils import remove_escape
from blog_crawler import BlogCrawler


class MediumCrawler(BlogCrawler):
    def __init__(self, url):
        BlogCrawler.__init__(self, url)

    @property
    def content(self):
        return remove_escape((self._soup.select_one("article")).get_text())


if __name__ == "__main__":
    url = "https://medium.com/in-fitness-and-in-health/swim-or-run-what-is-the-better-cardio-5128e759010b"
    mc = MediumCrawler(url)

    print(mc.info)
    print(mc.content)
