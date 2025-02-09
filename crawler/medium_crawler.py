import re
import json
import requests

from utils.utils import remove_escape
from crawler.blog_crawler import BlogCrawler


class MediumCrawler(BlogCrawler):
    def __init__(self):
        BlogCrawler.__init__(self)

        self.medium_url = "https://medium.com"

    def get_topic_urls(self, topic: str):
        """topic search
        주제에 대해 검색한 url 10개 반환

        Args:
            topic (str): 검색할 주제

        Returns:
            decoded_urls (list): 검색 결과 url 10개
        """
        search_url = f"{self.medium_url}/search?q={topic}"
        response = requests.get(
            url = search_url,
        )

        pattern = 'mediumUrl\":\"(.*?)\"'
        urls = re.findall(pattern, response.text)
        # \u002F 를 정상적으로 치환해준다
        decoded_urls = list(map(lambda url: json.loads(f'"{url}"'), urls))

        return decoded_urls


    def get_content(self, _soup: object):
        return remove_escape((_soup.select_one("article")).get_text())


if __name__ == "__main__":
    # url = "https://medium.com/@yahyanf2/a-unique-beverage-that-lowers-weight-sugar-and-strengthens-the-heart-42fe73343d48"
    mc = MediumCrawler()

    # soup = mc.parse(url=url)

    # print(mc.get_info(soup))
    # print(mc.get_content(soup))
    # print(mc.get_topic_urls("startup"))

    urls = mc.get_topic_urls("ufc middleweight")

    for url in urls:
        print("*"*50)
        print(url)
        soup = mc.parse(url)
        print(mc.get_info(soup))
        content = mc.get_content(soup)
        print(f"content length: {len(content)}")
        print(f"{content[:50]} ...")

    """
    **************************************************
    https://mysteryweevil.medium.com/python-in-the-food-and-beverage-industry-a-recipe-for-success-fb7636134d8a
    {'title': 'Python in the Food and Beverage Industry: A Recipe for Success', 'description': 'In today’s fast-paced world, the food and beverage industry is constantly evolving to meet the demands of consumers. From streamlining…', 'image': 'https://miro.medium.com/v2/da:true/resize:fit:1200/0*o15F66jH4hqsyNEw'}
    content length: 1356
    Member-only storyPython in the Food and Beverage I ...
    **************************************************
    https://pamchmiel.medium.com/how-one-scientist-creates-award-winning-strain-based-beverages-51863a52b4bb
    {'title': 'How One Scientist Creates Award-Winning Strain-Based Beverages', 'description': 'This article first appeared in Fat Nugs Magazine.', 'image': 'https://miro.medium.com/v2/resize:fit:1200/1*lj-6ZsI9Hh50M7dmZhmFuA.png'}
    content length: 7710
    How One Scientist Creates Award-Winning Strain-Bas ...
    **************************************************
    https://medium.com/@mssanta/morning-movements-setting-the-tone-for-a-healthier-day-with-the-perfect-beverage-5b0b191714bb
    {'title': 'Morning Movements: Setting the Tone for a Healthier Day with the Perfect Beverage', 'description': 'What’s your initial morning routine?', 'image': 'https://miro.medium.com/v2/resize:fit:1200/1*URND_Arbgzm5D_uH8C64uA.jpeg'}
    content length: 1318
    Member-only storyMorning Movements: Setting the To ...
    """
