from language_model.qwen import DeepSeekQwen
from language_model.kanana import Kanana


class ModelFactory:
    """크롤러 팩토리
    주어진 문자열에 해당하는 크롤러 클래스 객체를 반환하는 팩토리 클래스
    """

    def __init__(self):
        self.platform2crawler = {
            "qwen": DeepSeekQwen,
            "kanana": Kanana,
        }

    def get(self, platform_name: str):
        crawler = self.platform2crawler.get(platform_name, None)

        assert crawler is not None, f"Invalid platform name: {platform_name}. now {self.platform2crawler.keys()} available"

        return crawler
