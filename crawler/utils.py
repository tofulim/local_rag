import re

def remove_escape(text):
    return re.sub(r'\s+', ' ', text).strip()  # 여러 개의 공백/줄바꿈을 하나의 공백으로 정리
