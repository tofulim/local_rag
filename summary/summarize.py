from time import time
from tqdm import tqdm
from transformers import pipeline


class Summarizer:
    def __init__(self, model_name: str = "facebook/bart-large-cnn", device: str = "mps"):
        self.summarizer = pipeline("summarization", model=model_name, device=device)
        self.batch_size = 4

    def __call__(self, texts: list[str], max_length: int = 150, min_length: int = 30):
        """요약

        배치 입력에 대해서 배치처리를 사용하지 않고 요약을 수행합니다.
        gpu 병렬처리 이점이 없어 apple silicon chip은 for loop을 통해 직렬처리하는 것이 더 빠릅니다.

        Args:
            texts (list[str]): contents to summarize
            max_length (int, optional): 반환할 요약문 최대 길이. Defaults to 150.
            min_length (int, optional): 반환할 요약문 최소 길이. Defaults to 30.

        Returns:
            results (list[dict]): 요약 결과 shape(B, {summary_text, time}) ex) [{"summary_texts": ["요약문"], "time": "0.00 sec"}, ...]
        """
        results = []

        batch_texts = [
            texts[i : i + self.batch_size]
            for i in range(0, len(texts), self.batch_size)
        ]

        for batch_text in tqdm(batch_texts, desc="Summarizing"):
            start = time()
            sum_results = self.summarizer(
                batch_text,
                max_length=max_length,
                min_length=min_length,
                no_repeat_ngram_size=3,  # 핵심
                early_stopping=True,
                repetition_penalty=1.5,
                do_sample=False,
            )
            end = time()

            text_results = list(map(lambda result: result["summary_text"].strip(), sum_results))

            results.append({
                "summary_texts": text_results,
                "time": f"{end - start:.2f} sec",
            })

        return results

if __name__ == "__main__":
    summarizer = Summarizer(
        model_name="jhgan/ko-sbert-sts",
    )
    # text = "A unique beverage that lowers weight, sugar, and strengthens the heartyahya alshraa·Follow3 min read·Aug 26, 2024--24ListenShareMaintaining a healthy morning routine increases metabolism and general health while also keeping oneself busy. Making water with turmeric and cinnamon is one morning ritual that one can incorporate into their regular morning schedule. The subtle flavor and scent of cinnamon powder combined with the yellowish hue of turmeric in cinnamon and turmeric water will undoubtedly soothe your senses. Additionally, it will benefit general health in the following ways:actions that reduce inflammationHealth professionals claim that cinnamon and turmeric both have potent anti-inflammatory qualities. Turmeric’s curcumin and cinnamon’s cinnamaldehyde both aid in reducing inflammation in the body, which helps lessen the symptoms of long-term inflammatory diseases like arthritis.Activate the defenses of the bodyThe immune system may be strengthened by the antioxidant and antibacterial qualities of turmeric. However, cinnamon also possesses antibacterial qualities that can strengthen immunity overall and aid in the battle against infections.enhances digestionDigestion enzymes are stimulated by cinnamon, which aids in food breakdown and enhances digestion. Additionally, turmeric increases bile synthesis, which is necessary for the digestion of lipids.Control Blood Sugar LevelsNutritionists claim that cinnamon can help people with diabetes or those at risk of getting the disease because it improves insulin sensitivity and lowers blood sugar levels. It is claimed that turmeric increases insulin sensitivity, which facilitates better glucose use by the body and may result in better blood sugar regulation.Beneficial to Heart HealthStudies have shown that cinnamon and turmeric can both help lower blood pressure and improve blood flow, which can both lead to better heart health. By lowering vascular inflammation, turmeric’s anti-inflammatory qualities also promote heart health.Assistance in Losing WeightBoth cinnamon and turmeric can help control blood sugar levels and decrease food cravings, which can both contribute to long-term weight management. Turmeric can also increase metabolism.Beneficial to SkinTurmeric’s antioxidants are claimed to aid in the battle against free radicals, lessen aging symptoms, and encourage a radiant complexion. Conversely, it is believed that the antibacterial qualities of cinnamon aid in combating microorganisms that cause acne.Less Soreness and Pain in the MusclesExperts claim that the anti-inflammatory qualities of turmeric and cinnamon can aid in easing discomfort and tightness in the muscles.Enhances Mental AbilityAccording to certain research, cinnamon may help shield the brain, enhance cognitive performance, and lower the risk of neurodegenerative illnesses like Alzheimer’s.Enhances Mental Well-beingTurmeric’s curcumin, which has been demonstrated to have neuroprotective qualities, can pass the blood-brain barrier. Experts claim that cinnamon helps improve memory and learning, as well as other cognitive processes.detoxificationTurmeric and cinnamon are both considered to have detoxifying qualities since they aid in liver cleansing, enhance overall detoxification procedures, and enhance overall bodily function.Antimicrobial ResultsBecause of its inherent antibacterial qualities, cinnamon can aid in the treatment of illnesses and stop some bacteria from growing."
    text = """미들급의 차기 챔피언 후보는 누구일까요? 저는 함자트 치마예프라고 생각합니다. 뒤플레시와 션 스트릭랜드 둘 중에 누가 승리하던 치마예프는 엄청나게 위협적인 상대입니다. 개인적으로 치마예프에게 쉬운 상대는 스트릭랜드라고 생각하고 뒤플레시는 상성상 쉽지 않은 경기가 될 것 같습니다. 그럼에도 함자트 치마예프의 레슬링 스킬은 압도적이기 때문에 미들급 챔피언에 가장 유력한 후보라고 생각합니다.그 다음으로 눈여겨보고 있는 선수는 앤서니 에르난데스 선수입니다. 이 선수를 보면 전성기 케인 벨라스케즈를 보는 것 같습니다. 엄청난 압박능력과 그래플링의 연계가 매우 뛰어난 선수입니다. 앤서니 에르난데스가 쭉쭉 치고올라가서 탑5 선수들과 싸우는 모습을 보고 싶네요. 그리고 만약 함자트 치마예프와 경기가 펼쳐진다면, 치마예프에게 가장 힘든 스타일의 선수입니다. 앤서니 에르난데스가 어디까지 올라갈 수 있는 지 너무 궁금합니다.
"""

    res = summarizer(
        texts=[text, text, text],
    )
    print(res)
