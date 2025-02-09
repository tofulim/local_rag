from time import time
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import pipeline


class Vectorizer:
    """벡터 변환기
    주어진 입력 쿼리 문장 혹은 RAG에 사용할 참조 문서의 요약 문단을 벡터로 변환합니다.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "mps"):
        self.model = SentenceTransformer(model_name_or_path=model_name, device=device)
        self.batch_size = 16

    def __call__(self, texts: list[str], convert_to_tensor: bool = False):
        """요약

        배치 입력에 대해서 배치처리를 사용하지 않고 요약을 수행합니다.
        gpu 병렬처리 이점이 없어 apple silicon chip은 for loop을 통해 직렬처리하는 것이 더 빠릅니다.

        Args:
            texts (list[str]): contents to vectorize
            convert_to_tensor (bool, optional): 형변환 여부

        Returns:
            results (list[dict]): 요약 결과 ex) [{"summary_text": "요약문", "time": "0.00 sec"}]
        """
        results = []

        batch_texts = [
            texts[i : i + self.batch_size]
            for i in range(0, len(texts), self.batch_size)
        ]

        for batch_text in tqdm(batch_texts, desc="Vectorizing"):
            start = time()
            result = self.model.encode(batch_text, batch_size=self.batch_size, convert_to_tensor=convert_to_tensor)
            end = time()

            results.append({
                "embeddings": result,
                "time": f"{end - start:.2f} sec",
            })

        return results

if __name__ == "__main__":
    vectorizer = Vectorizer()
    text = "A unique beverage that lowers weight, sugar, and strengthens the heartyahya alshraa·Follow3 min read·Aug 26, 2024--24ListenShareMaintaining a healthy morning routine increases metabolism and general health while also keeping oneself busy. Making water with turmeric and cinnamon is one morning ritual that one can incorporate into their regular morning schedule. The subtle flavor and scent of cinnamon powder combined with the yellowish hue of turmeric in cinnamon and turmeric water will undoubtedly soothe your senses. Additionally, it will benefit general health in the following ways:actions that reduce inflammationHealth professionals claim that cinnamon and turmeric both have potent anti-inflammatory qualities. Turmeric’s curcumin and cinnamon’s cinnamaldehyde both aid in reducing inflammation in the body, which helps lessen the symptoms of long-term inflammatory diseases like arthritis.Activate the defenses of the bodyThe immune system may be strengthened by the antioxidant and antibacterial qualities of turmeric. However, cinnamon also possesses antibacterial qualities that can strengthen immunity overall and aid in the battle against infections.enhances digestionDigestion enzymes are stimulated by cinnamon, which aids in food breakdown and enhances digestion. Additionally, turmeric increases bile synthesis, which is necessary for the digestion of lipids.Control Blood Sugar LevelsNutritionists claim that cinnamon can help people with diabetes or those at risk of getting the disease because it improves insulin sensitivity and lowers blood sugar levels. It is claimed that turmeric increases insulin sensitivity, which facilitates better glucose use by the body and may result in better blood sugar regulation.Beneficial to Heart HealthStudies have shown that cinnamon and turmeric can both help lower blood pressure and improve blood flow, which can both lead to better heart health. By lowering vascular inflammation, turmeric’s anti-inflammatory qualities also promote heart health.Assistance in Losing WeightBoth cinnamon and turmeric can help control blood sugar levels and decrease food cravings, which can both contribute to long-term weight management. Turmeric can also increase metabolism.Beneficial to SkinTurmeric’s antioxidants are claimed to aid in the battle against free radicals, lessen aging symptoms, and encourage a radiant complexion. Conversely, it is believed that the antibacterial qualities of cinnamon aid in combating microorganisms that cause acne.Less Soreness and Pain in the MusclesExperts claim that the anti-inflammatory qualities of turmeric and cinnamon can aid in easing discomfort and tightness in the muscles.Enhances Mental AbilityAccording to certain research, cinnamon may help shield the brain, enhance cognitive performance, and lower the risk of neurodegenerative illnesses like Alzheimer’s.Enhances Mental Well-beingTurmeric’s curcumin, which has been demonstrated to have neuroprotective qualities, can pass the blood-brain barrier. Experts claim that cinnamon helps improve memory and learning, as well as other cognitive processes.detoxificationTurmeric and cinnamon are both considered to have detoxifying qualities since they aid in liver cleansing, enhance overall detoxification procedures, and enhance overall bodily function.Antimicrobial ResultsBecause of its inherent antibacterial qualities, cinnamon can aid in the treatment of illnesses and stop some bacteria from growing."

    res = vectorizer(
        texts=[text],
    )
    print(res)
