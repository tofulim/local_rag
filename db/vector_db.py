import faiss
import numpy as np

from embedding.text_embedding import Vectorizer


class VectorDB:
    """벡터 저장소
    faiss 라이브러리를 이용하여 벡터를 저장하고 검색하는 클래스
    """
    def __init__(self, dim=384):
        # 현재 지정한 embedding model이 384 dim을 사용하므로 default로 384 dim으로 설정
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)

    def add(self, vectors: np.ndarray):
        self.index.add(vectors)

    def search(self, query: np.ndarray, k: int = 1):
        distances, indices = self.index.search(query, k)
        return distances, indices


if __name__ == "__main__":
    vdb = VectorDB()

    vectorizor = Vectorizer()

    texts = [
        "A unique beverage that lowers weight",
        "A unique beverage that lowers weight and sugar",
        "A unique beverage that lowers weight, sugar, and strengthens the heart",
        "a healthy morning routine increases metabolism and general health while also keeping oneself busy.",
        "the man who loves south korea lives in japan.",
        "The subtle flavor and scent of cinnamon powder combined with the yellowish hue of turmeric in cinnamon and turmeric water will undoubtedly soothe your senses. Additionally, it will benefit general health in the following ways:actions that reduce inflammationHealth professionals claim that cinnamon and turmeric both have potent anti-inflammatory qualities. Turmeric’s curcumin and cinnamon’s cinnamaldehyde both aid in reducing inflammation in the body, which helps lessen the symptoms of long-term inflammatory diseases like arthritis.Activate the defenses of the bodyThe immune system may be strengthened by the antioxidant and antibacterial qualities of turmeric. However, cinnamon also possesses antibacterial qualities that can strengthen immunity overall and aid in the battle against infections.enhances digestionDigestion enzymes are stimulated by cinnamon, which aids in food breakdown and enhances digestion. Additionally, turmeric increases bile synthesis, which is necessary for the digestion of lipids.Control Blood Sugar LevelsNutritionists claim that cinnamon can help people with diabetes or those at risk of getting the disease because it improves insulin sensitivity and lowers blood sugar levels. It is claimed that turmeric increases insulin sensitivity, which facilitates better glucose use by the body and may result in better blood sugar regulation.Beneficial to Heart HealthStudies have shown that cinnamon and turmeric can both help lower blood pressure and improve blood flow, which can both lead to better heart health. By lowering vascular inflammation, turmeric’s anti-inflammatory qualities also promote heart health.Assistance in Losing WeightBoth cinnamon and turmeric can help control blood sugar levels and decrease food cravings, which can both contribute to long-term weight management. Turmeric can also increase metabolism.Beneficial to SkinTurmeric’s antioxidants are claimed to aid in the battle against free radicals, lessen aging symptoms, and encourage a radiant complexion. Conversely, it is believed that the antibacterial qualities of cinnamon aid in combating microorganisms that cause acne.Less Soreness and Pain in the MusclesExperts claim that the anti-inflammatory qualities of turmeric and cinnamon can aid in easing discomfort and tightness in the muscles.Enhances Mental AbilityAccording to certain research, cinnamon may help shield the brain, enhance cognitive performance, and lower the risk of neurodegenerative illnesses like Alzheimer’s.Enhances Mental Well-beingTurmeric’s curcumin, which has been demonstrated to have neuroprotective qualities, can pass the blood-brain barrier. Experts claim that cinnamon helps improve memory and learning, as well as other cognitive processes.detoxificationTurmeric and cinnamon are both considered to have detoxifying qualities since they aid in liver cleansing, enhance overall detoxification procedures, and enhance overall bodily function.Antimicrobial ResultsBecause of its inherent antibacterial qualities, cinnamon can aid in the treatment of illnesses and stop some bacteria from growing.",
        ]

    results = vectorizor(texts)

    for result in results:
        vectors = result["embeddings"]
        print(f"vector shape: {vectors.shape}")

        vdb.add(vectors)
        print(vdb.index.ntotal)

    # user_query = "japan and south korea's love"
    # query_vector = vectorizor([user_query])[0]["embeddings"].reshape(1, -1)
    # print(f"query_vector shape : {query_vector.shape}")
    # res = vdb.search(query_vector, 1)
    # # (array([[0.       , 1.4206209, 1.4327368]], dtype=float32), array([[3, 2, 1]]))
    # print(res)

    print(f"{vdb.index} is this")
