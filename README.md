# 💡 RECOGNIZE-SPEAKERS-IN-TEXT 📄

**모델 설명**: 네이버 웹소설 텍스트를 드라마 각본으로 변환해주는 모델입니다. 

**MODEL DESCRIPTION**: A model that converts the text of Naver's web novel into a drama script.

---
1. **초록**

  음성 인식 분야에서의 발화자 인식 기술과 달리, 한국어 텍스트 내 발화자 인식 기술은 그 필요성에도 불구하고 발전이 더딥니다. 트랜스포머도 이 분야에서는 강한 성능을 보이지 못합니다. 본 모델은 정확도를 위한 룰 베이스 전처리 및 후처리와 트랜스포머 모델의 사용으로 웹소설에서 화자를 인식합니다. 발화가 이루어지는 시간과 공간의 인식을 통해 장면을 구분하고, 이를 바탕으로 드라마 각본을 생성합니다. 웹소설 분야를 선택한 이유는 상업성과 문서의 특징 때문입니다. 웹소설은 최근 자체 시장과 드라마/영화 등 2차 가공물 시장에서 뛰어난 매출 실적을 보입니다. 또, 대부분 휴대전화로 읽는다는 특성 때문에 발화와 장면 전환마다의 줄바꿈이 명확합니다. 모델의 성능 평가 결과는 다음과 같습니다. [성능평가 결과 서술]

1. **ABSTRACT**

   Unlike speaker recognition technology in the field of speech recognition, speaker recognition technology in Korean text is slow to develop despite its need. The Transformer models also don't have strong performance in this area. Our model recognizes speakers in web novels with rule-based pre-processing and post-processing for accuracy and the use of Transformer models. The scene is distinguished through the perception of the time and space in which the utterance takes place, and a drama script is generated based on this. We chose the web novel field because of its commerciality and document characteristics. Web novels have recently shown outstanding sales performance in their own markets and secondary production markets such as dramas and movies. In addition, because most of them are read on mobile phones, the line changes for each speech and scene change are clear. The performance evaluation results of the model are as follows. [Description of performance evaluation results]
   
---
2. **배경**
   
   발화자 인식은 음성 인식 분야에서 주로 발전되어 왔으며, 문서 속 발화자 인식 기술은 두각을 드러내지 못하고 있습니다. 그러나 문서에서도 발화자 인식은 필요합니다. 이미 기록된 문서에서 발화자를 인식하고 찾는 작업은 현재 사람이 담당하고 있습니다. 따라서 회의, 상담, 소설 등 음성이 없는 텍스트에서 발화자를 구분할 수 있다면 생산성이 비약적으로 상승할 것입니다. 이러한 필요에도 불구하고 텍스트 내 발화자 인식 기술이 발전하지 못한 것은 구현의 어려움 때문입니다. 특히 우리말의 경우, 영어와 달리 인칭의 구분이 없으며 주어의 생략이 잦습니다. 따라서 발화 인식을 위해서는 앞뒤 문맥의 고려가 필요합니다.
   
   대부분의 자연어처리 기술은 트랜스포머 이전과 이후로 크게 달라졌습니다. chatGPT는 NLU를 실현한 것처럼 보이며, hallucination 등의 문제에도 불구하고 많은 사람의 성원을 받았습니다. 한국어 텍스트 내 발화자 인식 영역도 트랜스포머의 사용으로 기존보다 용이해질 것이라 예상했으나, 실제는 예상과 달랐습니다. #Markus Krug#에 따르면, 룰 기반 방식과 트랜스포머의 발화자 인식 정확도를 비교했을 때 후자가 유의하게 높지 않음을 확인했습니다.



2. **BACKGROUND AND PROBLEMS**
     
   Speaker recognition has mainly developed in the field of speech recognition, and speaker recognition technology in documents has not been prominent. However, the document also requires speaker recognition. Recognizing and finding speakers in already recorded documents is currently handled by a person. Therefore, productivity will rise dramatically if we can distinguish speakers from non-voice texts such as meetings, counseling, and novels. Despite this need, speaker recognition technology in the text has not advanced because of the difficulty of implementation. In particular, in the case of Korean, unlike English, there is no distinction between grammatical person, and the subject is often omitted. Therefore, consideration of the context before and after is required for speech recognition.
   
---
 2. **선행 연구**
    1. 한국어
    2. 영어
    3. 그 외
   
  2. **Related Works**
---
 3. **분야 선정**
    1. 웹소설
    2. 드라마
   
---
 4. **사용 기술**
    1. Named Entity Recognition (NER)
    2. Relation Extraction (RE)
    3. Pretrained Models
   
---
 5. **데이터**
    1. 네이버 웹소설 무료본
    2. 드라마 각본
---
 6. **모델**

---
 7. **성능**

---
 8. **배포**
