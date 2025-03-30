# 🍎 Fruit & Vegetable Classification with Pretrained CNNs
###  과일·채소 이미지 분류를 위한 사전학습 CNN 모델 기반 전이학습 성능 비교

> **카카오테크 부트캠프 (KakaoTech Bootcamp) | 개인 프로젝트 | 2025.03(2주)**  
> By Sandy.lee (이수인)

<br>

## Overview
본 프로젝트는 과일과 채소처럼 시각적으로 유사한 객체를 분류하는 과제를 통해, CNN 기반 전이 학습 모델들의 성능을 비교합니다.
<br><br>
본 프로젝트는 제한된 데이터 환경에서 CNN만으로 어떤 성능을 낼 수 있는지 탐색하며, 향후 식품 분류, 유통 관리, 레시피 추천 등 다양한 실생활 응용 가능성을 제시합니다.
- **주요 모델**: ResNet50, MobileNetV3-Large, EfficientNetV2-Small, ConvNeXt-Base
- **접근법**: Feature Extraction 방식으로 Transfer Learning
<br><br>
## Dataset
- **출처**: [Kritik Seth, "Fruits and Vegetables Image Recognition Dataset," Kaggle 2020](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition)
- **클래스 수**: 36개
- **데이터 구성**:
  - Train: 3,115장
  - Validation: 359장
  - Test: 351장
- **특징**:
  - 다양한 각도, 조명, 상태(절단, 조리 등) 포함
  - 클래스 간 데이터 균형 우수 → Accuracy 지표 활용 적합

![image](https://github.com/user-attachments/assets/393d60b7-1758-4433-9697-f137bf6ba748)

<br>

## Training Setup

- **전이학습 방식**: Feature Extraction (특징 추출 계층은 freeze)
- **데이터 증강**: 사각형 노이즈 + 모델별 transform
- **하이퍼파라미터**:
  - Optimizer: Adam (ConvNeXt는 AdamW)
  - Epochs: 50
  - Batch size: 64 or 32
  - Dropout: 0.2 (ConvNeXt는 LayerNorm)
  - ConvNeXt는 EMA(Exponential Moving Average) 적용
- **개발 환경**: Google Colab Pro+ (A100 GPU) + PyTorch
<br><br>
## 학습 로그 및 DB 연동

- **로그 저장**: Aiven MySQL + SQLAlchemy + pandas
- **분석 도구**: DBeaver
- **테이블 구성**:
  - `train_logs_{model}`: 에폭별 손실, 정확도 등
  - `test_results`: 모델별 테스트 성능 결과
<br><br>
## Results
<table align="center">
  <tr>
    <td>
        <table border="1" cellpadding="6" cellspacing="0">
        <tr><th>Model</th><th>Test Accuracy</th><th>Test Loss</th></tr>
        <tr><td><b>EfficientNetV2-S</b></td><td><b>97.4%</b></td><td><b>0.0995</b></td></tr>
        <tr><td>ResNet50</td><td>96.9%</td><td>0.1045</td></tr>
        <tr><td>MobileNetV3-L</td><td>96.7%</td><td>0.1066</td></tr>
        <tr><td>ConvNeXt-Base</td><td>96.6%</td><td>0.1472</td></tr>
      </table>
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/179ed1dd-9ca1-4572-974e-841fc5a7627c" width="600px" />
    </td>
  </tr>
</table>

- MobileNetV3는 경량 모델임에도 높은 성능을 보여 실용성 입증
- EfficientNetV2는 정확도 및 손실 모두 가장 우수한 결과를 기록
<br><br>
<table align="center">
  <tr>
    <td align="center"><b>ConvNeXt w/o EMA</b></td>
    <td align="center"><b>ConvNeXt w/ EMA</b></td>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/056aa53e-0492-432a-858b-be7182f1f77a" width="500px" />
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/7bae1e55-73b1-44b6-90c9-8723cdd429df" width="500px" />
    </td>
  </tr>
</table>

- EMA 적용 전 ConvNeX는 정확도 및 손실의 진동이 컸으나, EMA 적용 후 안정화됨

<br>

##  Tech Stack
- **Python 3.12**
- **PyTorch**
- **Google Colab (A100 GPU)**
- **Aiven MySQL + DBeaver**: 학습 로그 저장
- **Kaggle API**: 데이터 다운로드
<br><br>
## Reference
- Kritik Seth, "Fruits and Vegetables Image Recognition Dataset," Kaggle 2020 [https://www.kaggle.com/kritikseth/fruit-and-vegetable-image-recognition]
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. arXiv preprint arXiv:1512.03385. https://arxiv.org/abs/1512.03385
- Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling forconvolutional neural networks. arXiv preprint arXiv:1905.11946. https://arxiv.org/abs/1905.11946
- Howard, A., Sandler, M., et al. (2019). Searching for MobileNetV3. arXivpreprint arXiv:1905.02244. https://arxiv.org/abs/1905.02244
- Liu, Z., Mao, H., et al. (2022). A ConvNet for the 2020s. arXiv preprintarXiv:2201.03545. https://arxiv.org/abs/2201.03545
- Kakao Tech. (2022, May 4). 이미지 분류 모델 개발, 아직도 데이터만 기다려? Kakao Tech. https://tech.kakao.com/posts/623
