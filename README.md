# 이기훈

이메일: rheekh00@snu.ac.kr | 전화번호: 010-3024-3937 | GitHub: [rheekh00](https://github.com/rheekh00)

---

## EDUCATION

- **서울대학교 컴퓨터공학과 석사과정 (졸업 예정)**  
  최적화 및 금융공학 연구실 (문병로 교수님)
- **서울대학교 기계공학과 학사 졸업**
- **경기북과학고등학교 졸업**

---

## SKILLS

- **프로그래밍 언어**: Python, C++, Java
- **AI/ML 프레임워크**: PyTorch, TensorFlow, Hugging Face, Deepspeed
- **클라우드 및 도구**: AWS, Docker, Git, Postman
- **기타**: CUDA, Nsight 프로파일링, Distributed Data Parallel (DDP)

---

## PROJECT & RESEARCH

### [1. Transformer 토큰 임베딩 연구](https://github.com/rheekh00/transformer-token-embedding)
- **목표**: 트랜스포머 모델의 토큰 임베딩 비교 및 분석
- **기술 스택**: PyTorch, Byte Pair Encoding (BPE), Transformer
- **성과**:
  - BLEU 점수 하락 없이 모델의 파라미터 수 20% 감소 및 학습 속도 15% 향상
  - 코사인 유사도와 자카드 유사도를 활용한 토큰 임베딩 비교 분석
  - 임베딩 행렬 공유 실험을 통해 성능 및 효율성 개선

### [2. Transformer 토큰 임베딩 학습 과정 분석](https://github.com/rheekh00/transformer-token-embedding-2)
- **목표**: 토큰 임베딩 벡터의 변화와 데이터 등장 빈도, 중복 및 다양성이 임베딩에 미치는 영향 분석
- **기술 스택**: PyTorch, RoBERTa, Wikitext-103-v1 데이터셋
- **성과**:
  - 자주 등장하는 토큰과 드문 토큰 간 학습 패턴 차이 분석
  - PCA 및 t-SNE 시각화를 활용한 토큰 간 관계 분석
  - 학습 초기 및 진행 과정에서의 코사인 유사도 변화 관찰

### [3. Hidden Markov Models를 활용한 주식 가격 변동 예측](https://github.com/rheekh00/hmm-price-prediction)
- **목표**: Hidden Markov Models (HMMs)와 Machine Learning, Deep Learning 모델을 결합하여 주식 가격 변동 예측
- **기술 스택**: Python, HMM, LSTM, SVM, Random Forest
- **성과**:
  - HMM 기반 시장 상태 분석을 통해 ML/DL 모델의 예측 정확도 개선
  - LSTM 모델에서 HMM-derived features 통합 후 accuracy 56.13% 달성
  - KOSPI 200 데이터 및 기술 지표를 활용한 실험

### [4. Bayesian Optimization을 이용한 스마트 팩토리 장비 배치 최적화](https://github.com/rheekh00/bayesian-optimization)
- **목표**: Digital Twin(DT) 플랫폼을 기반으로 Smart Factory 환경에서 장비 배치 최적화
- **기술 스택**: NVIDIA Omniverse Isaac Sim, Bayesian Optimization
- **성과**:
  - Digital Twin 환경 구축 및 작업 시간 28.8%~40.2% 단축
  - NVIDIA Omniverse를 통해 실제 공장 환경에 적용 가능한 배치 전략 도출
  - 2023 한국CDE 학회 우수 포스터상 수상

### [5. Transformer 모델 최적화](https://github.com/rheekh00/transformer-performance-optimization)
- **목표**: Transformer 모델의 학습 속도 및 성능 개선
- **기술 스택**: PyTorch, Mixed Precision Training, Nsight 프로파일링, AllReduce 최적화
- **성과**:
  - 혼합 정밀도 학습(Mixed Precision Training)을 통해 학습 속도 3배 향상
  - Nsight 프로파일링을 통한 GPU 통신 병목 분석 및 AllReduce 최적화
  - CUDA Streams 및 Gradient Bucketing을 활용한 GPU 메모리 효율성 개선

### [6. YouTube Shorts Generator](https://github.com/rheekh00/youtube-shorts-generator)
- **목표**: Generative AI를 활용한 유튜브 쇼츠 자동 생성 시스템 개발
- **기술 스택**: OpenAI API, ElevenLabs API, ShotStack API, Python
- **성과**:
  - 텍스트 생성, 음성 내레이션, 영상 편집을 자동화한 워크플로우 구축
  - 유튜브 채널 운영을 통해 총 113,617회 조회수 및 8.8만 회의 단일 영상 조회수 기록

### [7. 카드 사용내역 계정과목 분류기 개발](https://github.com/rheekh00/rag-expense-explainer)
- **목표**: RAG(Retrieval-Augmented Generation)를 활용한 분류 결과 설명 서비스 개발
- **기술 스택**: RAG, Python, PyTorch
- **성과**:
  - 도메인 지식 기반 문서를 활용해 모델의 분류 결과를 설명하는 서비스 구현
  - 모델의 신뢰성 및 사용자 이해도 향상

### [8. MPI 및 CUDA를 활용한 gpt2 125M 모델의 대규모 데이터 병렬 처리 및 최적화](https://github.com/rheekh00/gpt2-parallel-optimization)
- **목표**: 대규모 데이터 처리에서 효율성과 확장성 극대화
- **기술 스택**: MPI, OpenMP, CUDA, CUDA Streams, Shared Memory
- **성과**:
  - MPI를 활용한 데이터 분산 처리와 효율적 결과 수집
  - CUDA를 활용한 행렬 곱셈(Matmul) 최적화 및 Softmax 연산 효율화
  - GPU 자원 활용도를 높이며 처리 속도 및 확장성 극대화

---

## CAREER

### AI 솔루션 스타트업 달파
- **AI 엔지니어 (4개월)**  
  AWS, Docker, Git, Postman 등을 활용한 고객사 맞춤형 AI 솔루션 제공  
  백엔드 팀과 협업을 통한 시스템 통합 및 데이터 파이프라인 구축

---

## TEACHING ASSISTANT

- 문병로 교수님 '자료구조' 및 '알고리즘' 강의 조교 (4회)
- 삼성전자 DS 과정 '알고리즘' 강의 조교 (3회)

---

## AWARD

- **2023 한국CDE 학회 우수 포스터상**  
  디지털 트윈 환경 연구

---

## 개인 역량

- **데이터 중심 연구**: 데이터 분석 및 모델링 능숙
- **문제 해결 능력**: 복잡한 문제의 구조적 접근 및 해결
- **팀 협업**: 다양한 이해관계자와의 효과적인 소통 및 협업

