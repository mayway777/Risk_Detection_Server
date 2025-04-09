
# 🚨 Video Hazard Detection Server

이 프로젝트는 YOLOv5 기반의 차량, 화재, 낙상, 안전 위험 물체 감지를 수행하고, 분석 결과를 실시간으로 Firebase Firestore에 저장하는 **FastAPI 기반 백엔드 서버**입니다.  
해당 서버는 프론트엔드 웹 애플리케이션과 연동되어 위험 상황을 사용자에게 시각적으로 제공하는 기능을 담당합니다.
---
👉 [클라이언트 레포지토리 보기](https://github.com/mayway777/Risk_Detection_Client.git)
---

## ✅ 주요 기능

- **YOLOv5 모델 4종 활용**
  - `yolov5s`: 차량(Car) 감지
  - `FallDown.pt`: 낙상 감지
  - `fire.pt`: 화재 감지
  - `safe.pt`: 안전 위험 물체 감지 (ex. 위험한 도구 등)
- **감지 로직 차별화**
  - 차량은 2초 이상 감지 시만 카운트
  - 화재·낙상·위험 물체는 1회 감지 즉시 카운트
- **Firebase 연동**
  - 감지 로그는 Firestore에 실시간 저장
  - 감지 카운트도 카메라별로 저장 및 누적 관리
- **비디오 스트리밍 분석**
  - RTSP 또는 비디오 스트림 URL을 통해 실시간 프레임 분석
- **FastAPI 기반 REST API 제공**
  - `/start_analysis`: 감지 시작
  - `/stop_analysis`: 감지 중지
  - `/health`: 서버 상태 확인

---

## 🔗 시스템 구성도

```
┌─────────────┐           ┌────────────────────┐
│  CCTV/RTSP  ├──────────▶│   FastAPI Backend   │
└─────────────┘           │ - YOLOv5 감지       │
                          │ - Firebase 저장     │
                          └────────────────────┘
                                      │
                                      ▼
                          ┌────────────────────┐
                          │   Firebase Firestore│
                          └────────────────────┘
                                      │
                                      ▼
                          ┌────────────────────┐
                          │   Frontend Client   │
                          │ - 실시간 감지 시각화 │
                          └────────────────────┘
```

---

## 🔧 설치 및 실행 방법

### 1. Firebase 인증 정보 설정

`my.json` 이라는 이름으로 Firebase 서비스 계정 키 JSON 파일을 다운로드하여 루트 디렉토리에 위치시켜야 합니다.  
해당 JSON은 [Firebase 콘솔 > 프로젝트 설정 > 서비스 계정 > 새 비공개 키 생성]을 통해 얻을 수 있습니다.

```
project-root/
│
├── main.py
├── my.json       👈 이 파일 필요
├── models/
│   ├── fire.pt
│   ├── FallDown.pt
│   └── safe.pt
```

### 2. YOLO 모델 다운로드

- `yolov5s`는 자동으로 PyTorch Hub에서 다운로드됩니다.
- 나머지 커스텀 모델(`fire.pt`, `FallDown.pt`, `safe.pt`)은 `models/` 폴더에 직접 배치해야 합니다.

### 3. 환경 설정

Python 3.12 버전을 사용하며, 다음 명령어로 가상 환경을 구성하고 의존성을 설치합니다.

```bash
# 가상환경 생성 (선택)
python3.12 -m venv venv
source venv/bin/activate

# 패키지 설치
pip install -r requirements.txt
```

**필수 패키지 주요 버전**

```makefile
torch==2.1.0+cu118  # CUDA 11.8 기준
opencv-python
firebase-admin
fastapi
uvicorn
```

⚠️ CUDA 환경에 맞는 torch wheel을 선택하세요. 위 예시는 CUDA 11.8 기준입니다.

### 🚀 서버 실행

```bash
uvicorn main:app --host 0.0.0.0 --port 5000 --reload
```

위 명령어 실행 후 `/start_analysis` 등 API를 통해 실시간 분석 시작이 가능합니다.
