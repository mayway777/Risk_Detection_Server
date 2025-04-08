import torch
import torch.amp
import cv2
import asyncio
import time
import firebase_admin
from firebase_admin import credentials, firestore
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn
from datetime import datetime
import logging
import numpy as np

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('video_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Firebase 초기화
try:
    firebase_admin.get_app()
except ValueError:
    cred = credentials.Certificate("my.json")
    firebase_admin.initialize_app(cred)
db = firestore.client()

# 데이터 안전 변환 함수
def safe_convert(value):
    """
    Firestore에 저장 가능한 형태로 값 변환
    """
    # PyTorch 텐서인 경우 float나 int로 변환
    if torch.is_tensor(value):
        return float(value.cpu().detach().numpy())
    
    # NumPy 배열인 경우 리스트로 변환
    if isinstance(value, np.ndarray):
        return value.tolist()
    
    # 리스트나 튜플의 경우 재귀적으로 변환
    if isinstance(value, (list, tuple)):
        return [safe_convert(item) for item in value]
    
    # 딕셔너리의 경우 재귀적으로 변환
    if isinstance(value, dict):
        return {k: safe_convert(v) for k, v in value.items()}
    
    # 그 외의 경우 그대로 반환
    return value

# 모델 및 상태 관리 클래스
class HazardDetectionTracker:
    def __init__(self, threshold_seconds=0):
        self.detection_start_time = None
        self.threshold_seconds = threshold_seconds
        self.total_detection_count = 0
        self.last_detection_details = None
        self.is_currently_detected = False

    def update(self, is_detected, current_time, additional_info=None):
        detection_result = {
            'count_increased': False,
            'details': None
        }

        if is_detected:
            # 임계 시간이 0인 경우 (화재, 낙상, 안전 문제) 즉시 카운트
            if self.threshold_seconds == 0:
                self.total_detection_count += 1
                detection_result['count_increased'] = True
                detection_result['details'] = {
                    'timestamp': datetime.now().isoformat(),
                    'duration': 0,
                    'additional_info': additional_info
                }
                return detection_result

            # 차량 감지의 경우 (2초 이상)
            if not self.is_currently_detected:
                self.detection_start_time = current_time
                self.is_currently_detected = True
            
            duration = current_time - self.detection_start_time
            
            if duration >= self.threshold_seconds:
                self.total_detection_count += 1
                detection_result['count_increased'] = True
                detection_result['details'] = {
                    'timestamp': datetime.now().isoformat(),
                    'duration': duration,
                    'additional_info': additional_info
                }
                self.last_detection_details = detection_result['details']
        else:
            self.detection_start_time = None
            self.is_currently_detected = False

        return detection_result

class Models:
    car_model = None
    fall_model = None
    fire_model = None
    safe_model = None
    
    # 각 감지 유형별 트래커
    car_tracker = HazardDetectionTracker(threshold_seconds=2)
    fire_tracker = HazardDetectionTracker()
    fall_tracker = HazardDetectionTracker()
    safety_tracker = HazardDetectionTracker()

    @classmethod
    async def load_models(cls):
        logger.info("YOLO 모델 로딩 시작...")
        start_time = time.time()
        
        try:
            
            cls.car_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
            cls.fall_model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/FallDown.pt')
            cls.fire_model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/fire.pt')
            cls.safe_model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/safe.pt')
            
            end_time = time.time()
            logger.info(f"YOLO 모델 로딩 완료. 총 소요 시간: {end_time - start_time:.2f}초")
        except Exception as e:
            logger.error(f"모델 로딩 중 오류 발생: {e}")
            raise

# 카메라별 위험 감지 유형별 카운트 업데이트
async def update_camera_hazard_count(user_id, camera_id, detection_type):
    """카메라별 위험 감지 유형별 카운트 업데이트"""
    camera_hazard_count_ref = (
        db.collection('users').document(user_id)
           .collection('cameras').document(camera_id)
           .collection('total_hazard_counts').document('counts')
    )

    # 감지 유형별 카운트 매핑
    count_fields = {
        'car_detection': 'car_detection_count',
        'fire_detection': 'fire_detection_count',
        'fall_detection': 'fall_detection_count',
        'safety_issues': 'safety_issue_count'
    }

    update_field = count_fields.get(detection_type)
    if update_field:
        await asyncio.to_thread(
            camera_hazard_count_ref.set,
            {update_field: firestore.Increment(1)},
            merge=True
        )

# 감지 시간 저장 함수
async def save_detection_time(user_id, camera_id, detection_type):
    """감지된 시간 저장"""
    detection_ref = (
        db.collection('users').document(user_id)
           .collection('cameras').document(camera_id)
           .collection(detection_type)
    )

    detection_doc = {
        'timestamp': datetime.now()
    }

    await asyncio.to_thread(detection_ref.add, detection_doc)

# 프레임 분석 함수
async def analyze_frame(frame, camera_id, user_id):
    current_time = time.time()
    detection_results = {
        'car_detection': {},
        'fire_detection': {},
        'fall_detection': {},
        'safety_issues': {}
    }

    try:
        # 1. 차량 감지
        car_results = Models.car_model(frame)
        cars = car_results.pandas().xyxy[0]
        confident_cars = cars[
            (cars['class'] == 2) &  # class 2: car
            (cars['confidence'] >= 0.7)
        ]
        
        car_detections = confident_cars.values.tolist()
        
        car_detection_result = Models.car_tracker.update(len(car_detections) > 0, current_time, {
            'car_count': len(car_detections),
            'bounding_boxes': car_detections
        })
        detection_results['car_detection'] = {
            'count': Models.car_tracker.total_detection_count,
            'currently_detected': len(car_detections) > 0,
            'details': car_detection_result['details'] if car_detection_result['count_increased'] else None
        }

        # 2. 화재 감지
        with torch.amp.autocast('cuda'):
            fire_results = Models.fire_model(frame)
            fire_detections = [box for *box, conf, cls in fire_results.xyxy[0] if conf > 0.75]
            fire_detected = len(fire_detections) > 0
            
            fire_detection_result = Models.fire_tracker.update(fire_detected, current_time, {
                'confidence': max([conf for *box, conf, cls in fire_results.xyxy[0] if conf > 0.75]) if fire_detected else None,
                'bounding_boxes': fire_detections
            })
            detection_results['fire_detection'] = {
                'count': Models.fire_tracker.total_detection_count,
                'currently_detected': fire_detected,
                'details': fire_detection_result['details'] if fire_detection_result['count_increased'] else None
            }

        # 3. 낙상 감지
        fall_results = Models.fall_model(frame)
        fall_detections = [box for *box, conf, cls in fall_results.xyxy[0] if conf > 0.65]
        fall_detected = len(fall_detections) > 0
        
        fall_detection_result = Models.fall_tracker.update(fall_detected, current_time, {
            'confidence': max([conf for *box, conf, cls in fall_results.xyxy[0] if conf > 0.65]) if fall_detected else None,
            'bounding_boxes': fall_detections
        })
        detection_results['fall_detection'] = {
            'count': Models.fall_tracker.total_detection_count,
            'currently_detected': fall_detected,
            'details': fall_detection_result['details'] if fall_detection_result['count_increased'] else None
        }

        # 4. 안전 문제 감지
        safe_results = Models.safe_model(frame)
        unsafe_objects = [
            (*box, class_name, conf) 
            for *box, conf, cls in safe_results.xyxy[0] 
            if (conf > 0.65 and 
                (class_name := safe_results.names[int(cls)]) not in 
                ["chair", "car", "person", "truck", "motorcycle", "dining table", "airplane", "potted plant"])
        ]
        safety_detected = len(unsafe_objects) > 0
        
        safety_detection_result = Models.safety_tracker.update(safety_detected, current_time, {
            'objects': [
                {
                    'class_name': class_name,
                    'confidence': float(conf),
                    'bbox': box
                } for *box, class_name, conf in unsafe_objects
            ]
        })
        detection_results['safety_issues'] = {
            'count': Models.safety_tracker.total_detection_count,
            'currently_detected': safety_detected,
            'details': safety_detection_result['details'] if safety_detection_result['count_increased'] else None
        }

        # 각 감지 유형에 대해 처리
        for detection_type, detection_info in detection_results.items():
            if detection_info.get('details'):
                # 감지된 시간 저장
                await save_detection_time(user_id, camera_id, detection_type)
                
                # 해당 유형의 카운트 증가
                await update_camera_hazard_count(user_id, camera_id, detection_type)

    except Exception as e:
        logger.error(f"프레임 분석 중 오류 발생: {e}")
        logger.error(str(e), exc_info=True)

    return detection_results

# 비디오 스트림 처리 함수 (이전 코드와 동일)
async def process_video_stream(user_id: str, camera_id: str, url: str):
    """비디오 스트림 처리"""
    logger.info(f"카메라 {camera_id} 스트림 분석 시작 - URL: {url}")
    
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        error_msg = f"카메라 스트림 열기 실패: {url}"
        logger.error(error_msg)
        raise Exception(error_msg)

    last_update_time = time.time()
    update_interval = 0.5  # 0.5초 간격으로 업데이트
    frame_count = 0

    try:
        while not stop_flags.get(camera_id, True):
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"카메라 {camera_id} 프레임 읽기 실패")
                break

            current_time = time.time()
            if current_time - last_update_time >= update_interval:
                # 프레임 분석
                await analyze_frame(frame, camera_id, user_id)

                frame_count += 1
                last_update_time = current_time

            await asyncio.sleep(0.1)  # CPU 부하 감소

        logger.info(f"카메라 {camera_id} 스트림 분석 종료. 총 처리된 프레임: {frame_count}")

    except Exception as e:
        logger.error(f"카메라 {camera_id} 스트림 처리 중 오류 발생: {e}")
    finally:
        cap.release()
        if camera_id in analysis_tasks:
            del analysis_tasks[camera_id]
        if camera_id in stop_flags:
            del stop_flags[camera_id]

# Pydantic 모델, FastAPI 앱 설정 등 나머지 코드는 이전과 동일

# Pydantic 모델
class Camera(BaseModel):
    id: str
    url: str

class AnalysisRequest(BaseModel):
    userId: str
    cameras: List[Camera]

class AnalysisResponse(BaseModel):
    status: str
    message: str

# FastAPI 앱 초기화
app = FastAPI(title="Video Analysis API")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 분석 상태 관리
analysis_tasks = {}
stop_flags = {}

# 앱 시작 시 모델 로드
@app.on_event("startup")
async def startup_event():
    """앱 시작 시 모델 로드"""
    await Models.load_models()

# 분석 시작 엔드포인트
@app.post("/start_analysis", response_model=AnalysisResponse)
async def start_analysis(request: AnalysisRequest):
    """분석 시작 엔드포인트"""
    try:
        for camera in request.cameras:
            if camera.id not in analysis_tasks:
                stop_flags[camera.id] = False
                task = asyncio.create_task(
                    process_video_stream(
                        request.userId,
                        camera.id,
                        camera.url
                    )
                )
                analysis_tasks[camera.id] = task

        return AnalysisResponse(
            status="success",
            message=f"{len(request.cameras)}개 카메라 분석이 시작되었습니다."
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 분석 중지 엔드포인트
@app.post("/stop_analysis", response_model=AnalysisResponse)
async def stop_analysis(request: AnalysisRequest):
    """분석 중지 엔드포인트"""
    try:
        for camera in request.cameras:
            if camera.id in stop_flags:
                stop_flags[camera.id] = True

        # 모든 분석 태스크 종료 대기
        if analysis_tasks:
            await asyncio.gather(*analysis_tasks.values())

        # 상태 초기화
        analysis_tasks.clear()
        stop_flags.clear()

        return AnalysisResponse(
            status="success",
            message="모든 카메라 분석이 중지되었습니다."
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 상태 확인 엔드포인트
@app.get('/health')
async def health_check():
    return {'status': 'ok'}

# 서버 실행
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)