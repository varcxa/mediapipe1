#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
import cv2
import mediapipe as mp

# ===== 설정 =====
video_path = sys.argv[1] if len(sys.argv) > 1 else "face.mp4"
if not os.path.isabs(video_path):
    video_path = os.path.join(os.getcwd(), video_path)

print(f"[INFO] 사용 영상: {video_path}")

# ===== MediaPipe 초기화 =====
mp_face = mp.solutions.face_detection
detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# ===== 비디오 열기 =====
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("[ERROR] 동영상을 열 수 없습니다. 경로/파일을 확인하세요.")
    sys.exit(1)

print("[INFO] 재생 시작 — ESC 또는 q 로 종료")

while True:
    ok, frame = cap.read()
    if not ok:
        print("[INFO] 더 이상 프레임이 없습니다. 종료합니다.")
        break

    # BGR -> RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 얼굴 탐지
    res = detector.process(rgb)

    # 박스/키포인트 그리기
    if res.detections:
        h, w = frame.shape[:2]
        for d in res.detections:
            box = d.location_data.relative_bounding_box
            x, y = int(box.xmin * w), int(box.ymin * h)
            ww, hh = int(box.width * w), int(box.height * h)
            cv2.rectangle(frame, (x, y), (x+ww, y+hh), (0, 255, 0), 2)
            for kp in d.location_data.relative_keypoints:
                cx, cy = int(kp.x * w), int(kp.y * h)
                cv2.circle(frame, (cx, cy), 3, (255, 0, 255), -1)

    cv2.imshow("MediaPipe Face (Video)", frame)
    key = cv2.waitKey(20) & 0xFF
    if key in (27, ord('q')):  # ESC 혹은 q
        print("[INFO] 사용자 종료")
        break

cap.release()
cv2.destroyAllWindows()
