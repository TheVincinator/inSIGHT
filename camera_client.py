"""
Run:
  # Terminal 1
  source venv311/bin/activate
  pip install fastapi uvicorn
  uvicorn fusion_server:app --host 127.0.0.1 --port 8000 --reload

  # Terminal 2
  source venv311/bin/activate
  pip install mediapipe==0.10.14 opencv-python numpy websockets pynput requests
  python camera_client.py

Quit: press 'q' in the OpenCV window or terminate the program in terminal using Ctrl-C
"""

# ================================================================
# camera_client.py  (FULL VERSION — WITH SERVO FACE TRACKING +
#                     FREEZE LAST EYE METRICS WHEN FACE LOST)
# ================================================================

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import websockets
import requests 

stress_score = 0.0

# ------------------------------------------------
# Tunables
# ------------------------------------------------
WS_URL = "ws://127.0.0.1:8000/ws"
SEND_HZ = 1.0
WINDOW_SEC = 15.0
MIN_FACE_CONF = 0.5
MIN_TRACK_CONF = 0.5

CAM_WIDTH = 640
CAM_HEIGHT = 480
CAMERA_INDEX = 0
CAP_BACKEND = cv2.CAP_AVFOUNDATION

BASELINE_SEC = 15.0

EAR_BASELINE_SEC = 2.0
EAR_THRESH_FRAC = 0.75
EAR_FALLBACK_THRESH = 0.21

PUPIL_OPEN_MARGIN = 0.02
PUPIL_MEDIAN_N = 5

# =========================================================
# SERVO SETTINGS
# =========================================================
ESP32_SERVO_URL = "http://10.48.126.77/servo"

DEAD_ZONE = 0.08
SERVO_GAIN = 0.6
SERVO_ALPHA = 0.25
SERVO_SEND_HZ = 10

servo_filtered = 0.0
last_servo_send = 0.0

def send_servo_command(speed):
    try:
        requests.get(
            ESP32_SERVO_URL,
            params={"pan": float(speed)},
            timeout=0.02,
        )
    except:
        pass

# =========================================================
# Freeze last valid eye metrics (NEW)
# =========================================================
last_valid_eye_metrics = None

# ------------------------------------------------
# Landmarks
# ------------------------------------------------
L_EYE = {"p1":33,"p4":133,"p2":160,"p6":144,"p3":158,"p5":153}
R_EYE = {"p1":362,"p4":263,"p2":387,"p6":373,"p3":385,"p5":380}

NOSE_TIP_IDX = 1
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 362

# ------------------------------------------------
# Helpers
# ------------------------------------------------
def _pt(landmarks, idx, w, h):
    lm = landmarks[idx]
    return np.array([lm.x*w, lm.y*h], dtype=np.float32)

def ear_from_eye(landmarks, eye_idx_map, w, h):
    p1=_pt(landmarks,eye_idx_map["p1"],w,h)
    p2=_pt(landmarks,eye_idx_map["p2"],w,h)
    p3=_pt(landmarks,eye_idx_map["p3"],w,h)
    p4=_pt(landmarks,eye_idx_map["p4"],w,h)
    p5=_pt(landmarks,eye_idx_map["p5"],w,h)
    p6=_pt(landmarks,eye_idx_map["p6"],w,h)

    v1=np.linalg.norm(p2-p6)
    v2=np.linalg.norm(p3-p5)
    hdist=np.linalg.norm(p1-p4)+1e-6
    return float((v1+v2)/(2.0*hdist))

# ------------------------------------------------
# Window Stats
# ------------------------------------------------
@dataclass
class WindowStats:
    samples:Deque[Tuple[float,bool,int,float]]
    def __init__(self):
        self.samples=deque()
    def prune(self,now,window_sec):
        while self.samples and (now-self.samples[0][0]>window_sec):
            self.samples.popleft()
    def add(self,ts,is_closed,blink_event,motion_norm):
        self.samples.append((ts,is_closed,blink_event,motion_norm))
    def blink_rate_per_min(self):
        if len(self.samples)<2:return 0.0
        t0=self.samples[0][0];t1=self.samples[-1][0]
        dt=max(1e-6,t1-t0)
        blinks=sum(s[2] for s in self.samples)
        return float(blinks*60.0/dt)
    def perclos(self):
        if not self.samples:return 0.0
        closed=sum(1 for s in self.samples if s[1])
        return float(closed/len(self.samples))
    def head_motion_var(self):
        if len(self.samples)<3:return 0.0
        m=np.array([s[3] for s in self.samples],dtype=np.float32)
        return float(np.var(m))

# ------------------------------------------------
# Receive loop
# ------------------------------------------------
async def receive_loop(ws):
    global stress_score
    while True:
        try:
            raw=await ws.recv()
            msg=json.loads(raw)
            if msg.get("type")=="stress_score":
                stress_score=msg.get("value",0.0)
        except:
            break

# ================================================================
# CAMERA LOOP
# ================================================================
async def camera_loop():
    global servo_filtered,last_servo_send,last_valid_eye_metrics

    cap=cv2.VideoCapture(CAMERA_INDEX,CAP_BACKEND)
    if not cap.isOpened():
        raise RuntimeError("Camera failed to open.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,CAM_HEIGHT)

    mp_face_mesh=mp.solutions.face_mesh
    mesh=mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=MIN_FACE_CONF,
        min_tracking_confidence=MIN_TRACK_CONF,
    )

    win=WindowStats()
    last_send=0.0

    was_closed=False
    blink_armed=False

    ear_baseline_samples=[]
    ear_baseline_start=None
    ear_thresh=EAR_FALLBACK_THRESH

    prev_nose_xy=None

    async with websockets.connect(WS_URL) as ws:
        asyncio.create_task(receive_loop(ws))

        while True:
            ok,frame=cap.read()
            if not ok:
                await asyncio.sleep(0.01)
                continue

            frame=cv2.flip(frame,1)
            h,w=frame.shape[:2]
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            res=mesh.process(rgb)
            now=time.time()

            blink_event=0
            ear=None
            is_closed=False
            motion_norm=0.0
            face_detected=False

            if res.multi_face_landmarks:
                face_detected=True
                face=res.multi_face_landmarks[0]
                lms=face.landmark

                ear_l=ear_from_eye(lms,L_EYE,w,h)
                ear_r=ear_from_eye(lms,R_EYE,w,h)
                ear=(ear_l+ear_r)/2.0

                if ear_baseline_start is None:
                    ear_baseline_start=now
                if (now-ear_baseline_start)<=EAR_BASELINE_SEC:
                    ear_baseline_samples.append(ear)
                if (now-ear_baseline_start)>EAR_BASELINE_SEC and len(ear_baseline_samples)>=10:
                    open_med=float(np.median(ear_baseline_samples))
                    ear_thresh=max(0.10,EAR_THRESH_FRAC*open_med)

                is_closed=ear<ear_thresh

                nose_xy=_pt(lms,NOSE_TIP_IDX,w,h)

                # ============================
                # SERVO FACE CENTERING
                # ============================
                frame_center_x=w/2
                face_error=(nose_xy[0]-frame_center_x)/frame_center_x

                if abs(face_error)<DEAD_ZONE:
                    face_error=0.0

                target_speed=SERVO_GAIN*face_error
                servo_filtered=SERVO_ALPHA*target_speed+(1-SERVO_ALPHA)*servo_filtered
                servo_speed=servo_filtered

                if face_error!=0.0 and (now-last_servo_send)>(1.0/SERVO_SEND_HZ):
                    send_servo_command(servo_speed)
                    last_servo_send=now

                l_outer=_pt(lms,LEFT_EYE_OUTER,w,h)
                r_outer=_pt(lms,RIGHT_EYE_OUTER,w,h)
                face_scale=float(np.linalg.norm(l_outer-r_outer)+1e-6)

                if prev_nose_xy is not None:
                    motion_norm=float(np.linalg.norm(nose_xy-prev_nose_xy)/face_scale)
                prev_nose_xy=nose_xy

                if not was_closed and is_closed:
                    blink_armed=True
                elif was_closed and not is_closed and blink_armed:
                    blink_event=1
                    blink_armed=False
                was_closed=is_closed

                win.add(now,is_closed,blink_event,motion_norm)
                cv2.circle(frame,(int(nose_xy[0]),int(nose_xy[1])),3,(0,255,0),-1)

            else:
                blink_armed=False
                was_closed=False
                prev_nose_xy=None
                cv2.putText(frame,"Face not detected",(10,25),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

            win.prune(now,WINDOW_SEC)

            blink_rate=win.blink_rate_per_min()
            perclos=win.perclos()
            head_var=win.head_motion_var()

            # =====================================================
            # FREEZE LAST VALID EYE METRICS (NEW)
            # =====================================================
            current_eye_metrics={
                "blink_rate_per_min":blink_rate,
                "perclos":perclos,
                "head_motion_var":head_var,
                "ear":ear,
                "ear_thresh":ear_thresh,
                "face_detected":face_detected,
                "window_sec":WINDOW_SEC,
                "camera_index":CAMERA_INDEX,
            }

            if face_detected:
                last_valid_eye_metrics=current_eye_metrics

            send_metrics=last_valid_eye_metrics if last_valid_eye_metrics else current_eye_metrics

            # =====================================================

            cv2.putText(frame,"STRESS",(20,35),
                        cv2.FONT_HERSHEY_DUPLEX,0.5,(200,200,200),1)
            cv2.putText(frame,f"{stress_score:.0f}",(20,70),
                        cv2.FONT_HERSHEY_DUPLEX,1.1,(0,255,120),2)

            cv2.imshow("camera_client",frame)
            if cv2.waitKey(1)&0xFF==ord("q"):
                break

            if now-last_send>=(1.0/SEND_HZ):
                payload={
                    "type":"eye_metrics",
                    "data":send_metrics,
                    "timestamp":now,
                }
                await ws.send(json.dumps(payload))
                last_send=now

            await asyncio.sleep(0)

    cap.release()
    cv2.destroyAllWindows()

# ================================================================
if __name__=="__main__":
    asyncio.run(camera_loop())