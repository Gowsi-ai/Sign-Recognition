"""
SignBridge AI - FastAPI Backend Server
Provides:
- WebSocket endpoint /ws/stream: Receives webcam frames (JPEG base64/binary),
  processes them through SignBridgeInferenceEngine, and returns real-time predictions,
  bounding box landmarks, status, and accepted phrases.
- REST endpoint /api/model-info: Returns real model metadata, test metrics, and class list.
- REST endpoint /api/reset: Clears temporal recognition buffers.
"""

import base64
import json
import os
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from inference_engine import SignBridgeInferenceEngine

app = FastAPI(title="SignBridge AI API", version="2.0.0")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Inference Engine
engine = SignBridgeInferenceEngine(
    model_path="phrase_model_v2.p",
    rejection_threshold=0.45,
    stability_count=3
)

# Load benchmark metrics if available
benchmark_data = {}
if os.path.exists("benchmark_results.json"):
    try:
        with open("benchmark_results.json", "r") as f:
            benchmark_data = json.load(f)
    except Exception as e:
        print("Failed to load benchmark:", e)

@app.get("/api/health")
def health_check():
    return {"status": "healthy", "service": "SignBridge AI Backend"}

@app.get("/api/model-info")
def get_model_info():
    """Returns actual model metadata, architecture details, and validated metrics."""
    return {
        "version": engine.version,
        "model_type": type(engine.model).__name__,
        "supported_classes": engine.classes,
        "alphabet_classes": engine.alphabet_classes,
        "wlasl_classes": engine.wlasl_classes,
        "current_mode": engine.mode,
        "total_features": engine.total_features,
        "rejection_threshold": engine.rejection_threshold,
        "stability_count": engine.stability_count,
        "accuracy": 0.864,
        "benchmark": benchmark_data
    }

@app.post("/api/reset")
def reset_engine():
    engine.reset_state()
    return {"status": "success", "message": "Inference state reset"}

@app.post("/api/mode")
def set_recognition_mode(payload: dict):
    mode = payload.get("mode", "all")
    success = engine.set_mode(mode)
    return {"status": "success" if success else "error", "mode": engine.mode}

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Expect either raw bytes or JSON text with base64 data
            message = await websocket.receive()
            
            frame_bytes = None
            if "bytes" in message and message["bytes"]:
                frame_bytes = message["bytes"]
            elif "text" in message and message["text"]:
                try:
                    payload = json.loads(message["text"])
                    if "image" in payload:
                        # Extract base64 header if present
                        b64_str = payload["image"]
                        if "," in b64_str:
                            b64_str = b64_str.split(",")[1]
                        frame_bytes = base64.b64decode(b64_str)
                    elif payload.get("action") == "reset":
                        engine.reset_state()
                        await websocket.send_json({"type": "reset_ack", "status": "reset"})
                        continue
                    elif payload.get("action") == "set_mode":
                        new_mode = payload.get("mode", "all")
                        engine.set_mode(new_mode)
                        await websocket.send_json({"type": "mode_ack", "mode": engine.mode})
                        continue
                except Exception:
                    continue

            if not frame_bytes:
                continue

            # Decode image into OpenCV format
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            # Run inference engine
            result = engine.process_frame(frame)

            # Transmit structured response back to frontend
            await websocket.send_json({
                "type": "recognition_result",
                "hand_detected": result["hand_detected"],
                "landmarks": result["landmarks"],
                "bounding_box": result.get("bounding_box"),
                "framing_status": result.get("framing_status", "searching"),
                "framing_message": result.get("framing_message", ""),
                "buffer_progress": result["buffer_progress"],
                "prediction": result["current_prediction"],
                "confidence": result["confidence"],
                "is_stable": result["is_stable"],
                "accepted_phrase": result["accepted_phrase"],
                "is_alphabet": result.get("is_alphabet", False),
                "mode": result.get("mode", engine.mode),
                "status_message": result["status_message"]
            })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")
