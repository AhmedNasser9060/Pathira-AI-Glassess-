# Pathira Backend

FastAPI + async SQLAlchemy + Supabase Postgres.

## Quick start

```bash
cd backend
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

`.env` lives in `backend/.env` (see committed file for required keys).

## Server-side ML

ML inference runs in this backend, lazy-loaded on first request. Each
capability is added one phase at a time. Model weights are NEVER committed —
they live under `backend/ml_weights/` (gitignored).

### Phase 1 — OCR (Tesseract)

Tesseract is an OS-level binary; install it on the host before starting the
server. **Do not** vendor the binary into the repo.

| OS | Install command |
|---|---|
| Linux (Debian/Ubuntu) | `sudo apt-get install tesseract-ocr tesseract-ocr-ara` |
| macOS (Homebrew) | `brew install tesseract tesseract-lang` |
| Windows | Official UB-Mannheim installer: <https://github.com/UB-Mannheim/tesseract/wiki>. During the installer's *Additional language data* step, tick **English** and **Arabic**. |

After installing, confirm `tesseract --version` works in a fresh shell.

#### Environment variables (set in `backend/.env`)

| Variable | Default | Purpose |
|---|---|---|
| `OCR_TESSERACT_CMD` | unset (uses `PATH`) | Absolute path to `tesseract` / `tesseract.exe`. Set this only if the binary is not on `PATH` (e.g. `C:\Program Files\Tesseract-OCR\tesseract.exe`). |
| `OCR_MAX_IMAGE_BYTES` | `5242880` (5 MB) | Multipart upload size cap. Larger payloads → HTTP 413. |
| `ML_WEIGHTS_DIR` | `backend/ml_weights` | Where future phases (YOLO, ArcFace) load weights from. Phase 1 doesn't use it. |

#### Endpoint

```
POST /api/vision/ocr
  multipart: image=<file>
  query:     lang=eng+ara   (default; e.g. eng, ara, eng+ara)
  auth:      Bearer <access_token>
  -> 200 { "text": "...", "event_id": "<uuid>" }
  -> 413 if image > OCR_MAX_IMAGE_BYTES
```

The endpoint runs `pytesseract` in a worker thread via `asyncio.to_thread`
and persists a `vision_detections` row with `summary='ocr'` and
`detections=[{"text": "..."}]`.

#### curl smoke test

```bash
TOKEN=$(curl -s -X POST http://localhost:8000/api/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"email":"you@example.com","password":"..."}' | jq -r .access_token)

curl -X POST http://localhost:8000/api/vision/ocr \
  -H "Authorization: Bearer $TOKEN" \
  -F "image=@/path/to/page.jpg"
```

### Phase 2 — Object detection (server-side YOLO)

Inference runs on Ultralytics YOLO. Weights live at
`backend/ml_weights/yolo.pt` (gitignored). Drop in a YOLOv8/YOLO11 `.pt`
of your choice — `yolov8n.pt` (~6 MB) is the recommended default for CPU
inference.

```bash
# Download the official Ultralytics yolov8n weights:
curl -L -o backend/ml_weights/yolo.pt \
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt
```

The first request after server boot triggers `ultralytics.YOLO(...)` to
load the weights and warm torch — expect a 2–5 s latency on the first
call, then ~80–200 ms per 640×480 frame on CPU.

#### Endpoint

`POST /api/vision/detect` now accepts EITHER:

| Content-Type | Body | Behavior |
|---|---|---|
| `application/json` | `VisionDetectRequest` (existing on-device pipeline) | logs as-is. Unchanged. |
| `multipart/form-data` | field `image=<file>` | runs YOLO server-side, persists detections in the same row shape, returns `{logged, event_id}`. |

```bash
curl -X POST http://localhost:8000/api/vision/detect \
  -H "Authorization: Bearer $TOKEN" \
  -F "image=@./scene.jpg"
# -> {"logged":true,"event_id":"<uuid>"}
```

The 5 MB cap (`OCR_MAX_IMAGE_BYTES`) is reused for the multipart branch.

### Phase 3 — Face recognition (ArcFace ONNX)

Face detection uses OpenCV's bundled Haar cascade — no extra detector to
install or download. The 512-d embedding comes from an ArcFace ResNet-100
ONNX model running under `onnxruntime`. Cosine similarity against
`faces.face_embedding` rows is computed in Python (no pgvector yet).

Place ArcFace weights at `backend/ml_weights/arcface.onnx` (~249 MB,
gitignored). Any standard ArcFace ONNX from InsightFace's model-zoo
works.

#### Settings

| Variable | Default | Purpose |
|---|---|---|
| `FACE_MATCH_THRESHOLD` | `0.6` | Minimum cosine similarity for a positive match. Recognize requests below this never trigger an Alert. |

#### Endpoints

`POST /api/companion/faces` and `POST /api/companion/faces/recognize`
both now accept EITHER `application/json` (existing on-device pipeline,
unchanged) OR `multipart/form-data`:

```bash
# Register Mom from a photo:
curl -X POST http://localhost:8000/api/companion/faces \
  -H "Authorization: Bearer $COMPANION_TOKEN" \
  -F "user_id=<main-user-uuid>" \
  -F "name=Mom" \
  -F "relationship=parent" \
  -F "image=@./mom.jpg"
# -> 201 FaceOut (with embedding populated)

# Recognize someone in a photo against the main user's face library:
curl -X POST http://localhost:8000/api/companion/faces/recognize \
  -H "Authorization: Bearer $COMPANION_TOKEN" \
  -F "user_id=<main-user-uuid>" \
  -F "image=@./visitor.jpg"
# Match:
# {"logged":true,"alert_created":true,"alert_id":"<uuid>",
#  "matched_face_id":"<uuid>","matched_name":"Mom","similarity":0.87}
# No match (below threshold):
# {"logged":true,"alert_created":false,"alert_id":null,
#  "matched_face_id":null,"matched_name":null,"similarity":0.41}
```

If no face is detected in the upload, both endpoints return HTTP 422.
The 5 MB cap (`OCR_MAX_IMAGE_BYTES`) is reused for the multipart branch.

### Phase 5 — Obstacle detection (server-side YOLO + voice guidance)

Reuses Phase 2's stack (`ultralytics`, `torch`, `opencv-python-headless`,
`numpy`) — **no new pip install**. The custom-trained obstacle model is
loaded from `backend/ml_weights/obstacle.pt` (gitignored) via a separate
lazy registry entry (`get_obstacle_yolo`) so it coexists with the general
YOLO model from Phase 2.

```bash
# 39 MB. Drop in your own custom-trained obstacle YOLOv8 .pt here.
cp /path/to/obstacle.pt backend/ml_weights/obstacle.pt
```

#### Endpoint

```
POST /api/vision/obstacle
  multipart:  image=<file>
  query:      conf_threshold=0.25  (0.0..1.0)
  query:      top_n=3              (1..10, count of voice-guidance lines)
  auth:       Bearer <access_token>
```

```bash
curl -X POST 'http://localhost:8000/api/vision/obstacle?top_n=3' \
  -H "Authorization: Bearer $TOKEN" \
  -F "image=@./scene.jpg"
```

Response:

```json
{
  "logged": true,
  "event_id": "<uuid>",
  "detections": [
    {"label":"person","confidence":0.91,"bbox":[10,20,100,200],
     "is_critical":true,"priority_score":0.74}
  ],
  "highest_priority": {...},
  "voice_guidance": ["Person close on your left"],
  "image_shape": {"width":1920,"height":1080}
}
```

Priority is heuristic: bigger box + lower in frame + critical class +
higher confidence → higher score. Voice guidance is generated from the
top-N detections (by priority); empty path returns `["Path is clear"]`.

A `vision_detections` row is persisted with `summary='obstacle'` and the
detection list. The same payload is bridged to the user's WS feed with
`kind="obstacle"` plus an extra `voice_guidance` field.

The 5 MB cap (`OCR_MAX_IMAGE_BYTES`) is reused for the multipart branch.

### Scene description (BLIP-base + MarianMT EN→AR)

The original Colab notebook used BLIP-2 (~15 GB, GPU-only). For the
backend we use the lighter `Salesforce/blip-image-captioning-base` (~1 GB)
which runs in ~2-5 s on CPU, plus `Helsinki-NLP/opus-mt-en-ar` (~300 MB)
for the EN→AR translation. Both are downloaded from Hugging Face on
first call and cached under `~/.cache/huggingface/`. **No weights are
committed; first request to `/api/vision/scene` after a clean install
will pause for ~1-2 minutes while the model downloads.**

TTS is intentionally not server-side. The Expo app already has device
TTS (`expo-speech`) which is faster and works offline; the endpoint
returns the Arabic text and the client speaks it.

#### Endpoint

```
POST /api/vision/scene
  multipart:  image=<file>
  query:      translate=true   (default; set false to skip MarianMT)
  auth:       Bearer <access_token>
  rate-limit: 10/minute (BLIP is heavier than the other vision routes)
```

```bash
curl -X POST 'http://localhost:8000/api/vision/scene' \
  -H "Authorization: Bearer $TOKEN" \
  -F "image=@./scene.jpg"
# -> {"logged":true, "event_id":"<uuid>",
#     "caption_en":"a desk with a laptop and a keyboard",
#     "caption_ar":"مكتب عليه حاسوب محمول ولوحة مفاتيح"}
```

A `vision_detections` row is persisted with `summary='scene'` and the
captions, and the same payload is bridged via the WS feed with
`kind='scene'`.

## Tests

```bash
pytest backend/tests
```

Tests mock all ML calls — no real weights are loaded in pytest.
