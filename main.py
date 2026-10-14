"""
Blind-navigation assistant v3.3 (hackathon demo)

Reflex layer : phone cam -> YOLO + ByteTrack + time-to-collision -> beep/speech (interrupts everything)
Scene layer  : Gemma3 (Ollama), grounded with YOLO output, streamed, gated, paused during danger
Voice Q&A    : press 'a', speak for 5 s -> Whisper -> VLM -> spoken answer
Audio        : Windows SAPI via pywin32 (instant interrupt). You should hear "Audio ready" at startup.

Keys (click the video window first!):
  d = describe scene now | a = ask question (voice) | c = toggle local/cloud
  s = auto-scene on/off  | q = quit

Setup:
  pip install ultralytics "opencv-python<5" faster-whisper pyttsx3 pywin32 sounddevice requests numpy
  ollama pull gemma3:4b
  Camera (best): adb forward tcp:8080 tcp:8080  -> SOURCE below works as-is
  Optional cloud: set GEMINI_API_KEY=your_key
"""
import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rw_timeout;5000000"   # 5 s network timeout

import cv2, json, re, threading, time, queue, base64, requests, collections, traceback
import numpy as np
from ultralytics import YOLO

# ------------------------------- CONFIG -------------------------------
SOURCE = "http://10.79.24.182:8080/video"     # USB via adb. WiFi: http://PHONE_IP:8080/video
YOLO_WEIGHTS = "yolo11n.pt"                 # nano = safest. Try "yolo11s.pt" once stable
YOLO_IMGSZ = 416                            # on CPU fallback, try 320
VLM_MODEL = "gemma3:4b"                     # must match `ollama list` (auto-corrected if close)
OLLAMA_URL = "http://localhost:11434"
OLLAMA_NUM_GPU = None                       # None = let Ollama decide
CLOUD_MODEL = "gemini-2.0-flash"            # change if AI Studio lists a newer Flash name
BACKEND = "local"                           # "local" or "cloud" (toggle with 'c')

SCENE_MIN_GAP = 15      # min seconds between automatic scene descriptions
DANGER_PAUSE = 5        # no auto-scenes this many seconds after a danger alert
SCENE_THRESH = 14       # scene-change sensitivity (lower = more sensitive)

# Big things you can walk into. Alert when they are large in the walking corridor.
ALERT_CLASSES = {
    # people and vehicles
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "train",
    # animals (cows, dogs, goats on Indian streets)
    "dog", "cat", "cow", "horse", "sheep", "elephant", "bear",
    # street furniture
    "bench", "fire hydrant", "stop sign", "parking meter", "traffic light",
    # indoor obstacles
    "chair", "couch", "bed", "dining table", "potted plant", "tv", "refrigerator",
    "toilet", "sink", "oven",
}
# Small things on the floor you can trip over. Alert only when close to your feet
# (bottom of the box near the bottom of the frame).
SMALL_HAZARDS = {"suitcase", "backpack", "handbag", "skateboard", "sports ball",
                 "bottle", "umbrella", "cup", "bowl"}
# Nicer spoken names
SPOKEN = {"dining table": "table", "fire hydrant": "hydrant", "potted plant": "plant",
          "sheep": "goat or sheep", "motorcycle": "motorbike", "tv": "television",
          "refrigerator": "fridge", "parking meter": "pole", "traffic light": "pole",
          "stop sign": "sign pole", "sports ball": "ball", "handbag": "bag"}

FRAME, FID = None, 0
lock = threading.Lock()
vlm_ready = threading.Event()               # set once Gemma has loaded (or failed)
state = {"fps": 0.0, "vlm": "", "vlm_ms": 0, "status": "starting", "last": time.time(),
         "annot": None, "danger": "", "objects": "", "last_danger": 0.0, "last_urgent": 0.0,
         "auto_scene": True, "cam_alerted": False, "device": "?", "vlm_busy": False}

# ------------------------------- AUDIO -------------------------------
speech_q = queue.PriorityQueue()   # (priority, timestamp, text) 0=danger 1=answer 2=scene

def say(text, prio):
    if prio == 0:
        keep = []                                   # drop queued lower-priority speech
        while True:
            try: item = speech_q.get_nowait()
            except queue.Empty: break
            if item[0] == 0: keep.append(item)
        for item in keep: speech_q.put(item)
    speech_q.put((prio, time.time(), text))

def beep():
    try:
        import winsound
        threading.Thread(target=winsound.Beep, args=(1200, 120), daemon=True).start()
    except Exception:
        print("\a", end="")

def speaker_pyttsx3():
    """Fallback only (cannot interrupt mid-sentence)."""
    import pyttsx3
    while True:
        prio, ts, text = speech_q.get()
        if prio > 0 and time.time() - ts > 12:
            continue
        try:
            eng = pyttsx3.init(); eng.setProperty("rate", 185)
            eng.say(text); eng.runAndWait()
        except Exception as e:
            print("TTS error:", e)

def speaker():
    try:
        import pythoncom, win32com.client
        pythoncom.CoInitialize()
        voice = win32com.client.Dispatch("SAPI.SpVoice")
        voice.Rate = 2                              # -10 (slow) .. 10 (fast)
        voice.Volume = 100
        voice.Speak("Audio ready")                  # blocking self-test: you should hear this
        print("Audio: SAPI ready")
    except Exception as e:
        print(f"SAPI unavailable ({e}). Run: pip install pywin32 . Falling back to pyttsx3.")
        return speaker_pyttsx3()
    while True:
        try: item = speech_q.get(timeout=0.05)
        except queue.Empty: continue
        prio, ts, text = item
        if prio > 0 and time.time() - ts > 12:      # stale speech is dropped
            continue
        try:
            if prio == 0:
                voice.Speak(text, 3)                # 3 = async + purge: cuts current speech instantly
            elif voice.WaitUntilDone(0):            # idle: speak
                voice.Speak(text, 1)                # 1 = async
            else:                                   # busy: wait, but let danger jump the queue
                speech_q.put(item); time.sleep(0.05)
        except Exception as e:
            print("TTS error:", e)

# ------------------------------- CAMERA -------------------------------
def grabber():
    global FRAME, FID
    warned = False
    while True:
        cap = cv2.VideoCapture(SOURCE, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap.release()                           # never leak captures
            state["status"] = "camera unreachable - check IP Webcam / adb forward"
            if not warned:
                print(f"Cannot open {SOURCE}. Open it in a browser; if that fails, fix the network "
                      f"or run: adb forward tcp:8080 tcp:8080 and use 127.0.0.1")
                warned = True
            time.sleep(2)
            continue
        warned = False
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        try:
            while True:
                ok, f = cap.read()
                if not ok: break
                f = cv2.resize(f, (640, 480))       # constant size avoids tracker errors
                with lock:
                    FRAME, FID = f, FID + 1
                state["last"] = time.time()
        except cv2.error as e:
            print("camera error:", e)
        finally:
            cap.release()
        state["status"] = "camera lost, reconnecting"
        time.sleep(2)

def latest():
    with lock:
        return (None, 0) if FRAME is None else (FRAME.copy(), FID)

# ------------------------------- REFLEX LAYER -------------------------------
def init_yolo():
    """Try GPU, then GPU without cuDNN, then CPU. Warm up with track() so trackers register."""
    import torch
    dummy = np.zeros((480, 640, 3), np.uint8)
    attempts = []
    if torch.cuda.is_available():
        attempts += [(0, True), (0, False)]
    attempts += [("cpu", True)]
    for dev, cudnn in attempts:
        torch.backends.cudnn.enabled = cudnn
        try:
            model = YOLO(YOLO_WEIGHTS)
            model.track(dummy, persist=True, verbose=False, imgsz=YOLO_IMGSZ, device=dev,
                        tracker="bytetrack.yaml")
            print(f"YOLO ready on device={dev} (cudnn={cudnn})")
            return model, dev
        except Exception as e:
            print(f"YOLO init failed on device={dev} cudnn={cudnn}: {e}")
            try: torch.cuda.empty_cache()
            except Exception: pass
    raise RuntimeError("YOLO could not start on GPU or CPU")

def reflex():
    state["status"] = "loading Gemma first..."
    vlm_ready.wait(timeout=300)                     # Gemma gets the GPU first
    state["status"] = "loading YOLO..."
    try:
        model, device = init_yolo()
    except Exception:
        traceback.print_exc()
        state["status"] = "YOLO FAILED TO START"
        return
    state["device"] = str(device)
    state["status"] = "running"
    hist = collections.defaultdict(lambda: collections.deque(maxlen=10))
    streak = collections.defaultdict(int)
    last_alert, last_any = {}, 0.0
    seen, n, t0, last_err = 0, 0, time.time(), ""
    while True:
        f, fid = latest()
        if f is None or fid == seen:
            time.sleep(0.005); continue
        seen = fid
        try:
            res = model.track(f, persist=True, verbose=False, imgsz=YOLO_IMGSZ, conf=0.35,
                              iou=0.5, device=device, tracker="bytetrack.yaml")[0]
        except Exception as e:                      # keep the thread alive on odd frames
            if str(e) != last_err:
                traceback.print_exc(); last_err = str(e)
            time.sleep(0.5); continue
        H, W = f.shape[:2]
        now = time.time()
        in_corridor, worst, objs = set(), None, []
        for b in res.boxes:
            name = model.names[int(b.cls[0])]
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            cx, rel = (x1 + x2) / 2, (y2 - y1) / H
            side = "left" if cx < W * 0.4 else "right" if cx > W * 0.6 else "ahead"
            objs.append(f"{name} {side}")           # grounding info for the VLM
            if b.id is None: continue
            tid = int(b.id[0])
            hist[tid].append((now, rel))
            big, small = name in ALERT_CLASSES, name in SMALL_HAZARDS
            if not (big or small) or not (W * 0.28 < cx < W * 0.72):
                continue
            in_corridor.add(tid)
            streak[tid] += 1
            if streak[tid] < 4: continue            # stable for ~4 frames
            if small:                               # floor-level trip hazard close to the feet
                urgent, near = False, (y2 > H * 0.8 and rel > 0.1)
            else:
                ttc = None
                if len(hist[tid]) >= 6:
                    (ta, ha), (tb, hb) = hist[tid][0], hist[tid][-1]
                    grow = (hb - ha) / max(tb - ta, 1e-3)
                    if grow > 0.05: ttc = hb / grow
                urgent = rel > 0.7 or (ttc is not None and ttc < 1.5 and rel > 0.3)
                near = rel > 0.5
            if urgent or near:
                level = 2 if urgent else 1
                if worst is None or level > worst[0]:
                    worst = (level, name, cx)
        for tid in list(streak):
            if tid not in in_corridor: streak[tid] = 0
        state["objects"] = ", ".join(sorted(set(objs))[:8])
        if worst and now - last_any > 1.5:
            level, name, cx = worst
            side = "left" if cx < W * 0.4 else "right" if cx > W * 0.6 else "ahead"
            key = (name, side, level)
            if now - last_alert.get(key, 0) > 4:
                last_alert[key] = last_any = now
                state["last_danger"] = now
                spoken = SPOKEN.get(name, name)
                if level == 2:
                    state["last_urgent"] = now
                    msg = f"Stop! {spoken} {side}"
                    beep()
                else:
                    msg = f"{spoken} {side}, close"
                state["danger"] = msg
                say(msg, 0)
        state["annot"] = res.plot()
        n += 1
        if now - t0 > 1:
            state["fps"], n, t0 = n / (now - t0), 0, now
        if state["vlm_busy"]:                       # leave GPU time for Gemma (~10 FPS cap)
            time.sleep(0.07)

# ------------------------------- VLM LAYER -------------------------------
req_q = queue.Queue()
last_thumb = {"t": None}
GENERATE = f"{OLLAMA_URL}/api/generate"
KEEP_ALIVE = -1                                     # never unload Gemma while the app runs

def vlm_options(num_predict):
    o = {"num_predict": num_predict, "num_ctx": 2048, "temperature": 0.2}
    if OLLAMA_NUM_GPU is not None: o["num_gpu"] = OLLAMA_NUM_GPU
    return o

def check_ollama():
    """Verify Ollama is up and the model exists; auto-correct near matches; load the model."""
    global VLM_MODEL
    try:
        names = [m["name"] for m in requests.get(f"{OLLAMA_URL}/api/tags", timeout=5).json()["models"]]
    except Exception as e:
        state["vlm"] = "Ollama not reachable. Start it with: ollama serve"
        print("Ollama check failed:", e)
        return
    print("Ollama models:", names)
    if VLM_MODEL not in names:
        base = VLM_MODEL.split(":")[0]
        close = [n for n in names if n.startswith(base)]
        if close:
            print(f"Model '{VLM_MODEL}' not found, using '{close[0]}'")
            VLM_MODEL = close[0]
        else:
            state["vlm"] = f"Model '{VLM_MODEL}' not pulled. Run: ollama pull {VLM_MODEL}"
            print(state["vlm"], "| available:", names)
            return
    try:
        state["status"] = "loading Gemma (20-60 s)..."
        r = requests.post(GENERATE, timeout=300, json={
            "model": VLM_MODEL, "prompt": "hi", "stream": False, "keep_alive": KEEP_ALIVE,
            "options": vlm_options(1)}).json()
        if "error" in r:
            state["vlm"] = f"VLM load error: {r['error']}"
            print(state["vlm"])
        else:
            print("VLM warmed up")
    except Exception as e:
        print("Warm-up failed:", e)

def scene_changed(frame):
    t = cv2.cvtColor(cv2.resize(frame, (32, 24)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    if last_thumb["t"] is None or np.abs(t - last_thumb["t"]).mean() > SCENE_THRESH:
        last_thumb["t"] = t
        return True
    return False

RULES = ("Only state what you can clearly see. Never say the path is safe or clear. "
         "Use only 'left', 'right', 'ahead' or clock directions. Never use compass directions "
         "like east or west. ")

def build_prompt(kind, question=None):
    ctx = f"Sensors detected: {state['objects'] or 'nothing'}. "
    if kind == "ask":
        return ("You assist a blind person who is walking. " + ctx + RULES +
                "Answer in 1-2 short sentences. If unsure, say you are not sure. "
                "Question: " + question)
    return ("You describe scenes for a blind pedestrian. " + ctx + RULES +
            "In 2 short sentences describe what is directly ahead and notable objects or "
            "landmarks (crossings, doors, shops, turns). If something is unclear, say so.")

def encode(frame, size=(448, 336)):
    return base64.b64encode(cv2.imencode(".jpg", cv2.resize(frame, size))[1]).decode()

def ask_local_stream(prompt, frame, prio):
    started = time.time()
    buf, full = "", ""
    with requests.post(GENERATE, stream=True, timeout=120, json={
            "model": VLM_MODEL, "prompt": prompt, "images": [encode(frame)], "stream": True,
            "keep_alive": KEEP_ALIVE, "options": vlm_options(50)}) as r:
        if r.status_code != 200:
            raise RuntimeError(f"Ollama HTTP {r.status_code}: {r.text[:200]}")
        for line in r.iter_lines():
            if not line: continue
            data = json.loads(line)
            if "error" in data:
                raise RuntimeError(data["error"])
            tok = data.get("response", "")
            buf += tok; full += tok
            if prio > 0 and state["last_urgent"] > started:   # urgent "Stop!" only: abandon answer
                return full.strip() + " [interrupted]"
            while (m := re.search(r"[.!?]\s", buf)):          # speak each finished sentence
                say(buf[:m.end()].strip(), prio)
                buf = buf[m.end():]
    if buf.strip(): say(buf.strip(), prio)
    return full.strip()

def ask_cloud(prompt, frame, prio):
    key = os.environ.get("GEMINI_API_KEY")
    if not key: raise RuntimeError("GEMINI_API_KEY not set")
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{CLOUD_MODEL}:generateContent?key={key}")
    body = {"contents": [{"parts": [
        {"text": prompt},
        {"inline_data": {"mime_type": "image/jpeg", "data": encode(frame, (640, 480))}}]}],
        "generationConfig": {"maxOutputTokens": 80, "temperature": 0.2}}
    r = requests.post(url, json=body, timeout=30).json()
    if "error" in r:
        raise RuntimeError(r["error"].get("message", str(r["error"])))
    text = r["candidates"][0]["content"]["parts"][0]["text"].strip()
    say(text, prio)
    return text

def vlm_worker():
    global BACKEND
    try:
        check_ollama()                              # Gemma loads FIRST
    except Exception:
        traceback.print_exc()
    finally:
        vlm_ready.set()                             # always let YOLO start, even if Gemma failed
    last_scene = 0.0
    while True:
        try: kind, q = req_q.get(timeout=2)
        except queue.Empty: kind, q = "auto", None
        now = time.time()
        f, _ = latest()
        if f is None: continue
        if kind == "auto":
            if (not state["auto_scene"] or now - last_scene < SCENE_MIN_GAP
                    or now - state["last_danger"] < DANGER_PAUSE or not scene_changed(f)):
                continue
            kind = "scene"
        if kind == "scene": last_scene = now
        prompt = build_prompt(kind, q)
        prio = 1 if kind == "ask" else 2
        print(f"[VLM] {kind} via {BACKEND} | objects: {state['objects']}")
        try:
            state["vlm_busy"] = (BACKEND == "local")
            t = time.time()
            ans = ask_cloud(prompt, f, prio) if BACKEND == "cloud" else ask_local_stream(prompt, f, prio)
            state["vlm_ms"], state["vlm"] = int((time.time() - t) * 1000), ans
            print(f"[VLM] done in {state['vlm_ms']} ms: {ans}")
        except Exception as e:
            traceback.print_exc()
            state["vlm"] = f"VLM error: {e}"
            if BACKEND == "local" and os.environ.get("GEMINI_API_KEY"):
                BACKEND = "cloud"
                state["vlm"] += " -> switched to cloud"
        finally:
            state["vlm_busy"] = False

# ------------------------------- VOICE Q&A -------------------------------
whisper = {"m": None}

def listen_and_ask():
    try:
        import sounddevice as sd
        from faster_whisper import WhisperModel
        if whisper["m"] is None:
            whisper["m"] = WhisperModel("base", device="cpu", compute_type="int8")
        state["status"] = "listening..."
        audio = sd.rec(int(5 * 16000), samplerate=16000, channels=1, dtype="float32")
        sd.wait()
        segs, _ = whisper["m"].transcribe(audio.flatten(), vad_filter=True)
        text = " ".join(s.text for s in segs).strip()
        if text:
            state["vlm"] = f"Q: {text}"
            req_q.put(("ask", text))
        else:
            state["vlm"] = "Did not hear a question"
    except Exception as e:
        traceback.print_exc()
        state["vlm"] = f"Voice error: {e}"
    finally:
        state["status"] = "running"

# ------------------------------- DASHBOARD -------------------------------
def main():
    global BACKEND
    for fn in (speaker, grabber, vlm_worker, reflex):
        threading.Thread(target=fn, daemon=True).start()
    while True:
        img = state["annot"]
        img = np.zeros((480, 640, 3), np.uint8) if img is None else img.copy()
        if state["annot"] is not None:
            lost = time.time() - state["last"] > 2
            if lost and not state["cam_alerted"]:
                state["status"], state["cam_alerted"] = "CAMERA BLOCKED OR LOST", True
                say("Camera lost", 0)
            elif not lost:
                state["cam_alerted"] = False
                if state["status"].startswith("CAMERA"): state["status"] = "running"
        head = (f"FPS {state['fps']:.0f} [{state['device']}] | VLM {state['vlm_ms']}ms [{BACKEND}] | "
                f"auto {'on' if state['auto_scene'] else 'off'} | {state['status']}")
        cv2.putText(img, head, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
        if time.time() - state["last_danger"] < 3 and state["danger"]:
            cv2.putText(img, state["danger"], (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        txt, y = state["vlm"], img.shape[0] - 60
        for i in range(0, len(txt), 75):
            cv2.putText(img, txt[i:i + 75], (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            y += 18
        cv2.imshow("Blind Guide", img)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("q"): break
        if k == ord("d"): req_q.put(("scene", None))
        if k == ord("a"): threading.Thread(target=listen_and_ask, daemon=True).start()
        if k == ord("c"): BACKEND = "cloud" if BACKEND == "local" else "local"
        if k == ord("s"): state["auto_scene"] = not state["auto_scene"]
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
