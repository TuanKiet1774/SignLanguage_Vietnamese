"""
Demo VSL - Nhan dang Bang chu cai Ky hieu Tieng Viet
Phien ban v3 - Fix distribution shift (hand cropping)
Fix font tieng Viet bang Pillow (cv2.putText khong ho tro Unicode)
========================================================
Yeu cau:
    pip install opencv-python mediapipe torch torchvision pillow

Chay:
    python demo_vsl.py
    python demo_vsl.py --checkpoint customcnn_checkpoint.pth
    python demo_vsl.py --checkpoint mobilenetv2_checkpoint.pth

Phim tat:
    Q / ESC  : Thoat
    S        : Chup man hinh -> saves/
    SPACE    : Pause / Resume
    C        : Xoa lich su du doan
    D        : Bat/tat debug (hien thi skeleton dang duoc dua vao model)
"""

import sys, time, argparse, urllib.request
from pathlib import Path
from collections import deque
from datetime import datetime

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
from torchvision import models, transforms

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:
    print("Thieu mediapipe. Chay: pip install mediapipe")
    sys.exit(1)

# ── Cau hinh ──────────────────────────────────────────────────────────────
DEFAULT_CHECKPOINT = "mobilenetv2_checkpoint.pth"
IMG_SIZE           = 224
CONF_THRESHOLD     = 0.60
SMOOTH_WINDOW      = 8
TOP_K              = 3
HAND_PAD           = 0.25    # Padding quanh vung ban tay (% of bbox size)

MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = "hand_landmarker.task"

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),(0,5),(0,17),
]

# ── Font tieng Viet ────────────────────────────────────────────────────────
# Danh sach font uu tien, ho tro Unicode/tieng Viet
_FONT_CANDIDATES = [
    # Font co san tren Linux
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    # Font co san tren Windows
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/tahoma.ttf",
    # Font co san tren macOS
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
]

def _load_font(size: int) -> ImageFont.FreeTypeFont:
    """Tim va load font ho tro Unicode voi kich co chi dinh."""
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    # Fallback: dung font mac dinh cua Pillow (khong co dau nhung van chay duoc)
    return ImageFont.load_default()

# Cache font de tranh load lai moi frame
_font_cache: dict[int, ImageFont.FreeTypeFont] = {}

def get_font(size: int) -> ImageFont.FreeTypeFont:
    if size not in _font_cache:
        _font_cache[size] = _load_font(size)
    return _font_cache[size]


def put_text_vn(img_bgr: np.ndarray, text: str, org: tuple,
                font_size: int = 16,
                color: tuple = (255, 255, 255),
                thickness: int = 1) -> np.ndarray:
    """
    Ve chu Unicode/tieng Viet len anh OpenCV (BGR numpy array).
    Su dung Pillow thay cho cv2.putText de ho tro day du ky tu co dau.

    Args:
        img_bgr   : Anh BGR (numpy array), se bi chinh sua truc tiep.
        text      : Chuoi Unicode can hien thi.
        org       : (x, y) vi tri goc trai-duoi cua dong chu (giong cv2).
        font_size : Kich co chu (pixel).
        color     : Mau BGR (giong cv2), vi du (255,255,255).
        thickness : Khong dung cho Pillow nhung giu lai de tuong thich API.
    Returns:
        Anh BGR da duoc ve chu len.
    """
    # Chuyen BGR -> RGB cho Pillow
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw    = ImageDraw.Draw(pil_img)
    font    = get_font(font_size)

    # Tinh chieu cao chu de can chinh org giong cv2 (goc trai-duoi)
    bbox = font.getbbox(text)          # (left, top, right, bottom)
    text_h = bbox[3] - bbox[1]
    x = org[0]
    y = org[1] - text_h - bbox[1]     # dich len tren de "org" la goc trai-duoi

    # Chuyen mau BGR -> RGB
    r, g, b = color[2], color[1], color[0]
    draw.text((x, y), text, font=font, fill=(r, g, b))

    # Chuyen lai RGB -> BGR
    result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    img_bgr[:] = result
    return img_bgr


def put_text_vn_centered(img_bgr: np.ndarray, text: str,
                          cx: int, cy: int,
                          font_size: int = 16,
                          color: tuple = (255, 255, 255)) -> np.ndarray:
    """Ve chu can giua theo toa do (cx, cy)."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw    = ImageDraw.Draw(pil_img)
    font    = get_font(font_size)
    bbox    = font.getbbox(text)
    text_w  = bbox[2] - bbox[0]
    text_h  = bbox[3] - bbox[1]
    x = cx - text_w // 2
    y = cy - text_h // 2
    r, g, b = color[2], color[1], color[0]
    draw.text((x, y), text, font=font, fill=(r, g, b))
    img_bgr[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img_bgr


def get_text_size_vn(text: str, font_size: int) -> tuple[int, int]:
    """Tra ve (width, height) cua chuoi text voi font_size cho truoc."""
    font = get_font(font_size)
    bbox = font.getbbox(text)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


# ── Download model ────────────────────────────────────────────────────────
def ensure_model(path=MODEL_PATH):
    if Path(path).exists():
        return path
    print(f"Downloading MediaPipe hand model (~30 MB)...")
    urllib.request.urlretrieve(MODEL_URL, path,
        reporthook=lambda b, bs, t: print(
            f"\r  {b*bs/1e6:.1f}/{t/1e6:.1f} MB", end="", flush=True))
    print(f"\nDone: {path}")
    return path


# ── Model definitions ─────────────────────────────────────────────────────
class CustomCNN(nn.Module):
    def __init__(self, num_classes, dropout=0.4):
        super().__init__()
        def conv_block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
                nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
                nn.MaxPool2d(2),
            )
        self.features   = nn.Sequential(conv_block(3,64), conv_block(64,128),
                                         conv_block(128,256), conv_block(256,512))
        self.pool       = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(512,256), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(256, num_classes))
    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))

def build_mobilenetv2(n, d=0.3):
    m = models.mobilenet_v2(weights=None)
    m.classifier = nn.Sequential(
        nn.Dropout(d), nn.Linear(m.classifier[1].in_features, 512), nn.ReLU(),
        nn.Dropout(d/2), nn.Linear(512, n))
    return m

def load_model(path, device):
    ckpt    = torch.load(path, map_location=device)
    n       = ckpt['num_classes']
    idx2cls = {i: c for c, i in ckpt['cls2idx'].items()}
    mean    = ckpt.get('mean', [0.055]*3)
    std     = ckpt.get('std',  [0.210]*3)
    name    = Path(path).stem.lower()
    if 'mobilenet' in name:
        model, mtype = build_mobilenetv2(n), 'MobileNetV2'
    else:
        model, mtype = CustomCNN(n), 'CustomCNN'
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    print(f"Loaded {mtype} | {n} classes | mean={mean[0]:.3f} std={std[0]:.3f}")
    return model, idx2cls, tfm, mtype


# ── MediaPipe ─────────────────────────────────────────────────────────────
def init_landmarker(model_path):
    opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    return mp_vision.HandLandmarker.create_from_options(opts)

def detect_hand(landmarker, bgr):
    rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_img)
    return result.hand_landmarks[0] if result.hand_landmarks else None


# ── KEY FIX: Crop ban tay truoc khi ve skeleton ───────────────────────────
def landmarks_to_skeleton(landmarks, frame_h, frame_w, pad=HAND_PAD, size=224):
    """
    Buoc 1: Tinh bounding box cua ban tay trong frame goc
    Buoc 2: Crop + padding
    Buoc 3: Ve skeleton trong vung da crop, scale len 224x224
    -> Ket qua giong voi cach dataset VSL v2 duoc tao ra
    """
    xs = [lm.x * frame_w for lm in landmarks]
    ys = [lm.y * frame_h for lm in landmarks]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    bw = x_max - x_min
    bh = y_max - y_min
    margin = max(bw, bh) * pad

    x0 = max(0, x_min - margin)
    y0 = max(0, y_min - margin)
    x1 = min(frame_w, x_max + margin)
    y1 = min(frame_h, y_max + margin)

    crop_w = x1 - x0
    crop_h = y1 - y0

    canvas = np.zeros((size, size, 3), dtype=np.uint8)

    def to_canvas(lm_x, lm_y):
        cx = int((lm_x * frame_w - x0) / crop_w * size)
        cy = int((lm_y * frame_h - y0) / crop_h * size)
        cx = max(0, min(size-1, cx))
        cy = max(0, min(size-1, cy))
        return cx, cy

    pts = [to_canvas(lm.x, lm.y) for lm in landmarks]

    for a, b in HAND_CONNECTIONS:
        cv2.line(canvas, pts[a], pts[b], (255,255,255), 2)
    for pt in pts:
        cv2.circle(canvas, pt, 4, (200,200,200), -1)

    crop_rect = (int(x0), int(y0), int(x1), int(y1))
    return canvas, crop_rect


def draw_landmarks_on_frame(frame, landmarks):
    h, w = frame.shape[:2]
    pts  = [(int(lm.x*w), int(lm.y*h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0,200,100), 2)
    for pt in pts:
        cv2.circle(frame, pt, 5, (0,255,180), -1)
        cv2.circle(frame, pt, 5, (0,130,70), 1)


# ── Inference ─────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(model, skeleton, tfm, device, topk=3):
    t = tfm(Image.fromarray(skeleton)).unsqueeze(0).to(device)
    p = torch.softmax(model(t), dim=1)[0]
    top_p, top_i = p.topk(topk)
    return [(i.item(), c.item()) for i, c in zip(top_i, top_p)]

class Smoother:
    def __init__(self, n=8):
        self.buf = deque(maxlen=n)
    def push(self, idx, c): self.buf.append((idx, c))
    def get(self):
        if not self.buf: return None, 0.0
        v = {}
        for i, c in self.buf: v[i] = v.get(i,0) + c
        b = max(v, key=v.get)
        return b, v[b]/len(self.buf)
    def clear(self): self.buf.clear()


# ── Giao dien ─────────────────────────────────────────────────────────────
def render(frame, skeleton, crop_rect, preds, idx2cls,
           smoother, fps, mtype, has_hand, paused, debug):
    h, w  = frame.shape[:2]
    PW    = 300
    panel = np.full((h, PW, 3), (22,22,32), dtype=np.uint8)

    # Ve hop crop len frame
    if has_hand and crop_rect:
        x0, y0, x1, y1 = crop_rect
        cv2.rectangle(frame, (x0,y0), (x1,y1), (0,200,255), 2)
        put_text_vn(frame, "Vùng bàn tay", (x0, y0-6),
                    font_size=14, color=(0,200,255))

    # Skeleton preview trong panel
    sk_sz = 224
    if skeleton is not None:
        xo = (PW - sk_sz) // 2
        panel[8:8+sk_sz, xo:xo+sk_sz] = skeleton
    put_text_vn(panel, "ĐẦU VÀO MÔ HÌNH", (10, sk_sz+18),
                font_size=12, color=(80,80,100))
    cv2.line(panel, (10, sk_sz+26), (PW-10, sk_sz+26), (50,50,70), 1)

    y = sk_sz + 42
    si, sc = smoother.get()

    if has_hand and si is not None and sc >= CONF_THRESHOLD:
        lbl = idx2cls[si]
        # Chu ky hieu (chu cai) -> font lon
        fs  = 52 if len(lbl) == 1 else 28
        tw, th = get_text_size_vn(lbl, fs)
        put_text_vn_centered(panel, lbl, PW//2, y + 40, font_size=fs,
                              color=(0,255,150))
        # Thanh confidence
        bw = int((PW-20)*sc)
        cv2.rectangle(panel, (10,y+72), (PW-10,y+86), (40,40,60), -1)
        cl = (0,220,100) if sc > 0.9 else (0,160,230)
        cv2.rectangle(panel, (10,y+72), (10+bw,y+86), cl, -1)
        put_text_vn_centered(panel, f"{sc*100:.1f}%", PW//2, y+104,
                              font_size=14, color=(190,190,190))
        y += 118
    elif not has_hand:
        put_text_vn(panel, "Không thấy tay", (10, y+38),
                    font_size=14, color=(80,80,110))
        y += 58
    else:
        put_text_vn(panel, "Độ tin cậy thấp...", (10, y+38),
                    font_size=13, color=(80,80,110))
        y += 58

    if preds and has_hand:
        cv2.line(panel, (10,y), (PW-10,y), (50,50,70), 1)
        y += 14
        for k, (idx, conf) in enumerate(preds[:TOP_K]):
            bl = int((PW-82)*conf)
            bc = (0,190,90) if k == 0 else (50,110,180)
            cv2.rectangle(panel, (58,y-10), (58+bl,y+2), bc, -1)
            tc = (255,255,255) if k == 0 else (140,140,140)
            put_text_vn(panel, f"{k+1}. {idx2cls[idx]}", (8, y),
                        font_size=14, color=tc)
            put_text_vn(panel, f"{conf*100:.0f}%", (PW-44, y),
                        font_size=12, color=(130,130,130))
            y += 27

    # Footer
    put_text_vn(panel, mtype,                (10, h-60), font_size=11, color=(60,100,180))
    put_text_vn(panel, f"FPS:{fps:.1f}",     (10, h-44), font_size=11, color=(80,80,110))
    dbg_col = (0,200,100) if debug else (80,80,110)
    put_text_vn(panel, f"Debug:{'BẬT' if debug else 'TẮT'}",
                (10, h-28), font_size=11, color=dbg_col)
    put_text_vn(panel, "Q:Thoát  S:Lưu  C:Xóa  SPC:Dừng  D:Debug",
                (10, h-10), font_size=10, color=(60,60,80))

    out = np.hstack([frame, panel])

    if paused:
        ov = out.copy()
        cv2.rectangle(ov, (0,0), (out.shape[1],out.shape[0]), (0,0,0), -1)
        cv2.addWeighted(ov, 0.5, out, 0.5, 0, out)
        cw, ch = out.shape[1], out.shape[0]
        put_text_vn_centered(out, "TẠM DỪNG", cw//2, ch//2,
                              font_size=72, color=(255,255,255))

    # Debug window
    if debug and skeleton is not None:
        big = cv2.resize(skeleton, (400,400), interpolation=cv2.INTER_NEAREST)
        put_text_vn(big, "Đầu vào mô hình (xem trước 400x400)",
                    (6, 20), font_size=13, color=(100,255,100))
        cv2.imshow("DEBUG - Skeleton", big)
    elif not debug:
        try: cv2.destroyWindow("DEBUG - Skeleton")
        except: pass

    return out


# ── Main ──────────────────────────────────────────────────────────────────
def main(args):
    global CONF_THRESHOLD, HAND_PAD
    CONF_THRESHOLD = args.threshold
    HAND_PAD       = args.pad

    if not Path(args.checkpoint).exists():
        print(f"Không tìm thấy: {args.checkpoint}")
        sys.exit(1)

    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = ensure_model(MODEL_PATH)
    model, idx2cls, tfm, mtype = load_model(args.checkpoint, device)
    landmarker = init_landmarker(model_path)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Không mở được camera {args.camera}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(f"Camera: {fw}x{fh} | Pad={HAND_PAD} | Threshold={CONF_THRESHOLD}")

    Path("saves").mkdir(exist_ok=True)
    smoother  = Smoother(SMOOTH_WINDOW)
    paused    = False
    debug     = False
    preds     = []
    skeleton  = None
    crop_rect = None
    has_hand  = False
    fps       = 0.0
    t0, fc    = time.time(), 0

    print("\nĐang chạy... Q:thoát SPACE:dừng S:chụp C:xóa D:debug\n")

    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.flip(frame, 1)

        if not paused:
            lms      = detect_hand(landmarker, frame)
            has_hand = lms is not None

            if has_hand:
                draw_landmarks_on_frame(frame, lms)
                skeleton, crop_rect = landmarks_to_skeleton(
                    lms, fh, fw, pad=HAND_PAD, size=IMG_SIZE)
                preds = predict(model, skeleton, tfm, device, TOP_K)
                if preds:
                    smoother.push(*preds[0])
            else:
                skeleton = None; crop_rect = None; preds = []

            fc += 1
            tn = time.time()
            if tn - t0 >= 0.5:
                fps = fc/(tn-t0); t0, fc = tn, 0

        display = render(frame, skeleton, crop_rect, preds, idx2cls,
                         smoother, fps, mtype, has_hand, paused, debug)
        cv2.imshow("VSL v2 - Nhận dạng Ký hiệu Tiếng Việt", display)
        k = cv2.waitKey(1) & 0xFF

        if k in (ord('q'), ord('Q'), 27): break
        elif k == 32:
            paused = not paused
            print("Tạm dừng" if paused else "Tiếp tục")
        elif k in (ord('s'), ord('S')):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"saves/cap_{ts}.png", display)
            if skeleton is not None:
                cv2.imwrite(f"saves/sk_{ts}.png", skeleton)
            print(f"Đã lưu: saves/cap_{ts}.png")
        elif k in (ord('c'), ord('C')):
            smoother.clear(); preds = []
            print("Đã xóa")
        elif k in (ord('d'), ord('D')):
            debug = not debug
            print(f"Debug {'BẬT' if debug else 'TẮT'}")

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print("Kết thúc.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--camera",     type=int,   default=0)
    ap.add_argument("--threshold",  type=float, default=CONF_THRESHOLD)
    ap.add_argument("--pad",        type=float, default=HAND_PAD,
                    help="Padding quanh bàn tay 0.0–0.5 (mặc định 0.25)")
    main(ap.parse_args())