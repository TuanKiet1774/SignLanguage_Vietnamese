"""
Danh gia & so sanh 2 model VSL (CustomCNN vs MobileNetV2) tren du lieu thuc
============================================================================
Doc anh thuc nghiem trong data/<NHAN>/*.jpg (frame DAY DU, chua crop),
chay LAI dung pipeline that cua demo_vsl.py cho ca 2 checkpoint:
    MediaPipe detect tay -> crop+padding -> ve skeleton 224x224 -> model
roi tinh accuracy, top-3 accuracy, confusion matrix, thoi gian inference,
va xuat bao cao so sanh (anh PNG + CSV).

Yeu cau:
    pip install opencv-python mediapipe torch torchvision pillow numpy matplotlib

Chay:
    python evaluate_models.py
    python evaluate_models.py --data data --out report
    python evaluate_models.py --customcnn customcnn_checkpoint.pth \
                               --mobilenet mobilenetv2_checkpoint.pth

Luu y quan trong:
    - Anh KHONG co tay phat hien duoc (MediaPipe khong tim thay landmark)
      se duoc tinh la "MISS" rieng, KHONG tinh la du doan sai vao mot
      nhan cu the trong confusion matrix, nhung CO tinh vao accuracy
      tong (vi day la 1 dang loi that cua he thong end-to-end).
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:
    print("Thieu mediapipe. Chay: pip install mediapipe")
    sys.exit(1)

import urllib.request

# ── Cau hinh (dong bo voi demo_vsl.py) ─────────────────────────────────────
IMG_SIZE   = 224
HAND_PAD   = 0.25
TOP_K      = 3
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = "hand_landmarker.task"

# Mapping nguoc: ten thu muc -> nhan hien thi (phai khop voi collect_data.py)
DIRNAME_TO_LABEL = {"Dd": "Đ", "Rau": "Râu", "mu_": "mũ"}

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),(0,5),(0,17),
]


# ── Model definitions (copy chinh xac tu demo_vsl.py de dam bao tuong thich) ─
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
    ckpt    = torch.load(path, map_location=device, weights_only=False)
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
    return model, idx2cls, tfm, mtype


# ── MediaPipe (copy tu demo_vsl.py) ────────────────────────────────────────
def ensure_model(path=MODEL_PATH):
    if Path(path).exists():
        return path
    print("Downloading MediaPipe hand model (~30 MB)...")
    urllib.request.urlretrieve(MODEL_URL, path)
    print(f"Done: {path}")
    return path


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


def landmarks_to_skeleton(landmarks, frame_h, frame_w, pad=HAND_PAD, size=224):
    xs = [lm.x * frame_w for lm in landmarks]
    ys = [lm.y * frame_h for lm in landmarks]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    bw, bh = x_max - x_min, y_max - y_min
    margin = max(bw, bh) * pad
    x0 = max(0, x_min - margin); y0 = max(0, y_min - margin)
    x1 = min(frame_w, x_max + margin); y1 = min(frame_h, y_max + margin)
    crop_w, crop_h = x1 - x0, y1 - y0
    canvas = np.zeros((size, size, 3), dtype=np.uint8)

    def to_canvas(lm_x, lm_y):
        cx = int((lm_x * frame_w - x0) / crop_w * size)
        cy = int((lm_y * frame_h - y0) / crop_h * size)
        return max(0, min(size-1, cx)), max(0, min(size-1, cy))

    pts = [to_canvas(lm.x, lm.y) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(canvas, pts[a], pts[b], (255,255,255), 2)
    for pt in pts:
        cv2.circle(canvas, pt, 4, (200,200,200), -1)
    return canvas


@torch.no_grad()
def predict(model, skeleton, tfm, device, topk=3):
    t = tfm(Image.fromarray(skeleton)).unsqueeze(0).to(device)
    p = torch.softmax(model(t), dim=1)[0]
    top_p, top_i = p.topk(min(topk, p.shape[0]))
    return [(i.item(), c.item()) for i, c in zip(top_i, top_p)]


# ── Progress bar don gian (khong can thu vien ngoai) ──────────────────────
def _progress(current, total, mtype, prefix_len=14, bar_len=30):
    pct  = current / total
    done = int(bar_len * pct)
    bar  = "█" * done + "░" * (bar_len - done)
    print(f"\r  {mtype:{prefix_len}s} [{bar}] {current:>5d}/{total} ({pct*100:5.1f}%)",
          end="", flush=True)


# ── Danh gia mot model tren toan bo dataset ────────────────────────────────
def evaluate_model(model, idx2cls, tfm, mtype, device, landmarker,
                    samples, cls2idx_truth):
    """samples: list of (true_label_str, image_path)
    Tra ve dict ket qua chi tiet."""
    records = []
    n_miss_hand = 0
    infer_times = []
    total = len(samples)

    for idx_s, (true_label, img_path) in enumerate(samples):
        _progress(idx_s + 1, total, mtype)

        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        h, w = frame.shape[:2]

        t_start = time.time()
        lms = detect_hand(landmarker, frame)
        if lms is None:
            n_miss_hand += 1
            records.append({
                "true_label": true_label, "pred_label": None,
                "correct": False, "top2_correct": False, "top3_correct": False,
                "confidence": 0.0, "hand_detected": False,
                "path": str(img_path),
            })
            continue

        skeleton = landmarks_to_skeleton(lms, h, w, pad=HAND_PAD, size=IMG_SIZE)
        preds = predict(model, skeleton, tfm, device, TOP_K)
        t_elapsed = time.time() - t_start
        infer_times.append(t_elapsed)

        top1_idx, top1_conf = preds[0]
        pred_label   = idx2cls[top1_idx]
        top2_labels  = [idx2cls[i] for i, _ in preds[:2]]
        top3_labels  = [idx2cls[i] for i, _ in preds]

        records.append({
            "true_label":   true_label,
            "pred_label":   pred_label,
            "correct":      pred_label == true_label,
            "top2_correct": true_label in top2_labels,
            "top3_correct": true_label in top3_labels,
            "confidence":   top1_conf,
            "hand_detected": True,
            "path": str(img_path),
        })

    print()  # xuong dong sau progress bar

    n_total   = len(records)
    n_correct = sum(r["correct"]      for r in records)
    n_top2    = sum(r["top2_correct"] for r in records)
    n_top3    = sum(r["top3_correct"] for r in records)

    return {
        "mtype":            mtype,
        "records":          records,
        "n_total":          n_total,
        "n_correct":        n_correct,
        "n_top2_correct":   n_top2,
        "n_top3_correct":   n_top3,
        "n_miss_hand":      n_miss_hand,
        "accuracy":         n_correct / n_total if n_total else 0.0,
        "top2_accuracy":    n_top2    / n_total if n_total else 0.0,
        "top3_accuracy":    n_top3    / n_total if n_total else 0.0,
        "hand_detect_rate": (n_total - n_miss_hand) / n_total if n_total else 0.0,
        "avg_infer_ms":     (sum(infer_times) / len(infer_times) * 1000) if infer_times else 0.0,
    }


def per_class_report(result, all_labels):
    """Tra ve dict: label -> {n, correct, accuracy, miss_hand}"""
    stats = {l: {"n": 0, "correct": 0, "miss_hand": 0} for l in all_labels}
    for r in result["records"]:
        tl = r["true_label"]
        if tl not in stats:
            stats[tl] = {"n": 0, "correct": 0, "miss_hand": 0}
        stats[tl]["n"] += 1
        if r["correct"]:
            stats[tl]["correct"] += 1
        if not r["hand_detected"]:
            stats[tl]["miss_hand"] += 1
    for l, s in stats.items():
        s["accuracy"] = s["correct"] / s["n"] if s["n"] else None
    return stats


def confusion_matrix(result, all_labels):
    """Ma tran nham lan: chi tinh tren cac mau co phat hien tay.
    Anh khong phat hien tay duoc xu ly rieng (xem hand_detect_rate)."""
    idx = {l: i for i, l in enumerate(all_labels)}
    n = len(all_labels)
    cm = np.zeros((n, n), dtype=int)
    for r in result["records"]:
        if not r["hand_detected"]:
            continue
        ti = idx.get(r["true_label"])
        pi = idx.get(r["pred_label"])
        if ti is not None and pi is not None:
            cm[ti, pi] += 1
    return cm


# ── Bao cao truc quan ───────────────────────────────────────────────────────
def save_reports(results, all_labels, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1) Bang tom tat tong quan (CSV + in console)
    summary_rows = []
    for res in results:
        summary_rows.append({
            "model":            res["mtype"],
            "n_samples":        res["n_total"],
            "accuracy":         round(res["accuracy"]      * 100, 2),
            "top2_accuracy":    round(res["top2_accuracy"] * 100, 2),
            "top3_accuracy":    round(res["top3_accuracy"] * 100, 2),
            "hand_detect_rate": round(res["hand_detect_rate"] * 100, 2),
            "avg_infer_ms":     round(res["avg_infer_ms"], 2),
        })
    print("\n=== TOM TAT TONG QUAN ===")
    header = (f"{'Model':14s}{'N':>6s}{'Accuracy':>11s}{'Top-2 Acc':>11s}"
              f"{'Top-3 Acc':>11s}{'Hand-detect':>13s}{'Infer(ms)':>11s}")
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        print(f"{row['model']:14s}{row['n_samples']:>6d}"
              f"{row['accuracy']:>10.2f}%{row['top2_accuracy']:>10.2f}%"
              f"{row['top3_accuracy']:>10.2f}%{row['hand_detect_rate']:>12.2f}%"
              f"{row['avg_infer_ms']:>10.2f}")

    import csv
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    # 2) Per-class accuracy comparison (bar chart)
    fig, ax = plt.subplots(figsize=(14, 6))
    width = 0.8 / len(results)
    x = np.arange(len(all_labels))
    for mi, res in enumerate(results):
        stats = per_class_report(res, all_labels)
        accs = [(stats[l]["accuracy"] or 0) * 100 for l in all_labels]
        ax.bar(x + mi*width, accs, width, label=res["mtype"])
    ax.set_xticks(x + width*(len(results)-1)/2)
    ax.set_xticklabels(all_labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Do chinh xac theo tung nhan")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "per_class_accuracy.png", dpi=150)
    plt.close(fig)

    # 3) Confusion matrices (1 per model)
    for res in results:
        cm = confusion_matrix(res, all_labels)
        fig, ax = plt.subplots(figsize=(10, 9))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(all_labels))); ax.set_xticklabels(all_labels, rotation=90)
        ax.set_yticks(range(len(all_labels))); ax.set_yticklabels(all_labels)
        ax.set_xlabel("Du doan"); ax.set_ylabel("Thuc te")
        ax.set_title(f"Confusion Matrix - {res['mtype']}")
        for i in range(len(all_labels)):
            for j in range(len(all_labels)):
                if cm[i, j] > 0:
                    color = "white" if cm[i, j] > cm.max()/2 else "black"
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                            fontsize=7, color=color)
        fig.colorbar(im)
        fig.tight_layout()
        fname = f"confusion_{res['mtype'].lower()}.png"
        fig.savefig(out_dir / fname, dpi=150)
        plt.close(fig)

    # 4) Per-class CSV chi tiet
    with open(out_dir / "per_class_detail.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "model", "n", "correct", "accuracy_%", "miss_hand"])
        for res in results:
            stats = per_class_report(res, all_labels)
            for l in all_labels:
                s = stats[l]
                acc = round(s["accuracy"]*100, 2) if s["accuracy"] is not None else ""
                writer.writerow([l, res["mtype"], s["n"], s["correct"], acc, s["miss_hand"]])

    # 5) Raw predictions (them cot top2_correct de debug)
    with open(out_dir / "raw_predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "true_label", "pred_label", "correct",
                          "top2_correct", "top3_correct",
                          "confidence", "hand_detected", "path"])
        for res in results:
            for r in res["records"]:
                writer.writerow([res["mtype"], r["true_label"], r["pred_label"],
                                  r["correct"], r["top2_correct"], r["top3_correct"],
                                  round(r["confidence"], 4), r["hand_detected"], r["path"]])

    # 6) Confidence distribution: correct vs incorrect (bo qua miss_hand)
    fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 5), sharey=False)
    if len(results) == 1:
        axes = [axes]
    bins = np.linspace(0, 1, 26)
    for ax, res in zip(axes, results):
        conf_correct = [r["confidence"] for r in res["records"]
                        if r["hand_detected"] and r["correct"]]
        conf_wrong   = [r["confidence"] for r in res["records"]
                        if r["hand_detected"] and not r["correct"]]
        ax.hist(conf_correct, bins=bins, alpha=0.7, color="steelblue",  label=f"Dung ({len(conf_correct)})")
        ax.hist(conf_wrong,   bins=bins, alpha=0.7, color="tomato",     label=f"Sai  ({len(conf_wrong)})")
        avg_c = sum(conf_correct) / len(conf_correct) if conf_correct else 0
        avg_w = sum(conf_wrong)   / len(conf_wrong)   if conf_wrong   else 0
        ax.axvline(avg_c, color="steelblue", linestyle="--", linewidth=1.5,
                   label=f"TB dung={avg_c:.2f}")
        ax.axvline(avg_w, color="tomato",    linestyle="--", linewidth=1.5,
                   label=f"TB sai={avg_w:.2f}")
        ax.set_title(f"Confidence Distribution - {res['mtype']}")
        ax.set_xlabel("Confidence (softmax top-1)")
        ax.set_ylabel("So luong anh")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "confidence_distribution.png", dpi=150)
    plt.close(fig)

    print(f"\nBao cao da luu tai: {out_dir.resolve()}")
    print("  - summary.csv                  : tong quan accuracy / top-2 / top-3 / toc do")
    print("  - per_class_accuracy.png        : bieu do so sanh accuracy theo nhan")
    print("  - confusion_<model>.png         : ma tran nham lan moi model")
    print("  - confidence_distribution.png   : phan phoi confidence (dung vs sai)")
    print("  - per_class_detail.csv          : chi tiet accuracy tung nhan")
    print("  - raw_predictions.csv           : du doan tung anh, co top2 (de debug)")


def gather_samples(data_dir: Path):
    """Doc thu muc data/, tra ve list (true_label, path) va danh sach nhan."""
    samples = []
    labels_found = []
    for sub in sorted(data_dir.iterdir()):
        if not sub.is_dir():
            continue
        label = DIRNAME_TO_LABEL.get(sub.name, sub.name)
        imgs = sorted(list(sub.glob("*.jpg")) + list(sub.glob("*.png")))
        if not imgs:
            continue
        labels_found.append(label)
        for p in imgs:
            samples.append((label, p))
    return samples, labels_found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data")
    ap.add_argument("--out", type=str, default="report")
    ap.add_argument("--customcnn", type=str, default="customcnn_checkpoint.pth")
    ap.add_argument("--mobilenet", type=str, default="mobilenetv2_checkpoint.pth")
    args = ap.parse_args()

    data_dir = Path(args.data)
    if not data_dir.exists():
        print(f"Khong tim thay thu muc du lieu: {data_dir}")
        print("Hay chay collect_data.py truoc.")
        sys.exit(1)

    samples, labels_found = gather_samples(data_dir)
    if not samples:
        print(f"Khong tim thay anh nao trong {data_dir}/")
        sys.exit(1)
    print(f"Tim thay {len(samples)} anh tren {len(labels_found)} nhan: {labels_found}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model_path = ensure_model(MODEL_PATH)
    landmarker = init_landmarker(model_path)

    checkpoints = []
    if Path(args.customcnn).exists():
        checkpoints.append(args.customcnn)
    else:
        print(f"[!] Khong tim thay {args.customcnn}, bo qua model nay.")
    if Path(args.mobilenet).exists():
        checkpoints.append(args.mobilenet)
    else:
        print(f"[!] Khong tim thay {args.mobilenet}, bo qua model nay.")

    if not checkpoints:
        print("Khong co checkpoint nao de danh gia.")
        sys.exit(1)

    results = []
    for ckpt_path in checkpoints:
        print(f"\nDang danh gia: {ckpt_path} ...")
        model, idx2cls, tfm, mtype = load_model(ckpt_path, device)
        all_labels_model = list(idx2cls.values())
        res = evaluate_model(model, idx2cls, tfm, mtype, device, landmarker,
                              samples, idx2cls)
        results.append(res)
        print(f"  {mtype}: accuracy={res['accuracy']*100:.2f}%  "
              f"top2={res['top2_accuracy']*100:.2f}%  "
              f"top3={res['top3_accuracy']*100:.2f}%  "
              f"hand_detect={res['hand_detect_rate']*100:.2f}%  "
              f"infer={res['avg_infer_ms']:.1f}ms")

    # Nhan dung cho bieu do/confusion matrix: uu tien danh sach trong checkpoint
    # (de hien thi du ca nhan khong xuat hien trong data thu thap)
    _, idx2cls0, _, _ = load_model(checkpoints[0], device)
    all_labels = list(idx2cls0.values())
    # Bo sung nhan co trong data nhung khong co trong model (vd go nham ten)
    for l in labels_found:
        if l not in all_labels:
            all_labels.append(l)

    save_reports(results, all_labels, Path(args.out))

    landmarker.close()


if __name__ == "__main__":
    main()