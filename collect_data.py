"""
Thu thap du lieu thuc nghiem cho VSL (Bang chu cai Ky hieu Tieng Viet)
========================================================================
Quay video ngan (mac dinh 5s) cho moi nhan, tu dong trich N frame trai deu
ra thu muc data/<NHAN>/frame_XXXX.jpg.

Frame duoc luu DAY DU (toan bo khung hinh webcam, KHONG crop san) de khi
danh gia (evaluate_models.py) chay lai dung pipeline that: MediaPipe
detect tay tren frame goc -> crop -> ve skeleton -> dua vao model.
Lam vay moi danh gia dung ca buoc phat hien tay, khong chi rieng classifier.

Yeu cau:
    pip install opencv-python

Chay:
    python collect_data.py
    python collect_data.py --duration 5.0 --frames 150
    python collect_data.py --labels A,B,C          # chi thu mot vai nhan
    python collect_data.py --resume                # tiep tuc, bo qua nhan da du frame

Phim tat trong luc chay:
    SPACE       : Bat dau quay video cho nhan hien tai (dem nguoc 3s roi quay)
    N           : Bo qua nhan hien tai, chuyen sang nhan tiep theo
    P           : Quay lai nhan truoc do
    R           : Quay lai tu dau cho nhan hien tai (xoa frame da chup)
    Q / ESC     : Thoat
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

# ── Danh sach nhan (KHOP CHINH XAC voi cls2idx trong checkpoint) ──────────
# A B C D E G H I K L M N O P Q R S T U V X Y Du Rau mu
LABELS_DEFAULT = [
    "A", "B", "C", "D", "E", "G", "H", "I", "K", "L", "M", "N", "O",
    "P", "Q", "R", "S", "T", "U", "V", "X", "Y", "Đ", "Râu", "mũ",
]

DATA_DIR = Path("data")
COUNTDOWN_SEC = 3
DEFAULT_DURATION = 5.0
DEFAULT_FRAMES = 150


def safe_label_dirname(label: str) -> str:
    """Mot vai he thong file nhay cam voi ky tu co dau / hoa-thuong (vd macOS).
    Dung mapping co dinh de ten thu muc on dinh, khong phu thuoc OS."""
    mapping = {"Đ": "Dd", "Râu": "Rau", "mũ": "mu_"}
    return mapping.get(label, label)


def put_overlay_text(frame, lines, org=(20, 40), color=(255, 255, 255),
                      scale=1.0, thickness=2, line_gap=40):
    """Chu khong dau (ASCII) dung cv2.putText binh thuong cho UI thu thap,
    de don gian (khong can phu thuoc Pillow font o day)."""
    x, y = org
    for line in lines:
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thickness, cv2.LINE_AA)
        y += line_gap


def count_existing_frames(label_dir: Path) -> int:
    if not label_dir.exists():
        return 0
    return len(list(label_dir.glob("frame_*.jpg")))


def record_clip(cap, label_dir: Path, duration: float, n_frames: int):
    """Dem nguoc, quay video trong `duration` giay, trich n_frames frame
    trai deu theo thoi gian, luu vao label_dir. Hien thi preview lien tuc."""
    label_dir.mkdir(parents=True, exist_ok=True)

    # Dem nguoc truoc khi quay, de nguoi dung kip dat tay vao tu the
    countdown_start = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            return False
        frame = cv2.flip(frame, 1)
        remaining = COUNTDOWN_SEC - (time.time() - countdown_start)
        if remaining <= 0:
            break
        put_overlay_text(frame, [f"Chuan bi: {remaining:.1f}s",
                                  "Dat tay vao tu the..."],
                          org=(20, 50), color=(0, 200, 255), scale=1.1)
        cv2.imshow("Thu thap du lieu VSL", frame)
        if (cv2.waitKey(1) & 0xFF) in (27, ord('q'), ord('Q')):
            return None  # thoat hoan toan

    # Quay video, luu MOI frame doc duoc kem timestamp; sau do downsample
    captured = []  # list of (timestamp, frame)
    rec_start = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        t = time.time() - rec_start
        captured.append((t, frame.copy()))

        preview = frame.copy()
        put_overlay_text(preview, ["DANG QUAY...", f"{t:.1f}/{duration:.1f}s"],
                          org=(20, 50), color=(0, 0, 255), scale=1.1)
        cv2.circle(preview, (preview.shape[1] - 40, 40), 14, (0, 0, 255), -1)
        cv2.imshow("Thu thap du lieu VSL", preview)
        if (cv2.waitKey(1) & 0xFF) in (27,):
            return None
        if t >= duration:
            break

    if len(captured) < n_frames:
        print(f"  [!] Chi doc duoc {len(captured)} frame (< {n_frames} yeu cau). "
              f"Se dung tat ca frame da quay.")
        chosen = captured
    else:
        # Chon n_frames frame trai deu theo CHI SO (khong phai thoi gian thuc,
        # de tranh thien lech do FPS khong deu)
        idxs = [round(i * (len(captured) - 1) / (n_frames - 1)) for i in range(n_frames)]
        idxs = sorted(set(idxs))
        chosen = [captured[i] for i in idxs]

    existing = count_existing_frames(label_dir)
    for k, (_, fr) in enumerate(chosen):
        fname = label_dir / f"frame_{existing + k + 1:04d}.jpg"
        cv2.imwrite(str(fname), fr, [cv2.IMWRITE_JPEG_QUALITY, 95])

    print(f"  -> Da luu {len(chosen)} frame vao {label_dir}/ "
          f"(tong cong: {count_existing_frames(label_dir)})")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--duration", type=float, default=DEFAULT_DURATION,
                     help="Thoi luong quay moi nhan (giay, mac dinh 5.0s)")
    ap.add_argument("--frames", type=int, default=DEFAULT_FRAMES,
                     help="So frame trich ra moi nhan (mac dinh 150)")
    ap.add_argument("--labels", type=str, default=None,
                     help="Danh sach nhan, cach nhau boi dau phay. "
                          "Mac dinh: toan bo 25 nhan khop checkpoint.")
    ap.add_argument("--out", type=str, default=str(DATA_DIR))
    ap.add_argument("--resume", action="store_true",
                     help="Bo qua nhan da co du --frames anh")
    args = ap.parse_args()

    labels = (args.labels.split(",") if args.labels else LABELS_DEFAULT)
    labels = [l.strip() for l in labels if l.strip()]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Khong mo duoc camera {args.camera}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print(f"Se thu thap {len(labels)} nhan, {args.frames} frame/nhan, "
          f"quay {args.duration}s/lan.")
    print("Phim tat: SPACE=quay  N=bo qua  P=nhan truoc  R=quay lai  Q/ESC=thoat\n")

    i = 0
    while 0 <= i < len(labels):
        label = labels[i]
        dirname = safe_label_dirname(label)
        label_dir = out_dir / dirname
        n_have = count_existing_frames(label_dir)

        if args.resume and n_have >= args.frames:
            print(f"[{i+1}/{len(labels)}] '{label}': da co {n_have} anh, bo qua (--resume).")
            i += 1
            continue

        # Man hinh cho lenh
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            put_overlay_text(frame, [
                f"Nhan [{i+1}/{len(labels)}]: {label}",
                f"Da co: {n_have}/{args.frames} anh",
                "SPACE: quay video  N: bo qua  P: nhan truoc",
                "R: quay lai tu dau  Q: thoat",
            ], org=(20, 45), color=(0, 255, 150), scale=0.85, line_gap=36)
            cv2.imshow("Thu thap du lieu VSL", frame)
            k = cv2.waitKey(1) & 0xFF

            if k in (27, ord('q'), ord('Q')):
                cap.release(); cv2.destroyAllWindows()
                print("Da thoat.")
                return
            elif k == 32:  # SPACE
                result = record_clip(cap, label_dir, args.duration, args.frames)
                if result is None:
                    cap.release(); cv2.destroyAllWindows()
                    print("Da thoat.")
                    return
                n_have = count_existing_frames(label_dir)
                # khong tu dong sang nhan tiep theo; nguoi dung tu quyet dinh
                # (co the quay them lan nua neu muon nhieu hon args.frames)
            elif k in (ord('n'), ord('N')):
                i += 1
                break
            elif k in (ord('p'), ord('P')):
                i = max(0, i - 1)
                break
            elif k in (ord('r'), ord('R')):
                for f in label_dir.glob("frame_*.jpg"):
                    f.unlink()
                n_have = 0
                print(f"  Da xoa toan bo anh cua nhan '{label}'.")

    cap.release()
    cv2.destroyAllWindows()

    # Tom tat cuoi cung
    print("\n=== TOM TAT THU THAP ===")
    total = 0
    for label in labels:
        d = out_dir / safe_label_dirname(label)
        n = count_existing_frames(d)
        total += n
        flag = "OK" if n >= args.frames else f"THIEU (can {args.frames})"
        print(f"  {label:6s}: {n:3d} anh  [{flag}]")
    print(f"  TONG: {total} anh / {len(labels)} nhan")
    print(f"\nDu lieu luu tai: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
