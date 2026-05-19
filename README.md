# VSL - Vietnamese Sign Language Recognition

Nhận dạng Bảng chữ cái Ký hiệu Tiếng Việt (Vietnamese Alphabet Sign Language Recognition)

## 📋 Mô tả Dự án

Dự án này xây dựng một hệ thống nhận dạng ký hiệu tay cho bảng chữ cái Tiếng Việt sử dụng:
- **MediaPipe** - Phát hiện và trích xuất đặc điểm landmarks của bàn tay
- **PyTorch** - Deep learning models (Custom CNN & MobileNetV2)
- **OpenCV** - Xử lý video từ camera trong thời gian thực

Hệ thống có khả năng nhận dạng **25 ký hiệu** bao gồm:
- Các chữ cái: A, B, C, D, E, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, X, Y
- Các ký hiệu đặc biệt: Đ, Râu, mũ

![alt text](image.png)

## ✨ Tính năng

- ✅ Nhận dạng ký hiệu tay theo thời gian thực từ camera
- ✅ Hỗ trợ 2 mô hình: Custom CNN và MobileNetV2
- ✅ Làm mịn dự đoán bằng sliding window
- ✅ Hiển thị top K dự đoán tốt nhất
- ✅ Debug mode để hiểm skeleton landmarks
- ✅ Lưu ảnh chụp màn hình tự động

## 📦 Yêu cầu

```
Python >= 3.7
opencv-python
mediapipe >= 0.10.0
torch >= 1.9.0
torchvision >= 0.10.0
pillow
```

## 🚀 Cài đặt

### 1. Clone hoặc tải project
```bash
git clone https://github.com/TuanKiet1774/SignLanguage_Vietnamese.git
```

### 2. Cài đặt các dependencies

```bash
pip install opencv-python mediapipe torch torchvision pillow
```

### 3. Tải hand landmark model (tự động)
Lần chạy đầu tiên, model `hand_landmarker.task` sẽ được tải tự động từ:
```
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

## 💻 Sử dụng

### Chạy với mô hình mặc định (MobileNetV2)
```bash
python demo_vsl.py
```

### Chạy với Custom CNN
```bash
python demo_vsl.py --checkpoint customcnn_checkpoint.pth
```

### Chạy với MobileNetV2
```bash
demo_vsl.py --checkpoint mobilenetv2_checkpoint.pth
```

## ⌨️ Phím tắt

Trong quá trình chạy chương trình, sử dụng các phím sau:

| Phím | Chức năng |
|------|----------|
| **Q** / **ESC** | Thoát chương trình |
| **S** | Chụp ảnh → `saves/` |
| **SPACE** | Tạm dừng / Tiếp tục |
| **C** | Xóa lịch sử dự đoán |
| **D** | Bật/Tắt debug mode (hiển thị skeleton) |

## 📁 Cấu trúc Dự án

```
demo/
├── README.md                          # Tài liệu này
├── demo_vsl.py                        # Chương trình chính
├── hand_landmarker.task              # MediaPipe hand landmark model
├── label_map.json                     # Ánh xạ class names
│
├── Checkpoint Files
├── customcnn_checkpoint.pth          # Custom CNN model weights
├── mobilenetv2_checkpoint.pth        # MobileNetV2 model weights
│
├── Notebook_Kaggle/
│   ├── eda-vslv2.ipynb              # Exploratory Data Analysis
│   └── train-vslv2.ipynb            # Model training notebook
│
├── saves/                            # Thư mục lưu ảnh chụp (tự tạo)
│
└── label_map.json                    # Ánh xạ từ class label sang tên ký hiệu
```

## 📊 Label Map

File `label_map.json` chứa ánh xạ giữa class index và tên ký hiệu:

```json
{
  "A": 0, "B": 1, "C": 2, "D": 3, "E": 4,
  "G": 5, "H": 6, "I": 7, "K": 8, "L": 9,
  "M": 10, "N": 11, "O": 12, "P": 13, "Q": 14,
  "R": 15, "Râu": 16, "S": 17, "T": 18, "U": 19,
  "V": 20, "X": 21, "Y": 22, "mũ": 23, "Đ": 24
}
```

## 🧠 Mô hình

### Custom CNN
- Kiến trúc tùy chỉnh được tối ưu hóa cho bàn tay landmarks
- **File:** `customcnn_checkpoint.pth`

### MobileNetV2
- Kiến trúc nhẹ, phù hợp cho inference nhanh
- **File:** `mobilenetv2_checkpoint.pth` (mặc định)

### Cấu hình

```python
IMG_SIZE        = 224          # Kích thước ảnh input
CONF_THRESHOLD  = 0.60         # Ngưỡng confidence
SMOOTH_WINDOW   = 8            # Cửa sổ làm mịn dự đoán
TOP_K           = 3            # Số top predictions hiển thị
HAND_PAD        = 0.25         # Padding quanh vùng bàn tay
```

## 📓 Notebook Kaggle

### `eda-vslv2.ipynb`
- Phân tích dữ liệu exploratory (EDA)
- Thống kê tập dữ liệu
- Visualizations

### `train-vslv2.ipynb`
- Huấn luyện mô hình
- Đánh giá hiệu suất
- Cross-validation

### `VSL-V2 Dataset (Kaggle)`

**Nguồn:** [https://www.kaggle.com/datasets/cuongnk9104/vsl-v2](https://www.kaggle.com/datasets/cuongnk9104/vsl-v2)

#### Mô tả Dữ liệu

Tập dữ liệu VSL-V2 chứa các ảnh skeleton bàn tay được xử lý đặc biệt:

- **Định dạng:** Skeleton bàn tay trắng trên nền đen
- **Kích thước:** 224 × 224 pixels
- **Số classes:** 25 ký hiệu (Bảng chữ cái Tiếng Việt + ký hiệu đặc biệt)
- **Xử lý:** Trích xuất 21 landmarks của bàn tay bằng **MediaPipe Hands**, sau đó vẽ lại thành skeleton image

#### Cấu trúc Landmarks

MediaPipe Hands trích xuất **21 điểm mốc (landmarks)** của bàn tay:

```
Landmarks:
- 0: Wrist (cổ tay)
- 1-4: Thumb (ngón cái)
- 5-8: Index (ngón trỏ)
- 9-12: Middle (ngón giữa)
- 13-16: Ring (ngón áp út)
- 17-20: Pinky (ngón út)
```

Các landmarks được kết nối bằng các đường thẳng để tạo thành skeleton structure, hiển thị cấu trúc xương ngón tay và các điểm nối giữa các đốt ngón tay.
