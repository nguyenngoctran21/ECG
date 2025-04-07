import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from scipy.interpolate import interp1d
from PIL import Image
import tempfile
import os
import pathlib

# === Định nghĩa mô hình ===
class BasicBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class Attention1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(channels, channels // 8, 1),
            nn.ReLU(),
            nn.Conv1d(channels // 8, 1, 1)
        )

    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=2)
        attended = torch.sum(x * weights, dim=2)
        return attended

class ResNet1D_Attn(nn.Module):
    def __init__(self, input_channels=1, num_classes=5):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv1d(input_channels, 64, 7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1)
        )
        self.layer1 = self._make_layer(64, 64, stride=1)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)

        self.attn = Attention1D(256)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256 * 2, num_classes)

    def _make_layer(self, in_c, out_c, stride):
        return nn.Sequential(
            BasicBlock1D(in_c, out_c, stride),
            BasicBlock1D(out_c, out_c)
        )

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        attn_feat = self.attn(x)
        pooled_feat = self.global_pool(x).squeeze(-1)
        x = torch.cat([attn_feat, pooled_feat], dim=1)
        return self.fc(x)

# === Cấu hình ===
LABEL_MAP = {0: 'N', 1: 'L', 2: 'R', 3: 'V', 4: 'A'}
MODEL_PATH = "ECGResNETAtt2.pth"

# === Load model từ file upload hoặc mặc định ===
@st.cache_resource
def load_model_from_file(file_path):
    model = ResNet1D_Attn(input_channels=1, num_classes=5)
    state_dict = torch.load(file_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    return model

# === Giao diện ===
st.title("🫀 Dự đoán nhịp tim từ ảnh ECG LEAD II (Wave 4)")

# Upload mô hình (tùy chọn)
uploaded_model_file = st.file_uploader("📁 Tải mô hình .pth từ máy (tùy chọn)", type=["pth"])

if uploaded_model_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tmp_model:
        tmp_model.write(uploaded_model_file.read())
        model_path = tmp_model.name
    model = load_model_from_file(model_path)
    st.info("✅ Đã sử dụng mô hình bạn tải lên.")
else:
    model = load_model_from_file(MODEL_PATH)
    # st.info("ℹ️ Đang sử dụng mô hình mặc định.")

# Upload ảnh ECG
uploaded_file = st.file_uploader("📤 Tải ảnh ECG lên (JPG hoặc PNG)", type=["jpg", "png", "jpeg"])
uploaded_filename = None

if uploaded_file is not None:
    uploaded_filename = pathlib.Path(uploaded_file.name).stem
    st.image(uploaded_file, caption="📷 Ảnh ECG gốc bạn đã tải lên", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.read())
        image_path = tmp_file.name

    ecg_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = ecg_image.shape
    cropped_ecg = ecg_image[int(0.19*h):h-int(0.05*h), int(0.06*w):w-int(0.05*w)]
    binary_ecg = np.where(cropped_ecg < 50, 0, 255).astype(np.uint8)

    wave_height = binary_ecg.shape[0] // 4
    wave_images = [binary_ecg[i * wave_height:(i + 1) * wave_height, :] for i in range(4)]
    lead_II_image = wave_images[3][:, 50:1950]

    st.image(lead_II_image, caption="🩺 LLEAD II (Wave 4) đã cắt từ ảnh", use_column_width=True, channels="GRAY")

    # Thông số chuẩn hóa
    pixel_to_mv = 1 / 10
    paper_speed = 25
    pixel_per_mm = 5
    pixel_to_ms = (1 / (paper_speed * pixel_per_mm)) * 1000
    mitbih_sample_ms = 1000 / 360
    num_samples_target = 1000

    # Trích tín hiệu
    edges = cv2.Canny(lead_II_image, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ecg_points = sorted([(x, y) for cnt in contours for x, y in cnt[:, 0]], key=lambda p: p[0])

    if not ecg_points:
        st.error("❌ Không tìm thấy điểm sóng trong ảnh!")
        st.stop()

    ecg_x, ecg_y = zip(*ecg_points)
    smoothed_y = savgol_filter(ecg_y, window_length=11, polyorder=2)
    ecg_signal = np.array(smoothed_y) * pixel_to_mv

    x_old = np.linspace(0, 1, len(ecg_signal))
    x_new = np.linspace(0, 1, num_samples_target)
    ecg_interp = interp1d(x_old, ecg_signal, kind="linear")(x_new)
    time_ms = np.arange(num_samples_target) * mitbih_sample_ms

    # Detect nhịp
    signal_smooth = savgol_filter(ecg_interp, window_length=11, polyorder=3)
    r_peaks, _ = find_peaks(signal_smooth, distance=50, prominence=0.2)

    window_size = 180
    beats_interp = []
    for r in r_peaks:
        start = r - window_size // 2
        end = r + window_size // 2
        if start >= 0 and end < len(signal_smooth):
            segment = signal_smooth[start:end]
            x_old = np.linspace(0, 1, len(segment))
            x_new = np.linspace(0, 1, 180)
            beat_interp = interp1d(x_old, segment, kind="linear")(x_new)
            beat_norm = (beat_interp - np.mean(beat_interp)) / (np.std(beat_interp) + 1e-6)
            beats_interp.append(beat_norm)

    if not beats_interp:
        st.error("❌ Không trích được nhịp tim hợp lệ!")
        st.stop()

    X_image_beats = torch.tensor(beats_interp, dtype=torch.float32).unsqueeze(1)

    # Dự đoán
    with torch.no_grad():
        outputs = model(X_image_beats)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        pred_labels = [LABEL_MAP[p] for p in preds]

    # Hiển thị kết quả
    st.success(f"✅ Trích được {len(pred_labels)} nhịp tim từ ảnh")
    st.write("### 🧾 Nhãn từng nhịp:")
    for i, lbl in enumerate(pred_labels):
        st.write(f"- Nhịp {i+1:2d}: {lbl}")

    # Vẽ waveform với nhãn
    fig, ax = plt.subplots(figsize=(18, 4))
    ax.plot(time_ms, signal_smooth, color='black', linewidth=1.2, label="ECG LEAD II")
    for i, r in enumerate(r_peaks[:len(pred_labels)]):
        t = time_ms[r]
        nhan = pred_labels[i]
        ax.axvline(x=t, color='red', linestyle='--', linewidth=1)
        ax.text(t + 5, signal_smooth[r] + 0.3, f"{nhan} ({i+1})", color='red', fontsize=9, fontweight='bold')

    ax.set_title("📈 Sóng ECG LEAD II với nhãn từng nhịp")
    ax.set_xlabel("Thời gian (ms)")
    ax.set_ylabel("Biên độ (mV)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Lưu ảnh với tên theo tên gốc
    save_name = f"{uploaded_filename}_pred.png" if uploaded_filename else "ecg_pred.png"
    fig_path = os.path.join(tempfile.gettempdir(), save_name)
    fig.savefig(fig_path, bbox_inches='tight')

    with open(fig_path, "rb") as f:
        img_bytes = f.read()

    st.download_button(
        label="📥 Tải ảnh ECG đã dán nhãn",
        data=img_bytes,
        file_name=save_name,
        mime="image/png"
    )

    # Cleanup
    os.unlink(image_path)
