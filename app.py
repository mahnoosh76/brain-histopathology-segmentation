import streamlit as st
import torch
import cv2
import numpy as np
from unet import UNet  # مدل خودت

# ===== تنظیمات پایه =====
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = 'final_model.pth'
INPUT_SIZE = (256, 256)
N_CLASSES = 6

# ===== رنگ‌ها برای نمایش ماسک =====
COLOR_MAP = {
    0: (0,   0,   0),
    1: (255, 0,   0),
    2: (0,   255, 0),
    3: (0,   0,   255),
    4: (255, 255, 0),
    5: (255, 0,   255),
}

# ===== تابع برای رنگی‌کردن ماسک =====
def colorize_mask(mask):
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_val, color in COLOR_MAP.items():
        color_mask[mask == class_val] = color
    return color_mask

# ===== بارگذاری مدل =====
@st.cache_resource
def load_model():
    model = UNet(in_channels=3, out_channels=N_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# ===== پردازش ورودی =====
def preprocess_image(image):
    image = cv2.resize(image, INPUT_SIZE)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # HWC → CHW
    tensor = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
    return tensor

# ===== پیش‌بینی =====
def predict(image_np):
    tensor = preprocess_image(image_np)
    with torch.no_grad():
        output = model(tensor)  # (1, C, H, W)
        pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy().astype(np.uint8)
    return pred_mask

# ===== رابط کاربری Streamlit =====
st.set_page_config(page_title="تشخیص ضایعات مغزی", layout="centered")
st.title(" اپلیکیشن تشخیص ضایعات مغزی ")

uploaded_file = st.file_uploader("یک تصویر MRI آپلود کن", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    col1, col2 = st.columns(2)

    with col1:
            st.image(image, caption="تصویر ورودی", use_column_width=True)

    if st.button("📊 پیش‌بینی کن"):
        pred_mask = predict(image)
        pred_colored = colorize_mask(pred_mask)
        with col2:
            st.image(pred_colored, caption="نتیجه سگمنتیشن (رنگی)", use_column_width=True)

        with st.expander("🔍 توضیح رنگ‌ها"):
            for class_id, color in COLOR_MAP.items():
                st.markdown(f"<span style='color:rgb{color};'>⬛</span> کلاس {class_id}", unsafe_allow_html=True)
