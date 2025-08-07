import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
from PIL import Image

st.title("🎭 AI Mood Mask Filter")
st.write("Take a selfie → AI detects emotion → overlays fun emotion mask!")

# Upload or capture image
img_data = st.camera_input("Take a selfie")

if img_data is not None:
    # Convert PIL to OpenCV format
    img_pil = Image.open(img_data)
    img_np = np.array(img_pil)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Emotion detection
    try:
        result = DeepFace.analyze(img_bgr, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        st.success(f"Detected Emotion: **{emotion.capitalize()}**")
    except Exception as e:
        st.error("Emotion detection failed.")
        st.stop()

    # Load corresponding mask
    mask_path = f"masks/{emotion.lower()}.png"
    try:
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError
    except:
        st.error(f"No mask found for {emotion}. Please add a file: `masks/{emotion.lower()}.png`")
        st.stop()

    # Resize mask to match face area (simplified — full face detection not used here)
    mask = cv2.resize(mask, (img_bgr.shape[1], img_bgr.shape[0]))

    # Overlay mask (assuming PNG with alpha)
    def overlay_transparent(background, overlay):
        if overlay.shape[2] < 4:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)

        alpha_mask = overlay[:, :, 3] / 255.0
        for c in range(0, 3):
            background[:, :, c] = background[:, :, c] * (1 - alpha_mask) + overlay[:, :, c] * alpha_mask
        return background

    output_img = overlay_transparent(img_bgr.copy(), mask)
    output_rgb = cv2.cvtColor(output_img.astype(np.uint8), cv2.COLOR_BGR2RGB)

    # Display result
    st.image(output_rgb, caption="Masked Mood", use_column_width=True)
