import string
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
from scipy import ndimage

st.set_page_config(page_title="Handwritten Character Recognition", page_icon="✍️", layout="centered")

DIGIT_LABELS = [str(i) for i in range(10)]
LETTER_LABELS = list(string.ascii_lowercase)


@st.cache_resource
def get_model(mode):
    path = "models/digit_model.h5" if mode == "Digits (0-9)" else "models/letter_model.h5"
    return load_model(path)


def center_on_mass(img28):

    cy, cx = ndimage.center_of_mass(img28)
    if np.isnan(cy) or np.isnan(cx):
        return img28
    rows, cols = img28.shape
    shift_y = np.round(rows / 2.0 - cy).astype(int)
    shift_x = np.round(cols / 2.0 - cx).astype(int)
    return ndimage.shift(img28, (shift_y, shift_x), cval=0)


def preprocess(canvas_rgba):
    """Canvas -> 28x28 normalized array shaped for the model."""
    img = Image.fromarray(canvas_rgba.astype("uint8"), "RGBA").convert("L")
    arr = np.array(img)

    ys, xs = np.where(arr > 20)
    if len(ys) == 0:
        return None
    pad = 15
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, arr.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, arr.shape[1])
    cropped = arr[y0:y1, x0:x1]

    resized = np.array(Image.fromarray(cropped).resize((28, 28), Image.LANCZOS))
    resized = center_on_mass(resized)

    normalized = resized.astype("float32") / 255.0
    return normalized.reshape(1, 28, 28, 1)
def main():
    st.title("✍️ Handwritten Character Recognition")
    st.caption("CNN trained on MNIST (digits) and EMNIST (letters)")

    mode = st.radio("Mode", ["Digits (0-9)", "Letters (a-z)"], horizontal=True)
    labels = DIGIT_LABELS if mode == "Digits (0-9)" else LETTER_LABELS

    col1, col2 = st.columns([1, 1])

    with col1:
        st.write("Draw a character below:")
        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=18,
            stroke_color="white",
            background_color="black",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key=f"canvas_{mode}",
        )

    with col2:
        st.write("Prediction:")
        if canvas_result.image_data is not None:
            processed = preprocess(canvas_result.image_data)
            if processed is not None:
                try:
                    model = get_model(mode)
                    preds = model.predict(processed, verbose=0)[0]
                    top_idx = np.argsort(preds)[::-1][:3]

                    top_label = labels[top_idx[0]]
                    st.metric("Predicted", top_label.upper() if mode != "Digits (0-9)" else top_label)

                    st.write("Top 3 confidence:")
                    for i in top_idx:
                        label_display = labels[i].upper() if mode != "Digits (0-9)" else labels[i]
                        st.progress(float(preds[i]), text=f"{label_display}: {preds[i]*100:.1f}%")
                except OSError:
                    st.error(
                        "Model file not found. Run train_mnist.py / train_emnist.py first "
                        "to generate models/digit_model.h5 and models/letter_model.h5."
                    )
            else:
                st.info("Draw something first ✏️")
        else:
            st.info("Draw something first ✏️")

    if st.button("Clear"):
        st.rerun()


if __name__ == "__main__":
    main()