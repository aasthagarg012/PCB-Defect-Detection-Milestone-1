import streamlit as st
from PIL import Image
import os
import warnings


from inference_new import (
    load_model,
    load_golden_db,
    best_golden,
    detect_and_classify,
    draw_results
)

# ======================================================
# STREAMLIT PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="PCB Defect Detection System",
    layout="wide"
)

st.title("🛠 PCB Defect Detection System")
st.write(
    "Upload a **PCB test image** to automatically detect defects "
    "using a deep learning–based inspection pipeline."
)

st.markdown("---")

# ======================================================
# LOAD BACKEND (CACHED)
# ======================================================
@st.cache_resource
def init_backend():
    model, class_names = load_model("models/best_model.pth")
    golden_db = load_golden_db("PCB_DATASET/PCB_USED")
    return model, class_names, golden_db

with st.spinner("Loading model and reference PCBs..."):
    model, class_names, golden_db = init_backend()

st.success("Backend loaded successfully ✅")

# ======================================================
# FILE UPLOADER
# ======================================================
uploaded_file = st.file_uploader(
    "📤 Upload PCB Test Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    test_img = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    # --------------------------
    # ORIGINAL IMAGE
    # --------------------------
    with col1:
        st.subheader("Original Image")
        st.image(test_img, use_column_width=True)

    # --------------------------
    # DEFECT DETECTION
    # --------------------------
    with st.spinner("Detecting defects..."):
        golden_img = best_golden(test_img, golden_db)
        detections = detect_and_classify(
            test_img,
            golden_img,
            model,
            class_names
        )
        result_img = draw_results(test_img, detections)

    # --------------------------
    # RESULT IMAGE
    # --------------------------
    with col2:
        st.subheader("Detected Defects")
        st.image(result_img, use_column_width=True)

    st.markdown("---")

    # ======================================================
    # DETECTION DETAILS (CLEAN, DEFECT-ONLY)
    # ======================================================
    st.subheader("Detection Summary")

    if detections:
        for i, d in enumerate(detections, 1):
            st.markdown(f"**Defect {i}**")
            st.write("Bounding Box:", d["box"])
            st.write("Confidence:", round(d["conf"], 3))
            st.markdown("---")
    else:
        st.success("🎉 No defects detected")

    # ======================================================
    # DOWNLOAD RESULT
    # ======================================================
    st.subheader("Download Result")

    result_path = "detected_output.png"
    result_img.save(result_path)

    with open(result_path, "rb") as f:
        st.download_button(
            label="⬇ Download Annotated Image",
            data=f,
            file_name="pcb_defect_result.png",
            mime="image/png"
        )

# ======================================================
# FOOTER
# ======================================================
st.markdown("---")
st.caption(
    "PCB Defect Detection | Milestone 3 | "
    "Deep Learning–Based Visual Inspection"
)
warnings.filterwarnings("ignore")
