"""Week 6, Session 1: visual inference demo.

A Streamlit app - upload an aircraft image, get the model's prediction,
its confidence, and its top alternative guesses. Uses final_model.pt,
the winning model from Week 5's tuning + selection process. Also includes
a separate Detect tab for the optional YOLO detection track, using
final_yolo_model.pt - finds AND classifies aircraft in a full, uncropped
scene, no manual cropping needed.

Run with: py -m streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = Path(__file__).resolve().parent / "sample_images"
SCENES_DIR = Path(__file__).resolve().parent / "sample_scenes"
sys.path.insert(0, str(ROOT / "src"))

from model import SimpleCNN, build_mobilenet  # noqa: E402

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEVICE = torch.device("cpu")

st.set_page_config(page_title="MAR20 Aircraft Recognition", page_icon="✈️", layout="wide")

# Two full palettes - green accent on white for light mode, a brighter green
# on near-black for dark mode. Picked live by the sidebar toggle in main(),
# then threaded through to every function that builds its own inline styles
# (confidence colors, chart bars) so the whole app switches consistently.
def get_palette(dark: bool) -> dict:
    if dark:
        return dict(
            bg="#141812", panel="#1E241D", border="#3A4238",
            accent="#5FAF4E", accent_dim="#3E7A32",
            text="#E8E8E8", text_dim="#A0A8A0",
            good="#4CAF50", caution="#D4A017", critical="#E05252",
            chart_neutral="#7A8870",
        )
    return dict(
        bg="#FFFFFF", panel="#F4F6F4", border="#D8D8D8",
        accent="#3E7A32", accent_dim="#2C5A24",
        text="#1A1A1A", text_dim="#5A5A5A",
        good="#2E7D32", caution="#B8860B", critical="#B71C1C",
        chart_neutral="#6E7B5E",
    )


def inject_theme(pal: dict):
    st.markdown(f"""
    <style>
    :root {{
        --app-bg: {pal['bg']}; --app-panel: {pal['panel']}; --app-border: {pal['border']};
        --app-accent: {pal['accent']}; --app-accent-dim: {pal['accent_dim']};
        --app-text: {pal['text']}; --app-text-dim: {pal['text_dim']};
    }}

    .stApp {{
        background-color: var(--app-bg);
    }}

    .stApp, .stApp p, .stApp li, .stApp label, .stMarkdown {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
        color: var(--app-text);
    }}

    h1, h2, h3 {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif !important;
        font-weight: 700 !important;
        color: var(--app-text) !important;
    }}
    .app-mono {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; }}

    [data-testid="stSidebar"] {{
        background-color: var(--app-panel);
        border-right: 2px solid var(--app-accent);
    }}
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
        color: var(--app-accent-dim) !important;
        font-size: 1rem;
        border-bottom: 2px solid var(--app-accent);
        padding-bottom: 6px;
    }}

    [data-testid="stMetric"] {{
        background: var(--app-panel);
        border: 1px solid var(--app-border);
        border-top: 3px solid var(--app-accent);
        border-radius: 2px;
        padding: 10px 12px 6px 12px;
    }}
    [data-testid="stMetricValue"] {{
        color: var(--app-accent-dim) !important;
        font-weight: 700 !important;
        font-size: 1.5rem !important;
    }}
    [data-testid="stMetricValue"], [data-testid="stMetricValue"] * {{
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
    }}
    [data-testid="stMetricLabel"] {{
        color: var(--app-text-dim) !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-size: 0.7rem !important;
    }}
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {{
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
        line-height: 1.2 !important;
    }}

    [data-baseweb="tab-list"] {{
        gap: 4px; background: var(--app-panel);
        padding: 5px; border: 1px solid var(--app-border); border-radius: 3px;
    }}
    [data-baseweb="tab"] {{
        background: transparent !important;
        color: var(--app-text-dim) !important;
        font-weight: 600 !important;
        border-radius: 2px !important;
    }}
    [data-baseweb="tab"][aria-selected="true"] {{
        background: var(--app-accent) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }}

    [data-baseweb="select"] > div {{
        background: var(--app-panel) !important;
        border-color: var(--app-border) !important;
        color: var(--app-text) !important;
    }}
    [data-baseweb="popover"] {{
        background: var(--app-panel) !important;
    }}
    [data-baseweb="menu"], [role="listbox"] {{
        background: var(--app-panel) !important;
    }}
    [role="option"] {{
        background: var(--app-panel) !important;
        color: var(--app-text) !important;
    }}
    [data-baseweb="tab-highlight"] {{ background: transparent !important; }}

    [data-testid="stFileUploader"] {{
        background: var(--app-panel);
        border: 1px dashed var(--app-border);
        border-radius: 3px;
    }}
    [data-testid="stFileUploaderDropzone"] {{
        background: var(--app-panel) !important;
    }}
    [data-testid="stFileUploaderDropzoneInstructions"] span,
    [data-testid="stFileUploaderDropzoneInstructions"] small {{
        color: var(--app-text-dim) !important;
    }}
    [data-testid="stBaseButton-secondary"] {{
        background: var(--app-bg) !important;
        color: var(--app-text) !important;
        border: 1px solid var(--app-border) !important;
    }}
    [data-testid="stBaseButton-secondary"] * {{
        color: var(--app-text) !important;
    }}
    [data-testid="stImage"] img {{
        border: 1px solid var(--app-border);
        border-radius: 2px;
    }}
    [data-testid="stAlert"] {{
        background: var(--app-panel) !important;
        border: 1px solid var(--app-border);
        border-left: 4px solid var(--app-accent);
    }}
    [data-testid="stDataFrame"] {{
        border: 1px solid var(--app-border);
        border-radius: 2px;
    }}
    hr {{ border-color: var(--app-border) !important; }}

    [data-testid="stHeader"] {{ background: var(--app-bg) !important; }}
    [data-testid="stToolbar"] {{ background: var(--app-bg) !important; }}
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_model():
    checkpoint = torch.load(ROOT / "models" / "final_model.pt", map_location=DEVICE)
    classes = checkpoint["classes"]
    image_size = checkpoint.get("image_size", 96)

    model = build_mobilenet(num_classes=len(classes)).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, classes, image_size


@st.cache_resource
def load_yolo_model():
    """Optional detection track - YOLOv8n trained in Week 5, finds and
    classifies aircraft directly in a full, uncropped scene."""
    from ultralytics import YOLO
    return YOLO(str(ROOT / "models" / "final_yolo_model.pt"))


@st.cache_resource
def load_baseline_model():
    """The very first model from Week 2, trained from scratch - used only for
    the side-by-side comparison, to show how far the project has come."""
    checkpoint = torch.load(ROOT / "models" / "baseline_cnn.pt", map_location=DEVICE)
    classes = checkpoint["classes"]
    model = SimpleCNN(num_classes=len(classes)).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, classes, 96


def predict(model, classes, image_size, image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    x = transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)[0]
    top5_probs, top5_idx = probs.topk(5)
    return [(classes[i], p.item()) for i, p in zip(top5_idx.tolist(), top5_probs)]


def confidence_color(prob: float, pal: dict) -> str:
    if prob >= 0.7:
        return pal["good"]
    if prob >= 0.4:
        return pal["caution"]
    return pal["critical"]


def plot_top5(results, pal):
    labels = [c for c, _ in reversed(results)]
    probs = [p for _, p in reversed(results)]
    colors = [confidence_color(p, pal) if i == len(probs) - 1 else pal["chart_neutral"]
              for i, p in enumerate(probs)]

    fig, ax = plt.subplots(figsize=(5, 3))
    fig.patch.set_facecolor(pal["panel"])
    ax.set_facecolor(pal["panel"])
    bars = ax.barh(labels, probs, color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Confidence", color=pal["text_dim"])
    ax.tick_params(colors=pal["text"])
    for spine in ax.spines.values():
        spine.set_color(pal["border"])
    for bar, p in zip(bars, probs):
        ax.text(p + 0.02, bar.get_y() + bar.get_height() / 2, f"{p:.1%}",
                 va="center", fontsize=9, color=pal["text"])
    fig.tight_layout()
    return fig


# A rotating earth-tone palette so the gallery badges aren't all one color
GALLERY_COLORS = ["#4B5320", "#8B5E2F", "#3F6B3F", "#6E6259", "#8C3A2B"]


def get_sample_files():
    if not SAMPLES_DIR.exists():
        return []
    return sorted(SAMPLES_DIR.glob("*.jpg"), key=lambda f: int(f.stem[1:]))


def render_gallery():
    st.subheader("The 20 aircraft classes")
    st.caption("One real example image per class. Class codes (A1-A20) are the reliable "
               "labels this project uses - see the report for why real aircraft names aren't shown here.")
    sample_files = get_sample_files()
    cols = st.columns(5)
    for i, f in enumerate(sample_files):
        color = GALLERY_COLORS[i % len(GALLERY_COLORS)]
        with cols[i % 5]:
            st.image(str(f), width="stretch")
            st.markdown(
                f"<div class='app-mono' style='background:{color};color:#FFFFFF;"
                f"text-align:center;border-radius:2px;padding:4px;font-weight:700;"
                f"letter-spacing:1px;margin-top:-8px'>{f.stem}</div>",
                unsafe_allow_html=True,
            )
            st.write("")


def render_predict_tab(model, classes, image_size, pal):
    with st.sidebar:
        st.header("About this model")
        st.caption(f"**final_model.pt** — MobileNetV2, tuned & fine-tuned — {image_size}×{image_size}px input")
        m1, m2 = st.columns(2)
        m1.metric("Test accuracy", "57.2%")
        m2.metric("Top-5 accuracy", "91.5%")
        st.divider()
        st.header("Try a sample image")
        st.caption("No file to upload? Pick one already in the project.")
        sample_files = get_sample_files()
        sample_choice = st.selectbox(
            "Sample aircraft", ["-- none --"] + [f.stem for f in sample_files]
        )

    uploaded_file = st.file_uploader("Or upload your own image", type=["jpg", "jpeg", "png"])
    compare_baseline = st.checkbox(
        "🆚 Compare with the original from-scratch baseline model (Week 2)"
    )

    image = None
    source_label = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        source_label = uploaded_file.name
    elif sample_choice != "-- none --":
        image = Image.open(SAMPLES_DIR / f"{sample_choice}.jpg")
        source_label = f"{sample_choice} (sample, true class {sample_choice})"

    if image is not None:
        col1, col2 = st.columns([1, 1.3])
        with col1:
            st.image(image, caption=source_label, width="stretch")

        with col2:
            with st.spinner("Thinking..."):
                results = predict(model, classes, image_size, image)
            top_class, top_conf = results[0]
            color = confidence_color(top_conf, pal)

            st.markdown(
                f"<div style='background:{color}22;border-left:6px solid {color};"
                f"padding:16px;border-radius:2px'>"
                f"<span style='font-size:14px;color:var(--app-text-dim)' class='app-mono'>PREDICTED CLASS</span><br>"
                f"<span style='font-size:36px;font-weight:700'>{top_class}</span>"
                f"<span style='font-size:20px;color:{color};font-weight:600'>  {top_conf:.1%}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.write("")
            st.subheader("Top 5 guesses")
            st.pyplot(plot_top5(results, pal))

        if compare_baseline:
            st.divider()
            st.subheader("🆚 Baseline model (Week 2, trained from scratch)")
            baseline_model, baseline_classes, baseline_size = load_baseline_model()
            with st.spinner("Running the old baseline..."):
                baseline_results = predict(baseline_model, baseline_classes, baseline_size, image)
            base_class, base_conf = baseline_results[0]
            base_color = confidence_color(base_conf, pal)

            bc1, bc2 = st.columns(2)
            with bc1:
                st.markdown(
                    f"<div style='background:{color}22;border-left:6px solid {color};padding:12px;border-radius:2px'>"
                    f"<span style='font-size:12px;color:var(--app-text-dim)'>FINAL MODEL (57.2% test acc.)</span><br>"
                    f"<span style='font-size:26px;font-weight:700'>{top_class}</span>"
                    f"<span style='font-size:16px;color:{color}'>  {top_conf:.1%}</span></div>",
                    unsafe_allow_html=True,
                )
            with bc2:
                st.markdown(
                    f"<div style='background:{base_color}22;border-left:6px solid {base_color};padding:12px;border-radius:2px'>"
                    f"<span style='font-size:12px;color:var(--app-text-dim)'>BASELINE (28.8% test acc.)</span><br>"
                    f"<span style='font-size:26px;font-weight:700'>{base_class}</span>"
                    f"<span style='font-size:16px;color:{base_color}'>  {base_conf:.1%}</span></div>",
                    unsafe_allow_html=True,
                )
            st.caption("Same image, two models, four project stages apart - a live look at the "
                       "improvement from transfer learning, fine-tuning, robustness, and tuning combined.")
    else:
        st.info("Upload an image above, or pick a sample from the sidebar, to see a prediction.")


def get_scene_files():
    if not SCENES_DIR.exists():
        return []
    return sorted(SCENES_DIR.glob("*.jpg"))


def render_detect_tab():
    with st.sidebar:
        st.header("About this model")
        st.caption("**final_yolo_model.pt** — YOLOv8n, optional detection track")
        d1, d2 = st.columns(2)
        d1.metric("Test precision", "72.8%")
        d2.metric("Test recall", "72.3%")
        d3, d4 = st.columns(2)
        d3.metric("Test mAP50", "77.9%")
        d4.metric("Test mAP50-95", "58.2%")
        st.divider()
        st.header("Try a sample scene")
        st.caption("A full, uncropped image with several real aircraft in it.")
        scene_files = get_scene_files()
        scene_choice = st.selectbox(
            "Sample scene", ["-- none --"] + [f.stem for f in scene_files]
        )

    st.caption("This tab uses a full satellite scene, not a single cropped aircraft like "
               "the Predict tab - the model has to find aircraft *and* name them.")
    uploaded_scene = st.file_uploader(
        "Or upload your own full scene", type=["jpg", "jpeg", "png"], key="scene_uploader"
    )
    conf_threshold = st.slider("Detection confidence threshold", 0.05, 0.90, 0.25, 0.05)

    scene_image = None
    source_label = None
    if uploaded_scene is not None:
        scene_image = Image.open(uploaded_scene).convert("RGB")
        source_label = uploaded_scene.name
    elif scene_choice != "-- none --":
        scene_image = Image.open(SCENES_DIR / f"{scene_choice}.jpg").convert("RGB")
        source_label = f"{scene_choice} (real MAR20 test image)"

    if scene_image is not None:
        yolo_model = load_yolo_model()
        with st.spinner("Scanning the scene..."):
            result = yolo_model.predict(scene_image, conf=conf_threshold, verbose=False)[0]
            annotated = result.plot()[:, :, ::-1]  # BGR -> RGB

        st.image(annotated, caption=f"{source_label} — {len(result.boxes)} aircraft found",
                  width="stretch")

        if len(result.boxes) > 0:
            st.subheader("Detections")
            det_rows = []
            for box in result.boxes:
                cls_name = result.names[int(box.cls.item())]
                conf = float(box.conf.item())
                det_rows.append({"Class": cls_name, "Confidence": f"{conf:.1%}"})
            st.dataframe(det_rows, width="stretch", hide_index=True)
        else:
            st.info("No aircraft found above this confidence threshold - try lowering it.")
    else:
        st.info("Upload a full scene above, or pick a sample from the sidebar, to run detection.")


def render_journey():
    st.subheader("From a from-scratch CNN to the final tuned model")
    journey_path = ROOT / "reports" / "figures" / "project_journey.png"
    if journey_path.exists():
        st.image(str(journey_path), width="stretch")
    st.markdown(
        "- **Baseline CNN** — trained from scratch, no pretrained knowledge\n"
        "- **Fine-tuned MobileNetV2** — pretrained model, transfer learning + fine-tuning\n"
        "- **Robust MobileNetV2** — + class weighting for rare aircraft + early stopping\n"
        "- **Final Model** — + tuned to a bigger input image size\n\n"
        "Every number here is measured on the same held-out test set - the model never saw "
        "these images during training, so this is an honest, real comparison at every stage."
    )


def main():
    with st.sidebar:
        dark_mode = st.toggle("🌙 Dark mode", value=st.session_state.get("dark_mode", False),
                                key="dark_mode")
        st.divider()
    pal = get_palette(dark_mode)
    inject_theme(pal)

    st.markdown(
        "<h1 style='margin-bottom:0'>✈️ MAR20 Aircraft Recognition</h1>"
        "<p style='color:var(--app-text-dim);margin-top:4px;'>"
        "Classify a single cropped aircraft, or detect every aircraft in a full scene.</p>",
        unsafe_allow_html=True,
    )

    model, classes, image_size = load_model()

    tab_predict, tab_detect, tab_gallery, tab_journey = st.tabs(
        ["🔍 Predict", "🎯 Detect", "📖 Aircraft Gallery", "📈 Project Journey"]
    )
    with tab_predict:
        render_predict_tab(model, classes, image_size, pal)
    with tab_detect:
        render_detect_tab()
    with tab_gallery:
        render_gallery()
    with tab_journey:
        render_journey()


if __name__ == "__main__":
    main()
