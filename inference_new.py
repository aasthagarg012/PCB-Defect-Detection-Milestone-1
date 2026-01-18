# inference_new.py
import os
import torch
import numpy as np
from torchvision import models, transforms
from torchvision.ops import nms
from PIL import Image, ImageDraw
import imagehash
from skimage.metrics import structural_similarity as ssim

# =========================
# CONFIG
# =========================
DEVICE = torch.device("cpu")  # keep CPU for stability
IMG_SIZE = 128
WINDOW = 128
STRIDE = 32
SSIM_THRESH = 0.95
CONF_THRESH = 0.80

# =========================
# LOAD MODEL
# =========================
def load_model(model_path):
    checkpoint = torch.load(model_path, map_location=DEVICE)

    class_names = checkpoint.get("class_names", ["defect"])
    num_classes = len(class_names)

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(
        model.classifier[1].in_features,
        num_classes
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model, class_names

# =========================
# TRANSFORMS
# =========================
tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# =========================
# GOLDEN PCB DATABASE
# =========================
def load_golden_db(golden_dir):
    db = []
    for f in os.listdir(golden_dir):
        if f.lower().endswith((".jpg", ".png", ".jpeg")):
            img = Image.open(os.path.join(golden_dir, f)).convert("RGB")
            db.append({
                "img": img,
                "hash": imagehash.phash(img)
            })
    return db

def best_golden(test_img, golden_db):
    h = imagehash.phash(test_img)
    return min(golden_db, key=lambda x: h - x["hash"])["img"]

# =========================
# DEFECT DETECTION (RESTORED)
# =========================
def detect_and_classify(test_img, golden_img, model, class_names):
    detections = []
    w, h = test_img.size

    for y in range(0, h - WINDOW + 1, STRIDE):
        for x in range(0, w - WINDOW + 1, STRIDE):

            tp = test_img.crop((x, y, x + WINDOW, y + WINDOW))
            gp = golden_img.crop((x, y, x + WINDOW, y + WINDOW))

            score = ssim(
                np.array(gp.convert("L")),
                np.array(tp.convert("L"))
            )

            if score < SSIM_THRESH:
                inp = tf(tp).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    probs = torch.softmax(model(inp), dim=1)
                    conf, _ = torch.max(probs, 1)

                if conf.item() > CONF_THRESH:
                    detections.append({
                        "box": [x, y, x + WINDOW, y + WINDOW],
                        "label": "Defect",
                        "conf": conf.item()
                    })

    if not detections:
        return []

    boxes = torch.tensor([d["box"] for d in detections], dtype=torch.float32)
    scores = torch.tensor([d["conf"] for d in detections])
    keep = nms(boxes, scores, iou_threshold=0.3)

    return [detections[i] for i in keep]

# =========================
# DRAW RESULTS
# =========================
def draw_results(img, detections):
    img = img.copy()
    draw = ImageDraw.Draw(img)

    for d in detections:
        draw.rectangle(d["box"], outline="red", width=3)
        draw.text(
            (d["box"][0], d["box"][1] - 12),
            "Defect",
            fill="red"
        )
    return img

if __name__ == "__main__":
    print("Inference module loaded successfully ✔")
