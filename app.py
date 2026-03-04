import os
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision import models
import torch.nn as nn

# ----------------- CONFIG -----------------
UPLOAD_FOLDER = "static/uploads"
MODEL_PATH = "model.pth"  # your trained model path
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = "change_this_secret"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------- LOAD MODEL -----------------
# Load ResNet18
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False  # freeze features

# Replace final layer
num_classes = 6  # your dataset has 6 classes
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Load saved model weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
model.to(device)

# Mapping class index to disease name
index_to_disease = {
    0: 'bacterial_leaf_blight',
    1: 'brown_spot',
    2: 'healthy',
    3: 'leaf_blast',
    4: 'leaf_scald',
    5: 'narrow_brown_spot'
}

# ----------------- TRANSFORM -----------------
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ----------------- ROUTES -----------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")  # your HTML file name

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        flash("No file part")
        return redirect(url_for("index"))

    file = request.files["image"]
    if file.filename == "":
        flash("No selected file")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Invalid file type")
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    # Load image
    image = Image.open(save_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        _, idx = torch.max(output, 1)
        pred_label = index_to_disease[idx.item()]

    image_url = url_for("static", filename=f"uploads/{filename}")
    return render_template("index.html", prediction=pred_label, image_url=image_url)

# ----------------- RUN -----------------
if __name__ == "__main__":
    app.run(debug=True)
