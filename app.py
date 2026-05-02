import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image

st.set_page_config(page_title="AI Image Detection", layout="centered")

st.title("🌩️ AI Image Detection System")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

# Load model once
@st.cache_resource
def load_model():
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model.eval()
    return model, weights

model, weights = load_model()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Labels (NO FILE NEEDED)
labels = weights.meta["categories"]

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image")

    img_t = transform(image).unsqueeze(0)

    with st.spinner("Detecting..."):
        with torch.no_grad():
            outputs = model(img_t)

        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        top3_prob, top3_catid = torch.topk(probabilities, 3)

    st.subheader("Results:")
    for i in range(3):
        st.write(f"{labels[top3_catid[i]]} ({top3_prob[i]*100:.2f}%)")