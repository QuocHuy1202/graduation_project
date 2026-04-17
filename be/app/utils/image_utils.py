from PIL import Image
import io
from torchvision import transforms
from models import device # Import device từ file models.py gốc

img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def process_image(file_bytes: bytes):
    """Biến mảng byte của ảnh thành Tensor PyTorch"""
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return img_transform(image).unsqueeze(0).to(device)
    except Exception as e:
        print(f"⚠️ Lỗi xử lý ảnh: {e}")
        return None