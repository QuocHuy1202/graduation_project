import torch
from huggingface_hub import hf_hub_download
from sentence_transformers import CrossEncoder
from models import TextEncoder, ImageEncoder, FusionGate,CtrlCERanker, D_LATENT, device
from app.core.config import settings

class MLManager:
    def __init__(self):
        self.ITEM_EMBS_GPU = None
        self.USER_EMBS_GPU = None
        self.USER_MAP = {}
        self.USER_DB = {}
        self.ITEM_META = []
        self.STATS = {'C': 3.0, 'm': 10.0}
        self.NUM_ITEMS = 0
        
        self.text_encoder = None
        self.img_encoder = None
        self.fusion_gate = None
        self.cross_encoder = None

    def download_from_hf(self, filename):
        try:
            return hf_hub_download(repo_id=settings.REPO_ID, filename=filename)
        except Exception as e:
            print(f" Lỗi tải {filename} từ HF: {e}")
            return None

    def load_all_models(self):
        print(" Đang tải dữ liệu từ Hugging Face vào RAM...")
        # 1. Load Embeddings & Data
        def load_pt(name):
            path = self.download_from_hf(name)
            return torch.load(path, map_location=device, weights_only=False) if path else None

        self.ITEM_EMBS_GPU = load_pt("beauty_item_embeddings.pt")
        self.USER_EMBS_GPU = load_pt("beauty_user_embeddings.pt")
        
        if self.ITEM_EMBS_GPU is not None: self.ITEM_EMBS_GPU = self.ITEM_EMBS_GPU.to(device)
        if self.USER_EMBS_GPU is not None: self.USER_EMBS_GPU = self.USER_EMBS_GPU.to(device)
        
        self.USER_MAP = load_pt("beauty_user_map.pt")
        

        # Kiểm tra nếu tải thành công (không phải None) thì gán, nếu lỗi/không có thì gán dictionary rỗng (hoặc DataFrame rỗng)
        self.USER_DB = load_pt("beauty_user_db.pt")
        self.ITEM_META = load_pt("beauty_item_metadata.pt")
        self.STATS = load_pt("stats.pt") or {'C': 3.0, 'm': 10.0}
        self.NUM_ITEMS = self.ITEM_EMBS_GPU.shape[0] if self.ITEM_EMBS_GPU is not None else 0

        # 2. Load PyTorch Models
        self.text_encoder = TextEncoder().to(device)
        self.img_encoder = ImageEncoder().to(device)
        self.fusion_gate = FusionGate(D_LATENT).to(device)

        def load_weights(model, name):
            path = self.download_from_hf(name)
            if path: model.load_state_dict(torch.load(path, map_location=device, weights_only=False), strict=False)

        load_weights(self.text_encoder, "beauty_text_encoder.pth")
        load_weights(self.img_encoder, "beauty_img_encoder.pth")
        load_weights(self.fusion_gate, "beauty_fusion_gate.pth")

        self.text_encoder.eval()
        self.img_encoder.eval()
        self.fusion_gate.eval()

        # 3. Load Cross-Encoder
        # 3. Load Custom Cross-Encoder (CtrlCE)
        print(" Đang tải Custom Cross-Encoder (CtrlCE)...")
        # Tải file trọng số của mô hình ranker từ repo Hugging Face
        ctrlce_weights_path = self.download_from_hf("beauty_ctrlce_weights.pth")
        
        # Khởi tạo mô hình
        self.cross_encoder = CtrlCERanker(weights_path=ctrlce_weights_path, device=device)
        
        print(" Toàn bộ Model đã sẵn sàng!")

# Khởi tạo biến toàn cục để các file khác import dùng chung
ml = MLManager()