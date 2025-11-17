import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import timm

# Config
TEXT_MODEL = "sentence-transformers/all-mpnet-base-v2"
IMG_MODEL = "vit_base_patch16_224"
D_LATENT = 256
MAX_SEQ_LEN = 40
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CÁC ENCODER CƠ BẢN (Giữ nguyên như trước) ---
class TextEncoder(nn.Module):
    def __init__(self, model_name=TEXT_MODEL, out_dim=D_LATENT):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.proj = nn.Linear(self.encoder.config.hidden_size, out_dim)

    def forward(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = self.encoder(**inputs)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        return F.normalize(self.proj(cls_emb), p=2, dim=1)

class ImageEncoder(nn.Module):
    def __init__(self, model_name=IMG_MODEL, out_dim=256):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.proj = nn.Linear(self.model.num_features, out_dim)
    
    def forward(self, images):
        features = self.model(images)
        return F.normalize(self.proj(features), p=2, dim=1)

class TabularEncoder(nn.Module): # Cho Item (không phải Review)
    def __init__(self, out_dim=256): # Giản lược input
        super().__init__()
        # Demo structure based on notebook inference
        # Cần khớp với notebook: numeric + embeddings
        self.dummy_proj = nn.Linear(10, out_dim) # Placeholder nếu không rõ input size

    def forward(self, numeric, main_cat, store, categories):
        # Logic xử lý tabular y hệt notebook
        # Vì không có code chi tiết phần này trong đoạn bạn gửi, 
        # tôi giả lập trả về vector 0 hoặc random
        B = numeric.shape[0]
        return torch.zeros(B, 256).to(device)

class FusionGate(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim, dim//2),
            nn.ReLU(),
            nn.Linear(dim//2, 1)
        )

    def forward(self, e_text, e_img, e_tab, mask=None):
        # Logic tính trọng số và cộng gộp
        g_t = self.gate_mlp(e_text)
        g_i = self.gate_mlp(e_img)
        g_b = self.gate_mlp(e_tab)
        
        if mask is not None:
            g_t = g_t.masked_fill(~mask[:, 0:1], -1e9)
            g_i = g_i.masked_fill(~mask[:, 1:2], -1e9)
            g_b = g_b.masked_fill(~mask[:, 2:3], -1e9)

        weights = F.softmax(torch.cat([g_t, g_i, g_b], dim=1), dim=1)
        
        fused = (weights[:, 0:1] * e_text + 
                 weights[:, 1:2] * e_img + 
                 weights[:, 2:3] * e_tab)
        return fused, weights

# --- REVIEW ENCODER & USER TOWER ---
class ReviewTabularEncoder(nn.Module):
    def __init__(self, num_numeric_features=2, embed_dim=4, out_dim=32):
        super().__init__()
        self.emb_verified = nn.Embedding(3, embed_dim, padding_idx=0)
        self.proj_numeric = nn.Linear(num_numeric_features, out_dim)
        self.proj_verified = nn.Linear(embed_dim, out_dim)
        self.final_proj = nn.Linear(out_dim * 2, out_dim)
        self.out_dim = out_dim
    def forward(self, ratings, votes, verified):
        e_verified = self.emb_verified(verified)
        numeric_feats = torch.cat([ratings, votes], dim=-1)
        return self.final_proj(torch.cat([F.relu(self.proj_numeric(numeric_feats)), F.relu(self.proj_verified(e_verified))], dim=-1))
# --- 5. GATING FUSION LAYER (Cho User - Tên class bạn đang thiếu) ---
class GatingFusionLayer(nn.Module):
    def __init__(self, item_emb_dim, text_emb_dim, tabular_emb_dim, d_model=D_LATENT):
        super().__init__()
        # 1. Chiếu 3 vector về d_model
        self.proj_item = nn.Linear(item_emb_dim, d_model)
        self.proj_text = nn.Linear(text_emb_dim, d_model)
        self.proj_tabular = nn.Linear(tabular_emb_dim, d_model)
        
        # 2. Mạng Gate để tính trọng số
        gate_input_dim = d_model * 3
        self.gate_net = nn.Sequential(
            nn.Linear(gate_input_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 3), # Ra 3 số (w1, w2, w3)
            nn.Softmax(dim=-1)
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, v_item, v_text, v_tabular):
        # Project về cùng không gian
        v_item_proj = self.proj_item(v_item)
        v_text_proj = self.proj_text(v_text)
        v_tabular_proj = self.proj_tabular(v_tabular)
        
        # Tính trọng số
        concat_v = torch.cat([v_item_proj, v_text_proj, v_tabular_proj], dim=-1)
        gates = self.gate_net(concat_v) # [B, S, 3]
        
        # Cộng gộp có trọng số
        weighted_v = (
            gates[..., 0:1] * v_item_proj +
            gates[..., 1:2] * v_text_proj +
            gates[..., 2:3] * v_tabular_proj
        )
        return self.layer_norm(weighted_v), gates
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)
        pe = torch.zeros(max_len, 1, d_model)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return self.dropout(x + self.pe[:x.size(0)])

import math
class UserTower(nn.Module):
    def __init__(self, item_embedding_tensor, text_encoder_model, tab_encoder_model, fusion_layer_model, d_model=D_LATENT):
        super().__init__()
        # Quan trọng: Item Embedding được nạp từ tensor bên ngoài
        self.item_embedding_lookup = nn.Embedding.from_pretrained(item_embedding_tensor, freeze=False)
        self.text_encoder = text_encoder_model
        self.step_tabular_encoder = tab_encoder_model
        self.fusion_layer = fusion_layer_model
        self.pos_encoder = PositionalEncoding(d_model, max_len=MAX_SEQ_LEN + 1)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=d_model*4, batch_first=True), 
            num_layers=2
        )
        self.final_proj = nn.Linear(d_model, d_model)

    def forward(self, batch):
        item_indices = batch["item_indices"]      
        combined_texts = batch["combined_texts"]             
        ratings = batch["ratings"].unsqueeze(-1)   
        votes = batch["helpful_votes"].unsqueeze(-1) 
        verified = batch["verified"]               
        mask = batch["transformer_mask"]           

        B, S = item_indices.shape
        v_item = self.item_embedding_lookup(item_indices)
        texts_flat = [text for user_texts in combined_texts for text in user_texts]
        v_text = self.text_encoder(texts_flat).view(B, S, -1)
        v_tabular = self.step_tabular_encoder(ratings, votes, verified) 
        
        h_seq, _ = self.fusion_layer(v_item, v_text, v_tabular) # Fusion trả về 2 giá trị
        h_seq_pos = self.pos_encoder(h_seq.permute(1, 0, 2)).permute(1, 0, 2)
        transformer_out = self.transformer_encoder(h_seq_pos, src_key_padding_mask=mask)
        
        mask_expanded = ~mask.unsqueeze(-1)
        sum_pooled = (transformer_out * mask_expanded).sum(dim=1)
        count_non_pad = mask_expanded.sum(dim=1)
        return F.normalize(self.final_proj(sum_pooled / (count_non_pad + 1e-9)), dim=-1)