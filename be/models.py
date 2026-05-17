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
# app/core/ctrl_ce.py (hoặc để chung trong models.py)
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import numpy as np
from typing import List, Dict

class CtrlCEItemMemory_TwoTowers(nn.Module):
    # ... (Bác copy y nguyên hàm __init__ và forward của class này như code cũ)
    def __init__(self):
        super(CtrlCEItemMemory_TwoTowers, self).__init__()
        self.enc_ce = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        self.hidden_size = self.enc_ce.config.hidden_size
        self.enc_mem = AutoModel.from_pretrained("sentence-transformers/multi-qa-mpnet-base-cos-v1")
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.mixing_mlp = nn.Sequential(
            nn.Linear(self.hidden_size + 4, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid() 
        )

    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, ce_input_ids, ce_attention_mask, query_mask, doc_mask, mem_input_ids, mem_attention_mask, phase=2):
        with torch.no_grad(): # Bắt buộc no_grad khi inference
            # --- Cross-Encoder ---
            ce_outputs = self.enc_ce(input_ids=ce_input_ids, attention_mask=ce_attention_mask)
            q = self.mean_pooling(ce_outputs.last_hidden_state, query_mask)
            d = self.mean_pooling(ce_outputs.last_hidden_state, doc_mask)   
            
            q_norm = F.normalize(q, p=2, dim=1)
            d_norm = F.normalize(d, p=2, dim=1)
            
            scale = torch.clamp(self.logit_scale.exp(), max=100.0)
            s_search = torch.sum(q_norm * d_norm, dim=1) * scale

            # --- Memory-Encoder ---
            batch_size, num_memories, seq_len = mem_input_ids.size()
            flat_mem_input_ids = mem_input_ids.reshape(-1, seq_len)
            flat_mem_attention_mask = mem_attention_mask.reshape(-1, seq_len)
            
            mem_outputs = self.enc_mem(input_ids=flat_mem_input_ids, attention_mask=flat_mem_attention_mask)
            flat_V = self.mean_pooling(mem_outputs.last_hidden_state, flat_mem_attention_mask)
            flat_V_norm = F.normalize(flat_V, p=2, dim=1)
            
            V = flat_V_norm.reshape(batch_size, num_memories, self.hidden_size)
            
            d_norm_expanded = d_norm.unsqueeze(2) 
            personal_scores_matrix = torch.bmm(V, d_norm_expanded).squeeze(2) 
            s_personal, _ = torch.max(personal_scores_matrix, dim=1) 
            s_personal = s_personal * scale

            if phase == 1:
                return s_search, s_personal

            # --- Phase 2: Mixing MLP ---
            len_q = query_mask.sum(dim=1, keepdim=True).float()
            valid_memories = (mem_attention_mask.sum(dim=2) > 0).sum(dim=1, keepdim=True).float() 
            
            mlp_input = torch.cat([
                q_norm, 
                len_q, 
                valid_memories,
                s_search.unsqueeze(1), 
                s_personal.unsqueeze(1)
            ], dim=1)
            
            w = self.mixing_mlp(mlp_input).squeeze(1) 
            s_total = w * s_search + (1 - w) * s_personal
            
            return s_total, w, s_search, s_personal

class CtrlCERanker:
    # Đã sửa lại __init__ để tương thích với cấu trúc HuggingFace tải từng file
    def __init__(self, weights_path: str, device: str):
        self.device = device
        #print(f" Khởi tạo CtrlCE Ranker trên: {self.device.upper()}")
        
        # Tải thẳng Tokenizer từ Hugging Face Hub (do base trên MPNet)
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2", use_fast=True)
        
        self.model = CtrlCEItemMemory_TwoTowers()
        
        # Nạp trọng số tải từ HF về
        if weights_path:
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=False), strict=False)
            
        self.model = self.model.to(self.device)
        self.model.eval()

    def _prepare_history_tensors(self, history: List[str], max_length: int = 64):
        """Tiền xử lý Lịch sử người dùng thành Tensors"""
        mem_input_ids_list, mem_attention_mask_list = [], []
        # Xử lý trường hợp User mới (Cold Start - Lịch sử rỗng)
        history_items = history if history else ["empty history"]
        
        for item in history_items[-5:]: # Chỉ lấy 5 món gần nhất để tránh lag
            mem_enc = self.tokenizer(
                item.strip(), max_length=max_length, padding='max_length', truncation=True, return_tensors='pt'
            )
            mem_input_ids_list.append(mem_enc['input_ids'].squeeze(0))
            mem_attention_mask_list.append(mem_enc['attention_mask'].squeeze(0))
            
        mem_input_ids = torch.stack(mem_input_ids_list).unsqueeze(0).to(self.device)
        mem_attention_mask = torch.stack(mem_attention_mask_list).unsqueeze(0).to(self.device)
        return mem_input_ids, mem_attention_mask

    def rank(self, query: str, history: List[str], candidates: List[str], batch_size: int = 32) -> List[Dict]:
        """
        Xếp hạng danh sách ứng viên dựa trên Query và Lịch sử.
        
        Args:
            query (str): Câu tìm kiếm của người dùng.
            history (List[str]): Danh sách tên sản phẩm người dùng đã mua trong quá khứ.
            candidates (List[str]): Danh sách 50-100 sản phẩm ứng viên cần xếp hạng lại (từ ElasticSearch/BM25).
            batch_size (int): Số lượng item xử lý mỗi lượt để chống tràn RAM.
            
        Returns:
            List[Dict]: Danh sách kết quả đã được sắp xếp từ cao xuống thấp.
        """
        if not candidates:
            return []

        # 1. Chuẩn bị Vector Ký ức (Chỉ làm 1 lần cho mỗi Request)
        mem_input_ids_base, mem_attention_mask_base = self._prepare_history_tensors(history)
        
        all_results = []
        
        # 2. Xử lý Candidates theo từng Batch nhỏ
        for i in range(0, len(candidates), batch_size):
            chunk_cands = candidates[i : i + batch_size]
            chunk_size = len(chunk_cands)
            
            # Xử lý Cross-Encoder
            ce_encoded = self.tokenizer(
                [query] * chunk_size, chunk_cands, 
                max_length=128, padding='max_length', truncation=True, return_tensors='pt'
            )
            ce_input_ids = ce_encoded['input_ids'].to(self.device)
            ce_attention_mask = ce_encoded['attention_mask'].to(self.device)
            
            # Tạo Masks cho Query và Document
            q_mask_list, d_mask_list = [], []
            for b in range(chunk_size):
                seq_ids = ce_encoded.sequence_ids(b)
                q_mask_list.append([1.0 if s == 0 else 0.0 for s in seq_ids])
                d_mask_list.append([1.0 if s == 1 else 0.0 for s in seq_ids])
                
            query_mask = torch.tensor(q_mask_list, dtype=torch.float).to(self.device)
            doc_mask = torch.tensor(d_mask_list, dtype=torch.float).to(self.device)
            
            # Lặp lại Lịch sử cho khớp với số lượng dòng trong Batch
            mem_input_ids_batch = mem_input_ids_base.expand(chunk_size, -1, -1).contiguous()
            mem_attention_mask_batch = mem_attention_mask_base.expand(chunk_size, -1, -1).contiguous()
            
            # 3. Chạy Mô hình (Phase 2)
            with torch.amp.autocast('cuda') if self.device == 'cuda' else torch.no_grad():
                s_total, w, s_search, s_personal = self.model(
                    ce_input_ids, ce_attention_mask, query_mask, doc_mask,
                    mem_input_ids_batch, mem_attention_mask_batch, phase=2
                )
            
            # Lấy giá trị ra khỏi GPU
            t_scores = s_total.cpu().tolist()
            w_vals = w.cpu().tolist()
            
            for idx, item in enumerate(chunk_cands):
                all_results.append({
                    "item": item,
                    "final_score": t_scores[idx],
                    "search_weight_percent": round(w_vals[idx] * 100, 2) # Dành cho App hiển thị Debug
                })
                
        # 4. Sắp xếp kết quả từ cao xuống thấp dựa vào final_score
        all_results.sort(key=lambda x: x['final_score'], reverse=True)
        return all_results
    def predict_raw(self, query: str, history: List[str], candidates: List[str], batch_size: int = 32) -> torch.Tensor:
        """
        Trả về Raw Tensor điểm số của danh sách candidates. 
        KHÔNG sắp xếp để giữ nguyên Index Mapping với kết quả của Two-Tower.
        """
        if not candidates:
            return torch.tensor([]).to(self.device)

        # 1. Encode Lịch sử (Chỉ làm 1 lần cho toàn bộ batch)
        mem_input_ids_base, mem_attention_mask_base = self._prepare_history_tensors(history)
        all_scores = []
        
        # 2. Xử lý Candidates theo từng batch
        for i in range(0, len(candidates), batch_size):
            chunk_cands = candidates[i : i + batch_size]
            chunk_size = len(chunk_cands)
            
            ce_encoded = self.tokenizer(
                [query] * chunk_size, chunk_cands, 
                max_length=128, padding='max_length', truncation=True, return_tensors='pt'
            )
            ce_input_ids = ce_encoded['input_ids'].to(self.device)
            ce_attention_mask = ce_encoded['attention_mask'].to(self.device)
            
            # Tạo Mask phân tách Query và Document
            q_mask_list, d_mask_list = [], []
            for b in range(chunk_size):
                seq_ids = ce_encoded.sequence_ids(b)
                q_mask_list.append([1.0 if s == 0 else 0.0 for s in seq_ids])
                d_mask_list.append([1.0 if s == 1 else 0.0 for s in seq_ids])
                
            query_mask = torch.tensor(q_mask_list, dtype=torch.float).to(self.device)
            doc_mask = torch.tensor(d_mask_list, dtype=torch.float).to(self.device)
            
            mem_input_ids_batch = mem_input_ids_base.expand(chunk_size, -1, -1).contiguous()
            mem_attention_mask_batch = mem_attention_mask_base.expand(chunk_size, -1, -1).contiguous()
            
            # Inference
            with torch.amp.autocast('cuda') if self.device == 'cuda' else torch.no_grad():
                s_total, _, _, _ = self.model(
                    ce_input_ids, ce_attention_mask, query_mask, doc_mask,
                    mem_input_ids_batch, mem_attention_mask_batch, phase=2
                )
            
            all_scores.append(s_total)
            
        # Nối tất cả điểm lại thành 1 Tensor 1 chiều
        return torch.cat(all_scores, dim=0)