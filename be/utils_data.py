import torch
import numpy as np
from torch.utils.data import Dataset

PAD_VALUE = 0
MAX_SEQ_LEN = 40

# Hàm tính Weighted Rating (Cold Start Logic)
def calculate_weighted_rating(v, R, m, C):
    # v: số lượng vote, R: rating trung bình, m: min votes, C: mean vote toàn server
    return (v / (v + m) * R) + (m / (v + m) * C)

# Class xử lý dữ liệu user để đưa vào UserTower
def prepare_user_batch(item_indices, reviews, ratings, votes, verified):
    # 1. Cắt bớt (Truncate)
    seq_len = min(len(item_indices), MAX_SEQ_LEN)
    
    input_indices = item_indices[-seq_len:]
    input_texts = reviews[-seq_len:]
    input_ratings = ratings[-seq_len:]
    input_votes = votes[-seq_len:]
    input_verified = verified[-seq_len:]

    padding_len = MAX_SEQ_LEN - seq_len

    # 2. Đệm (Padding) - Thêm vào bên TRÁI (Left Padding cho Transformer)
    padded_indices = [PAD_VALUE] * padding_len + input_indices
    padded_texts = [""] * padding_len + input_texts
    padded_ratings = [PAD_VALUE] * padding_len + input_ratings
    padded_votes = [PAD_VALUE] * padding_len + input_votes
    # verified: 0 là padding, 1 là verified, 2 là not verified
    # Input verified gốc thường là 0/1 -> map sang 1/2
    verified_mapped = [v + 1 for v in input_verified] 
    padded_verified = [PAD_VALUE] * padding_len + verified_mapped
    
    transformer_mask = torch.tensor([True] * padding_len + [False] * seq_len, dtype=torch.bool)

    return {
        "item_indices": torch.tensor([padded_indices], dtype=torch.long), # Batch size 1
        "combined_texts": [padded_texts],
        "ratings": torch.tensor([padded_ratings], dtype=torch.float),
        "helpful_votes": torch.tensor([padded_votes], dtype=torch.float),
        "verified": torch.tensor([padded_verified], dtype=torch.long),
        "transformer_mask": torch.tensor([transformer_mask], dtype=torch.bool)
    }