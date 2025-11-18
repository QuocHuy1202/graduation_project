import torch
ITEM_METADATA = torch.load("weights/item_metadata.pt", weights_only=False)
print(f"Tổng số sản phẩm: {len(ITEM_METADATA)}")

# Kiểm tra xem index 328, 88, 450 có tồn tại không
for idx in [328, 88, 450]:
    if idx < len(ITEM_METADATA):
        print(f"✅ ID {idx} tồn tại: {ITEM_METADATA[idx].get('title', 'No Title')}")
    else:
        print(f"❌ ID {idx} KHÔNG tồn tại (Out of range)")