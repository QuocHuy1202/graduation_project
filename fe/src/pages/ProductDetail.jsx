// src/pages/ProductDetail.jsx
import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import giftCardsRaw from '../assets/meta_Gift_Cards.jsonl?raw';
import '../css/ProductDetail.css'; 

function ProductDetail() {
  const { id } = useParams(); // Lấy ID từ URL (đóng vai trò là item_idx luôn)
  const navigate = useNavigate();
  const [product, setProduct] = useState(null);
  const [loading, setLoading] = useState(true);
  
  // State quản lý trạng thái nút mua hàng
  const [isBuying, setIsBuying] = useState(false);

  useEffect(() => {
    try {
      const allProducts = giftCardsRaw
        .trim()
        .split('\n')
        .map(line => {
           try { return JSON.parse(line); } catch (e) { return null; }
        })
        .filter(item => item !== null);

      const foundProduct = allProducts[parseInt(id)];
      
      if (foundProduct) {
        setProduct(foundProduct);
      } else {
        navigate('/');
      }
    } catch (error) {
      console.error("Lỗi tải dữ liệu", error);
    } finally {
      setLoading(false);
    }
  }, [id, navigate]);

  // Hàm xử lý khi người dùng bấm nút Mua Ngay
// Hàm xử lý khi người dùng bấm nút Mua Ngay
  const handleBuy = async () => {
    // Lấy đúng key 'currentUserId' mà trang Login đã lưu
    const currentUserId = localStorage.getItem('currentUserId');

    // Kiểm tra xem khách đã đăng nhập chưa
    if (!currentUserId) {
      alert("Vui lòng đăng nhập để thực hiện chức năng này!");
      navigate('/login'); // Đẩy người dùng về trang đăng nhập
      return;
    }

    setIsBuying(true);
    try {
      const response = await fetch('http://127.0.0.1:8000/buy', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: currentUserId, // Truyền ID thật của người dùng
          item_idx: parseInt(id)  // Index sản phẩm
        }),
      });

      const data = await response.json();

      if (response.ok) {
        alert(`🎉 Mua hàng thành công!\nHệ thống AI: ${data.message}`);
      } else {
        alert(`❌ Lỗi từ Server: ${data.detail || data.message}`);
      }
    } catch (error) {
      console.error("Lỗi khi kết nối API Mua hàng:", error);
      alert("Không thể kết nối đến server AI. Vui lòng kiểm tra lại Backend!");
    } finally {
      setIsBuying(false);
    }
  };
  if (loading) return <div className="loading-text">Đang tải chi tiết...</div>;
  if (!product) return null;

  const imageUrl = product.images?.[0]?.hi_res || product.images?.[0]?.large || 'https://placehold.co/600x800?text=No+Image';

  return (
    <div className="detail-container">
      <button onClick={() => navigate(-1)} className="back-btn">← Quay lại</button>
      
      <div className="detail-wrapper">
        <div className="detail-image-section">
          <img src={imageUrl} alt={product.title} />
        </div>

        <div className="detail-info-section">
          <span className="detail-category">{product.main_category}</span>
          <h1 className="detail-title">{product.title}</h1>
          
          <div className="detail-rating">
            <span className="stars">⭐ {product.average_rating || 'N/A'}</span>
            <span className="count">({product.rating_number} đánh giá)</span>
          </div>

          <div className="detail-description">
            <h3>Mô tả sản phẩm</h3>
            <p>
              {product.description && product.description.length > 0 
                ? product.description.join(' ') 
                : "Chưa có mô tả chi tiết cho sản phẩm này."}
            </p>
          </div>

          <div className="detail-features">
            <h3>Đặc điểm nổi bật</h3>
            <ul>
              {product.features && product.features.map((feature, index) => (
                <li key={index}>{feature}</li>
              ))}
            </ul>
          </div>

          {/* Nút bấm đã được tích hợp API gọi Backend */}
          <button 
            className="buy-btn" 
            onClick={handleBuy}
            disabled={isBuying} // Khóa nút khi đang gửi request
            style={{ opacity: isBuying ? 0.7 : 1, cursor: isBuying ? 'wait' : 'pointer' }}
          >
            {isBuying ? 'Đang xử lý AI...' : 'Mua Ngay'}
          </button>
        </div>
      </div>
    </div>
  );
}

export default ProductDetail;