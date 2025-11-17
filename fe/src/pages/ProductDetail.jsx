// src/pages/ProductDetail.jsx
import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import giftCardsRaw from '../assets/meta_Gift_Cards.jsonl?raw';
import '../css/ProductDetail.css'; 

function ProductDetail() {
  const { id } = useParams(); // Lấy ID từ URL (ví dụ: /product/0)
  const navigate = useNavigate();
  const [product, setProduct] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Load dữ liệu lại để đảm bảo có thông tin ngay cả khi refresh trang
    try {
      const allProducts = giftCardsRaw
        .trim()
        .split('\n')
        .map(line => {
           try { return JSON.parse(line); } catch (e) { return null; }
        })
        .filter(item => item !== null);

      // Tìm sản phẩm theo index
      const foundProduct = allProducts[parseInt(id)];
      
      if (foundProduct) {
        setProduct(foundProduct);
      } else {
        // Nếu không tìm thấy (ví dụ nhập ID bậy bạ), quay về trang chủ
        navigate('/');
      }
    } catch (error) {
      console.error("Lỗi tải dữ liệu", error);
    } finally {
      setLoading(false);
    }
  }, [id, navigate]);

  if (loading) return <div className="loading-text">Đang tải chi tiết...</div>;
  if (!product) return null;

  // Lấy ảnh độ phân giải cao nhất
  const imageUrl = product.images?.[0]?.hi_res || product.images?.[0]?.large || 'https://placehold.co/600x800?text=No+Image';

  return (
    <div className="detail-container">
      <button onClick={() => navigate(-1)} className="back-btn">← Quay lại</button>
      
      <div className="detail-wrapper">
        {/* Cột trái: Ảnh */}
        <div className="detail-image-section">
          <img src={imageUrl} alt={product.title} />
        </div>

        {/* Cột phải: Thông tin */}
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

          <button className="buy-btn" onClick={() => alert('Tính năng Mua hàng đang phát triển!')}>
            Mua Ngay
          </button>
        </div>
      </div>
    </div>
  );
}

export default ProductDetail;