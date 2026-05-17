// src/pages/Home.jsx
import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import giftCardsRaw from '../assets/beautymetadata.jsonl?raw';

function Home() {
  const [allProducts, setAllProducts] = useState([]); // Lưu TẤT CẢ data (không render)
  const [visibleProducts, setVisibleProducts] = useState([]); // Chỉ lưu data sẽ hiển thị lên màn hình
  const [visibleCount, setVisibleCount] = useState(24); // Số lượng render lần đầu (ví dụ: 24 item)
  
  const navigate = useNavigate();

  useEffect(() => {
    try {
      const parsedData = giftCardsRaw.trim().split('\n').map(line => {
          try { return JSON.parse(line); } catch (e) { return null; }
      }).filter(item => item !== null);
      
      setAllProducts(parsedData);
      // Lần đầu render: Chỉ cắt ra 24 item đầu tiên để hiển thị cho mượt
      setVisibleProducts(parsedData.slice(0, 24)); 
    } catch (error) { 
      console.error(error); 
    }
  }, []);

  // Hàm xử lý khi bấm nút "Tải thêm"
  const handleLoadMore = () => {
    const nextCount = visibleCount + 24; // Mỗi lần tải thêm 24 item
    setVisibleProducts(allProducts.slice(0, nextCount));
    setVisibleCount(nextCount);
  };

  return (
    <div>
      <header>
        <h1>🎁 Gift Cards Gallery</h1>
      </header>
      
      <div className="product-grid">
        {visibleProducts.map((product, index) => {
           const imageUrl = product.images?.[0]?.hi_res || product.images?.[0]?.large || 'https://placehold.co/300x400?text=No+Image';
           
           return (
             <div 
                 key={index} 
                 className="card" 
                 onClick={() => navigate(`/product/${index}`)}
                 style={{ cursor: 'pointer' }}
             >
               <div className="card-image">
                 <img 
                    src={imageUrl} 
                    loading="lazy" 
                    alt={product.title} 
                    onError={(e) => e.target.src = 'https://placehold.co/300x400?text=Error'}
                 />
               </div>
               <div className="card-content">
                 <h3 className="title">{product.title}</h3>
                 <div className="rating">⭐ {product.average_rating || 'N/A'}</div>
               </div>
             </div>
           );
        })}
      </div>

      {/* Hiển thị nút Tải thêm nếu vẫn còn sản phẩm chưa render */}
      {visibleCount < allProducts.length && (
        <div style={{ textAlign: 'center', margin: '40px 0' }}>
          <button 
            onClick={handleLoadMore}
            style={{
              padding: '12px 24px',
              fontSize: '16px',
              fontWeight: 'bold',
              cursor: 'pointer',
              borderRadius: '8px',
              border: 'none',
              backgroundColor: '#deff9a', // Màu xanh theo theme poster của bạn
              color: '#1a1a1a',
              boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
              transition: 'transform 0.2s'
            }}
            onMouseOver={(e) => e.target.style.transform = 'scale(1.05)'}
            onMouseOut={(e) => e.target.style.transform = 'scale(1)'}
          >
            Hiển thị thêm sản phẩm...
          </button>
        </div>
      )}
    </div>
  );
}

export default Home;