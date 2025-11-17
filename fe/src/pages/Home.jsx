// src/pages/Home.jsx
import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom'; // <--- IMPORT useNavigate
import giftCardsRaw from '../assets/meta_Gift_Cards.jsonl?raw';

function Home() {
  const [products, setProducts] = useState([]);
  const navigate = useNavigate(); // <--- Hook điều hướng

  useEffect(() => {
    try {
      const parsedData = giftCardsRaw.trim().split('\n').map(line => {
          try { return JSON.parse(line); } catch (e) { return null; }
      }).filter(item => item !== null);
      setProducts(parsedData);
    } catch (error) { console.error(error); }
  }, []);

  return (
    <div>
      <header>
        <h1> Gift Cards Gallery</h1> {/* Thêm icon cho sinh động */}
      </header>
      <div className="product-grid">
        {products.map((product, index) => {
           const imageUrl = product.images?.[0]?.hi_res || product.images?.[0]?.large || 'https://placehold.co/300x400?text=No+Image';
           
           return (
            // Thêm sự kiện onClick để chuyển trang
            <div 
                key={index} 
                className="card" 
                onClick={() => navigate(`/product/${index}`)} // Chuyển đến /product/0, /product/1...
                style={{ cursor: 'pointer' }} // Thêm icon bàn tay khi hover
            >
              <div className="card-image">
                <img src={imageUrl} loading="lazy" alt={product.title} onError={(e) => e.target.src = 'https://placehold.co/300x400?text=Error'}/>
              </div>
              <div className="card-content">
                <h3 className="title">{product.title}</h3>
                <div className="rating">⭐ {product.average_rating || 'N/A'}</div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
export default Home;