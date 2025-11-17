import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import Home from './pages/Home';
import Login from './pages/Login';
import Chatbot from './pages/Chatbot';
import ProductDetail from './pages/ProductDetail'; // <--- IMPORT MỚI
import './App.css';

function App() {
  const userId = localStorage.getItem('currentUserId');

  const handleLogout = () => {
    localStorage.removeItem('currentUserId');
    window.location.href = '/login';
  };

  return (
    <Router>
      <nav className="navbar-modern">
         {/* ... (Giữ nguyên phần Navbar cũ) ... */}
          {/* ... trong phần nav ... */}
          <div className="nav-brand" style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            {/* Logo SVG vẽ trực tiếp */}
            <svg width="40" height="40" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
              <defs>
                <linearGradient id="logoGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" style={{stopColor:'#6c5ce7', stopOpacity:1}} />
                  <stop offset="100%" style={{stopColor:'#a29bfe', stopOpacity:1}} />
                </linearGradient>
              </defs>
              <rect x="15" y="20" width="25" height="70" rx="5" fill="url(#logoGrad)" />
              <rect x="60" y="20" width="25" height="70" rx="5" fill="url(#logoGrad)" />
              <rect x="15" y="45" width="70" height="15" fill="#fdcb6e" />
              <path d="M35 20 Q 20 0, 50 10 Q 80 0, 65 20" fill="#fdcb6e" />
            </svg>
            
            <span>GiftCard Store</span>
          </div>
         <div className="nav-links">
            <Link to="/">Trang chủ</Link>
            <Link to="/chat">AI Chat</Link>
            {userId ? (
               <div className="user-profile">
                  <span>Xin chào, <b>{userId}</b></span>
                  <button onClick={handleLogout} className="btn-logout">Thoát</button>
               </div>
            ) : (
               <Link to="/login" className="btn-login-nav">Đăng nhập</Link>
            )}
         </div>
      </nav>

      <div className="app-content">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/login" element={<Login />} />
          <Route path="/chat" element={<Chatbot />} />
          
          {/* --- ROUTE MỚI CHO TRANG CHI TIẾT --- */}
          {/* :id là tham số động, ví dụ /product/0, /product/1 */}
          <Route path="/product/:id" element={<ProductDetail />} /> 
        </Routes>
      </div>
    </Router>
  );
}

export default App;