// src/pages/Login.jsx
import { useState } from 'react';
import { useNavigate } from 'react-router-dom';

function Login() {
  const [userId, setUserId] = useState('');
  const navigate = useNavigate();

  const handleLogin = (e) => {
    e.preventDefault();
    if (userId.trim()) {
      // Lưu UserID vào LocalStorage để dùng cho toàn app (Chatbot, Home...)
      localStorage.setItem('currentUserId', userId);
      navigate('/'); // Chuyển về trang chủ
      window.location.reload(); // Reload nhẹ để Navbar cập nhật tên
    }
  };

  return (
    <div className="login-wrapper">
      <div className="login-card">
        <div className="login-header">
          <h2>🚀 Welcome Back</h2>
          <p>Nhập định danh của bạn để tiếp tục</p>
        </div>
        <form onSubmit={handleLogin}>
          <div className="input-group">
            <label>User ID</label>
            <input 
              type="text" 
              value={userId} 
              onChange={(e) => setUserId(e.target.value)} 
              required 
              placeholder="Ví dụ: user123"
              autoFocus
            />
          </div>
          <button type="submit" className="btn-primary btn-block">Truy cập ngay</button>
        </form>
      </div>
    </div>
  );
}

export default Login;