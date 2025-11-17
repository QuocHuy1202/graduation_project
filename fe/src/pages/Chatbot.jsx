// src/pages/Chatbot.jsx
import { useState, useRef, useEffect } from 'react';
import axios from "axios";

function Chatbot() {
  const [messages, setMessages] = useState([
    { id: 1, text: "👋 Xin chào! Tôi là trợ lý AI. Nhấn dấu (+) để gửi ảnh hoặc nhập tên sản phẩm bạn cần tìm nhé.", sender: "bot" }
  ]);

  const [input, setInput] = useState("");
  const [selectedImage, setSelectedImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  // Lấy UserID
  let userId = localStorage.getItem("currentUserId");
  if (!userId) {
    userId = "USER_" + Date.now();
    localStorage.setItem("currentUserId", userId);
  }

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      // Focus lại vào ô input sau khi chọn ảnh
      document.getElementById("chat-input").focus();
    }
  };

  const handleSend = async () => {
    if (!input.trim() && !selectedImage) return;

    // 1. Hiển thị tin nhắn User
    const userMsgId = Date.now();
    let userTextDisplay = input;
    
    // Nếu có ảnh, hiển thị ảnh nhỏ gọn
    if (selectedImage) {
        userTextDisplay += `<div style="margin-top: 5px;"><img src="${previewUrl}" style="width: 100px; height: 100px; object-fit: cover; border-radius: 8px; border: 1px solid #ddd;" /></div>`;
    }

    setMessages(prev => [...prev, { id: userMsgId, text: userTextDisplay, sender: "user" }]);

    const currentInput = input;
    const currentImage = selectedImage;

    setInput("");
    setSelectedImage(null);
    setPreviewUrl(null);

    // 2. Loading
    const loadingId = Date.now() + 999;
    setMessages(prev => [...prev, { id: loadingId, text: "⏳ Đang tìm kiếm...", sender: "bot" }]);

    try {
      const form = new FormData();
      form.append("query_text", currentInput);
      form.append("user_id", userId);
      if (currentImage) form.append("image", currentImage);

      const res = await axios.post("http://localhost:8000/search", form, {
        headers: { "Content-Type": "multipart/form-data" }
      });

      setMessages(prev => prev.filter(m => m.id !== loadingId));
      const results = res.data.results;

      if (!results || results.length === 0) {
        setMessages(prev => [...prev, { id: Date.now() + 1, text: "😥 Không tìm thấy sản phẩm nào.", sender: "bot" }]);
        return;
      }

      // 3. TẠO GIAO DIỆN KẾT QUẢ (Dạng List nhỏ gọn)
      const replyHtml = results.slice(0, 5).map(item => {
        const scorePercent = (item.score * 100).toFixed(0);
        return `
          <div style="display: flex; gap: 12px; margin-bottom: 12px; border-bottom: 1px solid #f0f0f0; padding-bottom: 12px;">
            <div style="flex-shrink: 0;">
                <img src="${item.image}" 
                     alt="img" 
                     style="width: 70px; height: 70px; object-fit: contain; border-radius: 6px; border: 1px solid #eee; background: #fff;" 
                     onerror="this.src='https://placehold.co/70?text=N/A'" 
                />
            </div>
            <div style="flex-grow: 1; display: flex; flex-direction: column; justify-content: center;">
                <div style="font-weight: 600; font-size: 0.95rem; color: #333; line-height: 1.3; margin-bottom: 4px;">
                    ${item.title}
                </div>
                <div style="font-size: 0.8rem; color: #27ae60; font-weight: 500;">
                    Độ phù hợp: ${scorePercent}%
                </div>
            </div>
          </div>
        `;
      }).join("");

      setMessages(prev => [...prev, {
        id: Date.now() + 2,
        text: `✨ Đây là kết quả tìm kiếm:<br/><div style="margin-top:10px">${replyHtml}</div>`,
        sender: "bot"
      }]);

    } catch (error) {
      console.error(error);
      setMessages(prev => prev.filter(m => m.id !== loadingId));
      setMessages(prev => [...prev, { id: Date.now(), text: "⚠️ Lỗi kết nối server.", sender: "bot" }]);
    }
  };

  return (
    <div className="chatbot-full-page">
      <div className="chat-main">
        <div className="chat-header-modern">
          <div className="bot-avatar">🤖</div>
          <div>
            <h3>AI Assistant</h3>
            <span className="status-dot"></span> Online
          </div>
        </div>

        <div className="chat-messages-area">
          {messages.map((msg) => (
            <div key={msg.id} className={`message-row ${msg.sender}`}>
              {msg.sender === 'bot' && <div className="avatar tiny-bot">🤖</div>}
              <div className="message-bubble" 
                   dangerouslySetInnerHTML={{ __html: msg.text }}>
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        {/* KHUNG NHẬP LIỆU */}
        <div className="chat-input-wrapper">
            {/* Preview ảnh trước khi gửi (nằm đè lên trên) */}
            {previewUrl && (
                <div style={{ position: 'absolute', bottom: '80px', left: '20px', background: 'white', padding: '8px', borderRadius: '8px', boxShadow: '0 4px 12px rgba(0,0,0,0.15)', display: 'flex', alignItems: 'center', zIndex: 10 }}>
                    <img src={previewUrl} alt="Preview" style={{ height: '50px', width: '50px', objectFit: 'cover', borderRadius: '4px' }} />
                    <div style={{ marginLeft: '10px', fontSize: '0.85rem', color: '#666' }}>Đã chọn 1 ảnh</div>
                    <button onClick={() => { setSelectedImage(null); setPreviewUrl(null); }} style={{ marginLeft: '10px', border: 'none', background: '#ff4d4f', color: 'white', borderRadius: '50%', width: '20px', height: '20px', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '12px' }}>✕</button>
                </div>
            )}

            <div className="input-box" style={{ display: 'flex', alignItems: 'center', padding: '8px 15px', background: '#fff', border: '1px solid #ddd', borderRadius: '30px' }}>
                
                {/* Nút Dấu Cộng (+) */}
                <input type="file" ref={fileInputRef} onChange={handleFileChange} accept="image/*" style={{ display: 'none' }} />
                <button 
                    onClick={() => fileInputRef.current.click()}
                    style={{ 
                        background: '#f0f2f5', 
                        border: 'none', 
                        color: '#606770', 
                        width: '36px', 
                        height: '36px', 
                        borderRadius: '50%', 
                        fontSize: '24px', 
                        cursor: 'pointer', 
                        display: 'flex', 
                        alignItems: 'center', 
                        justifyContent: 'center',
                        marginRight: '10px',
                        transition: '0.2s'
                    }}
                    title="Thêm ảnh"
                    onMouseOver={(e) => e.target.style.background = '#e4e6eb'}
                    onMouseOut={(e) => e.target.style.background = '#f0f2f5'}
                >
                    +
                </button>

                {/* Ô nhập text */}
                <input
                    id="chat-input"
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && handleSend()}
                    placeholder="Nhập tin nhắn..."
                    style={{ flex: 1, border: 'none', outline: 'none', fontSize: '1rem', background: 'transparent' }}
                />
                
                {/* Nút Gửi */}
                <button 
                    onClick={handleSend} 
                    style={{ 
                        background: 'transparent', 
                        border: 'none', 
                        color: '#0084ff', 
                        fontSize: '20px', 
                        cursor: 'pointer', 
                        marginLeft: '10px',
                        padding: '5px'
                    }}
                >
                    ➤
                </button>
            </div>
        </div>
      </div>
    </div>
  );
}

export default Chatbot;