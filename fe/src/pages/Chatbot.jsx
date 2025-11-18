// src/pages/Chatbot.jsx
import { useState, useRef, useEffect } from 'react';
import axios from "axios";
import { useNavigate } from 'react-router-dom';
function Chatbot() {
  const navigate = useNavigate();
  const [messages, setMessages] = useState([
    { id: 1, text: "👋 Xin chào! Tôi là trợ lý AI. Nhấn dấu (+) để gửi ảnh hoặc nhập tên sản phẩm bạn cần tìm nhé.", sender: "bot" }
  ]);
  const [input, setInput] = useState("");
  const [selectedImage, setSelectedImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  
  // Sidebar States
  const [historyItems, setHistoryItems] = useState([]);
  const [loadingHistory, setLoadingHistory] = useState(true);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true); // Mặc định mở sidebar

  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);
  
  // Lấy UserID
  let userId = localStorage.getItem("currentUserId");
  if (!userId) {
    userId = "USER_" + Date.now();
    localStorage.setItem("currentUserId", userId);
  }
  useEffect(() => {
    window.openProduct = (id) => navigate(`/product/${id}`);
    return () => { delete window.openProduct; };
  }, [navigate]);
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // --- LOAD LỊCH SỬ MUA HÀNG ---
  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const res = await axios.get(`http://localhost:8000/history/${userId}`);
        setHistoryItems(res.data.history);
      } catch (error) {
        console.error("Lỗi tải lịch sử:", error);
      } finally {
        setLoadingHistory(false);
      }
    };
    fetchHistory();
  }, [userId]);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      document.getElementById("chat-input").focus();
    }
  };

  const handleSend = async () => {
    if (!input.trim() && !selectedImage) return;

    const userMsgId = Date.now();
    let userTextDisplay = input;
    if (selectedImage) {
        userTextDisplay += `<div style="margin-top: 5px;"><img src="${previewUrl}" style="width: 100px; height: 100px; object-fit: cover; border-radius: 8px; border: 1px solid #ddd;" /></div>`;
    }

    setMessages(prev => [...prev, { id: userMsgId, text: userTextDisplay, sender: "user" }]);

    const currentInput = input;
    const currentImage = selectedImage;
    setInput("");
    setSelectedImage(null);
    setPreviewUrl(null);

    const loadingId = Date.now() + 999;
    setMessages(prev => [...prev, { id: loadingId, text: "⏳ Đang tìm kiếm...", sender: "bot" }]);

    try {
      const form = new FormData();
      form.append("query_text", currentInput);
      form.append("user_id", userId);
      form.append("k_retrieval", 100); // Tìm kiếm rộng
      form.append("k_rerank", 10);     // Lấy top 10 kết quả cuối cùng
      form.append("use_rag", "true");
      if (currentImage) form.append("image", currentImage);

      const res = await axios.post("http://localhost:8000/search", form, {
        headers: { "Content-Type": "multipart/form-data" }
      });

      setMessages(prev => prev.filter(m => m.id !== loadingId));
      const results = res.data.results;
      const rag = res.data.rag_analysis;
      if (!results || results.length === 0) {
        setMessages(prev => [...prev, { id: Date.now() + 1, text: "😥 Không tìm thấy sản phẩm nào.", sender: "bot" }]);
        return;
      }
    if (rag && rag.recommendations && rag.recommendations.length > 0) {
        let ragHtml = `<div style="background: #e3f2fd; padding: 15px; border-radius: 10px; border-left: 4px solid #2196f3; margin-bottom: 15px;">`;
        
        // Lời khuyên chung
        if (rag.general_advice) {
            ragHtml += `<div style="font-style: italic; color: #555; margin-bottom: 10px;">💡 ${rag.general_advice}</div>`;
        }

        // Chi tiết từng sản phẩm
        rag.recommendations.forEach(rec => {
                ragHtml += `
                <div style="margin-top: 8px;">
                    <strong>✅ ${rec.selected_item_title}</strong>
                    <div style="font-size: 0.9rem; color: #444; margin-top: 2px;">👉 ${rec.recommendation_reason}</div>
                </div>
                `;
        });
        ragHtml += `</div>`;
        
        // Thêm tin nhắn Bot vào danh sách
        setMessages(prev => [...prev, { 
            id: Date.now() + 1, 
            text: ragHtml, 
            sender: "bot" 
        }]);
    }      
      // Lấy danh sách ID được AI chọn (nếu có)
      const recommendedIds = rag?.recommendations?.map(r => r.selected_item_index) || [];

      const replyHtml = results.map(item => {
        const scorePercent = (item.score * 100).toFixed(0);
        
        // Kiểm tra xem sản phẩm này có được AI chọn không
        const isRecommended = recommendedIds.includes(item.id);

        // Style động: Nếu được chọn thì nền xanh nhạt, viền xanh lá
        const bgStyle = isRecommended 
            ? 'background: #f6ffed; border: 2px solid #52c41a;' 
            : 'background: #fff; border-bottom: 1px solid #f0f0f0;';
        
        // Thêm nhãn (AI Pick) nếu được chọn
        const label = isRecommended 
            ? '<span style="color: #27ae60; font-weight: bold; font-size: 0.8rem; margin-left: 5px;">(AI Pick)</span>' 
            : '';

        return `
          <div onclick="window.openProduct(${item.id})" 
               style="cursor: pointer; display: flex; gap: 12px; margin-bottom: 12px; padding-bottom: 12px; padding: 10px; border-radius: 8px; ${bgStyle}">
            
            <div style="flex-shrink: 0; position: relative;">
                <img src="${item.image}" 
                     alt="img" 
                     style="width: 70px; height: 70px; object-fit: contain; border-radius: 6px; border: 1px solid #eee; background: #fff;" 
                     onerror="this.src='https://placehold.co/70?text=N/A'" 
                />
                ${isRecommended ? '<div style="position: absolute; top: -5px; left: -5px; font-size: 18px;">⭐</div>' : ''}
            </div>

            <div style="flex-grow: 1; display: flex; flex-direction: column; justify-content: center;">
                <div style="font-weight: 600; font-size: 0.95rem; color: #333; line-height: 1.3; margin-bottom: 4px;">
                    ${item.title} ${label}
                </div>
                
                <div style="display: flex; gap: 8px; align-items: center;">
                     <div style="font-size: 0.8rem; color: #27ae60; font-weight: 500;">
                        Độ phù hợp: ${scorePercent}%
                    </div>
                    <div style="font-size: 0.75rem; color: #888; background: #f1f1f1; padding: 2px 6px; borderRadius: 4px;">
                        ID: ${item.id}
                    </div>
                </div>
            </div>
          </div>
        `;
      }).join("");

      setMessages(prev => [...prev, {
        id: Date.now() + 2,
        text: `✨ Found ${results.length} products:<br/><div style="margin-top:10px">${replyHtml}</div>`,
        sender: "bot"
      }]);

    } catch (error) {
      console.error(error);
      setMessages(prev => prev.filter(m => m.id !== loadingId));
      setMessages(prev => [...prev, { id: Date.now(), text: "⚠️ Lỗi kết nối server.", sender: "bot" }]);
    }
  };

  return (
    <div className="chatbot-full-page" style={{ display: 'flex', height: '90vh', overflow: 'hidden', position: 'relative' }}>
      
      {/* --- NÚT TOGGLE SIDEBAR (Nổi lên trên) --- */}
      <button 
        onClick={() => setIsSidebarOpen(!isSidebarOpen)}
        style={{
            position: 'absolute',
            top: '15px',
            left: '15px',
            zIndex: 100,
            background: '#fff',
            border: '1px solid #ddd',
            borderRadius: '5px',
            padding: '5px 10px',
            cursor: 'pointer',
            boxShadow: '0 2px 5px rgba(0,0,0,0.1)'
        }}
      >
        {isSidebarOpen ? '◀' : '☰'}
      </button>

      {/* --- SIDEBAR LỊCH SỬ --- */}
      <div className="sidebar" style={{ 
          width: isSidebarOpen ? '300px' : '0px', // Ẩn hiện bằng width
          background: '#f8f9fa', 
          borderRight: '1px solid #ddd', 
          display: 'flex', 
          flexDirection: 'column',
          padding: isSidebarOpen ? '20px' : '0px',
          transition: 'width 0.3s ease, padding 0.3s ease', // Hiệu ứng trượt mượt mà
          overflow: 'hidden',
          whiteSpace: 'nowrap' // Tránh vỡ chữ khi thu nhỏ
      }}>
        <div style={{ marginTop: '30px', opacity: isSidebarOpen ? 1 : 0, transition: 'opacity 0.2s' }}>
            <h3 style={{ margin: '0 0 20px 0', color: '#333', fontSize: '1.2rem' }}>🛍️ Đã mua gần đây</h3>
            
            <div style={{ height: 'calc(100vh - 150px)', overflowY: 'auto' }}>
                {loadingHistory ? (
                    <div style={{ textAlign: 'center', color: '#888' }}>Đang tải...</div>
                ) : historyItems.length === 0 ? (
                    <div style={{ textAlign: 'center', color: '#888', fontStyle: 'italic' }}>Chưa có lịch sử.</div>
                ) : (
                    historyItems.map((item) => (
                        <div key={item.id} style={{ 
                            display: 'flex', 
                            alignItems: 'center', 
                            gap: '10px', 
                            marginBottom: '15px', 
                            background: '#fff', 
                            padding: '10px', 
                            borderRadius: '8px', 
                            boxShadow: '0 2px 5px rgba(0,0,0,0.05)' 
                        }}>
                            <img 
                                src={item.image} 
                                alt="thumb" 
                                style={{ width: '50px', height: '50px', objectFit: 'contain', borderRadius: '5px', border: '1px solid #eee' }} 
                                onError={(e) => e.target.src='https://placehold.co/50?text=?'}
                            />
                            <div style={{ overflow: 'hidden' }}>
                                <div style={{ fontSize: '0.9rem', fontWeight: '600', color: '#333', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }} title={item.title}>
                                    {item.title}
                                </div>
                                <div style={{ fontSize: '0.8rem', color: '#666' }}>ID: <b>{item.id}</b></div>
                            </div>
                        </div>
                    ))
                )}
            </div>
            
            <div style={{ marginTop: '20px', paddingTop: '15px', borderTop: '1px solid #ddd', fontSize: '0.85rem', color: '#666' }}>
                User: <b>{userId.length > 10 ? userId.substring(0,8)+'...' : userId}</b>
            </div>
        </div>
      </div>

      {/* --- KHUNG CHAT CHÍNH --- */}
      <div className="chat-main" style={{ flex: 1, display: 'flex', flexDirection: 'column', position: 'relative' }}>
        <div className="chat-header-modern" style={{ paddingLeft: '60px' }}> {/* Padding để tránh nút toggle đè lên */}
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
              <div className="message-bubble" dangerouslySetInnerHTML={{ __html: msg.text }}></div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        <div className="chat-input-wrapper">
            {/* Preview ảnh */}
            {previewUrl && (
                <div style={{ position: 'absolute', bottom: '80px', left: '20px', background: 'white', padding: '8px', borderRadius: '8px', boxShadow: '0 4px 12px rgba(0,0,0,0.15)', display: 'flex', alignItems: 'center', zIndex: 10 }}>
                    <img src={previewUrl} alt="Preview" style={{ height: '50px', width: '50px', objectFit: 'cover', borderRadius: '4px' }} />
                    <button onClick={() => { setSelectedImage(null); setPreviewUrl(null); }} style={{ marginLeft: '10px', border: 'none', background: '#ff4d4f', color: 'white', borderRadius: '50%', width: '20px', height: '20px', cursor: 'pointer' }}>✕</button>
                </div>
            )}

            <div className="input-box" style={{ display: 'flex', alignItems: 'center', padding: '8px 15px', background: '#fff', border: '1px solid #ddd', borderRadius: '30px' }}>
                <input type="file" ref={fileInputRef} onChange={handleFileChange} accept="image/*" style={{ display: 'none' }} />
                <button 
                    onClick={() => fileInputRef.current.click()}
                    style={{ background: '#f0f2f5', border: 'none', color: '#606770', width: '36px', height: '36px', borderRadius: '50%', fontSize: '24px', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', marginRight: '10px' }}
                    title="Thêm ảnh"
                >
                    +
                </button>

                <input
                    id="chat-input"
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && handleSend()}
                    placeholder="Nhập tin nhắn..."
                    style={{ flex: 1, border: 'none', outline: 'none', fontSize: '1rem', background: 'transparent' }}
                />
                
                <button onClick={handleSend} style={{ background: 'transparent', border: 'none', color: '#0084ff', fontSize: '20px', cursor: 'pointer', marginLeft: '10px' }}>➤</button>
            </div>
        </div>
      </div>
    </div>
  );
}

export default Chatbot;