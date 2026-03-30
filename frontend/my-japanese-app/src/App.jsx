import React, { useRef, useState } from 'react';
import CanvasDraw from "react-canvas-draw";
import './App.css'; 

function App() {
  const canvasRef = useRef(null);
  const fileInputRef = useRef(null);
  const [topPredictions, setTopPredictions] = useState([]);
  const [selectedPrediction, setSelectedPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  const sendToBackend = async (blob) => {
    setLoading(true);
    setSelectedPrediction(null);
    const formData = new FormData();
    formData.append("file", blob, "input.png");

    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });
      if (!response.ok) throw new Error('Backend error');
      
      const data = await response.json();
      setTopPredictions(data.top_results);
      if (data.top_results.length > 0) setSelectedPrediction(data.top_results[0].label);
      
    } catch (error) {
      console.error("Error connecting to backend:", error);
      alert("❌ Lỗi kết nối tới Backend.");
      setTopPredictions([]);
    } finally {
      setLoading(false);
    }
  };

  const handleRecognizeCanvas = async () => {
    if (!canvasRef.current) return;
    const dataUrl = canvasRef.current.getDataURL("png", false, "#ffffff");
    const blob = await (await fetch(dataUrl)).blob();
    sendToBackend(blob);
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) sendToBackend(file);
  };

  return (
    <div className="container">
      {/* HEADER CÓ NÚT TOOLBAR */}
      <header className="app-header">
        <h1>Japanese Word Detection</h1>
        <button className="btn-toolbar" onClick={() => alert("Thanh công cụ sẽ được phát triển sau!")}>
          🛠 Toolbar
        </button>
      </header>
      
      <div className="workspace">
        {/* VÙNG VẼ BÊN TRÁI */}
        <div className="canvas-section">
          <h3>Draw Kanji</h3>
          <div className="canvas-border">
            <CanvasDraw 
              ref={canvasRef}
              brushColor="#000000"      
              backgroundColor="#ffffff" 
              hideGrid={true}           
              brushRadius={5}          
              lazyRadius={0}            
              canvasWidth={320}         
              canvasHeight={320}        
            />
          </div>
          <div className="canvas-controls">
            <button onClick={handleRecognizeCanvas} disabled={loading} className="btn-predict">
              {loading ? "Analyzing..." : "Recognize"}
            </button>
            <button onClick={() => canvasRef.current.clear()} className="btn-clear">Clear</button>
          </div>
          
          <div className="upload-section">
            <p>Or upload an image:</p>
            <input type="file" accept="image/*" onChange={handleFileUpload} ref={fileInputRef}/>
          </div>
        </div>

{/* VÙNG KẾT QUẢ BÊN PHẢI */}
        <div className="result-section">
          <h3>Predictions (Top-5)</h3>
          {topPredictions.length === 0 ? (
            <div className="result-placeholder">Draw a character to see results.</div>
          ) : (
            <div className="results-container">
              {/* Ô CHỮ TO CĂN GIỮA VÀ CÓ LINK MAZII */}
              <div className="selected-result">
                <div className="main-kanji">{selectedPrediction || "?"}</div>
                {selectedPrediction && (
                  <a 
                    className="mazii-link" 
                    href={`https://mazii.net/search?dict=javi&type=w&query=${selectedPrediction}`}
                    target="_blank" 
                    rel="noopener noreferrer"
                  >
                    Tra từ điển Mazii ↗
                  </a>
                )}
              </div>
              
              {/* LƯỚI TOP-5 KẾT QUẢ RÚT GỌN */}
              <div className="top5-grid">
                {topPredictions.map((pred, index) => {
                  const confValue = pred.confidence > 1 ? pred.confidence : pred.confidence * 100;
                  return (
                    <div 
                      key={index} 
                      className={`grid-item ${selectedPrediction === pred.label ? 'active' : ''}`}
                      onClick={() => setSelectedPrediction(pred.label)}
                    >
                      <span className="kanji-char">{pred.label}</span>
                      <div className="conf-bar-bg">
                        <div className="conf-bar" style={{width: `${confValue}%`}}></div>
                      </div>
                      <span className="conf-text">{confValue.toFixed(1)}%</span>
                    </div>
                  );
                })}
              </div>
              <p className="conf-warning">(*Click vào ô chữ nhỏ để chọn)</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;