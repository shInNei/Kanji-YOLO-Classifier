import io
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, ImageFilter 
from ultralytics import YOLO

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# 1. LOAD MÔ HÌNH YOLO26 
model = YOLO('C:/Users/Dell G15/runs/classify/train6/weights/best.pt')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    
    # 2. XỬ LÝ ẢNH TỪ CANVAS CHỐNG LỖI MÀU
    img_rgba = Image.open(io.BytesIO(contents)).convert("RGBA")
    background = Image.new("RGBA", img_rgba.size, (255, 255, 255, 255))
    alpha_composite = Image.alpha_composite(background, img_rgba)
    img_gray = alpha_composite.convert('L')
    
    # 3. TỰ ĐỘNG CẮT SÁT CHỮ VÀ THÊM LỀ (Khớp với dữ liệu huấn luyện)
    inverted = ImageOps.invert(img_gray)
    bbox = inverted.getbbox()
    if bbox:
        img_gray = img_gray.crop(bbox)
        padding = int(max(img_gray.size) * 0.3)
        new_size = max(img_gray.size) + padding
        new_img = Image.new('L', (new_size, new_size), color=255)
        # Căn giữa chữ
        new_img.paste(img_gray, ((new_size - img_gray.size[0]) // 2, (new_size - img_gray.size[1]) // 2))
        img_gray = new_img

    # ===================================================================
    # --- MỚI: XỬ LÝ HIỆU ỨNG BÚT DẠ (GIÚP AI NHẬN DIỆN VƯƠNG/CÔNG TỐT HƠN) ---
    # Làm dày nét chữ (Lan truyền điểm đen). Số 3 là độ dày, nếu chữ vẫn mỏng bạn có thể đổi thành 5
    img_gray = img_gray.filter(ImageFilter.MinFilter(3)) 
    
    # Làm nhòe nhẹ viền mực mô phỏng độ thấm của giấy
    img_gray = img_gray.filter(ImageFilter.GaussianBlur(radius=1.5))
    # ===================================================================

    # YOLO mặc định nhận ảnh RGB nên ta chuyển từ Gray sang RGB
    img_final = img_gray.convert('RGB')
    
    # 4. DỰ ĐOÁN SIÊU TỐC VỚI YOLO26
    results = model(img_final)
    probs = results[0].probs # Lấy xác suất
    
    output = []
    # Lấy Top 5 kết quả
    for i in range(5):
        idx = probs.top5[i]
        label = model.names[idx] # YOLO tự map index ra chữ Kanji
        conf = float(probs.top5conf[i])
        output.append({"label": label, "confidence": conf})
        
    return {"status": "ok", "top_results": output}
# 1. Mở terminal, gõ: pip install fastapi uvicorn torch torchvision pillow python-multipart numpy
# 2. Gõ: uvicorn main:app --reload