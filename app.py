from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image
import os, uuid, io, base64
import numpy as np
import cv2

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
# 新增一個存放過濾標籤圖片的資料夾
TAGS_FOLDER = os.path.join(os.path.dirname(__file__), 'filter_tags')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TAGS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TAGS_FOLDER'] = TAGS_FOLDER

# 精確 X 座標與高度
BOXES_X = [[661, 963], [988, 1290], [1315, 1618], [1644, 1946], [1973, 2274]]
FIXED_HEIGHT = 552 

def should_filter_piece(image_piece):
    """
    掃描 filter_tags 資料夾內的所有圖片。
    只要其中任一標籤比對成功（相似度 > 0.8），就回傳 True 以過濾該圖。
    """
    tag_files = [f for f in os.listdir(app.config['TAGS_FOLDER']) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not tag_files:
        return False

    try:
        # 1. 將 PIL 轉換為 OpenCV 格式 (BGR)
        target_bgr = cv2.cvtColor(np.array(image_piece.convert('RGB')), cv2.COLOR_RGB2BGR)
        
        # 2. 鎖定標籤最常出現的區域 (右上角)，縮小範圍可以提高準確率並節省效能
        # 區域設定：y軸 0~120 像素，x軸從中間到最後
        h, w = target_bgr.shape[:2]
        roi = target_bgr[0:130, int(w*0.4):w]

        for tag_name in tag_files:
            tag_path = os.path.join(app.config['TAGS_FOLDER'], tag_name)
            template = cv2.imread(tag_path)
            if template is None: continue

            # 如果標籤範本比 ROI 還大，則跳過
            if template.shape[0] > roi.shape[0] or template.shape[1] > roi.shape[1]:
                continue

            # 3. 執行範本比對
            res = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)

            # 4. 門檻值設定為 0.8 (可視情況微調)
            if max_val > 0.9:
                print(f"自動過濾：偵測到標籤 [{tag_name}]，相似度：{max_val:.2f}")
                return True
                
        return False
    except Exception as e:
        print(f"標籤比對出錯: {e}")
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/image/<filename>')
def serve_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/upload_cover', methods=['POST'])
def upload_cover():
    f = request.files.get('cover')
    if not f: return jsonify({'error': 'No file'}), 400
    fname = f'cover_{uuid.uuid4().hex}.png'
    f.save(os.path.join(app.config['UPLOAD_FOLDER'], fname))
    return jsonify({'filename': fname})

@app.route('/upload_strip', methods=['POST'])
def upload_strip():
    f = request.files.get('strip')
    y_top = int(request.form.get('y_top', 0))
    if not f: return jsonify({'error': 'No file'}), 400
    
    strip_id = uuid.uuid4().hex
    img = Image.open(f).convert('RGBA')
    results = []
    
    for i, (x1, x2) in enumerate(BOXES_X):
        # 裁切小圖
        piece = img.crop((x1, y_top, x2, y_top + FIXED_HEIGHT))
        
        # ──────────────────────────────────────────────────
        # 核心功能：檢查是否含有要過濾的標籤
        # ──────────────────────────────────────────────────
        if should_filter_piece(piece):
            continue # 如果偵測到標籤，直接跳過不儲存
            
        pname = f'p_{strip_id}_{i}.png'
        piece.save(os.path.join(app.config['UPLOAD_FOLDER'], pname))
        results.append(pname)
        
    return jsonify({'pieces': results})

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    cover_name = data.get('cover')
    cells = data.get('cells', [])
    rows = int(data.get('grid_rows', 1))
    cols = int(data.get('grid_cols', 5))
    
    output_width = 2000
    cell_w = output_width // cols
    cell_h = int(cell_w * (FIXED_HEIGHT / (BOXES_X[0][1] - BOXES_X[0][0])))
    
    images_to_combine = []

    if cover_name:
        cover_path = os.path.join(app.config['UPLOAD_FOLDER'], cover_name)
        if os.path.exists(cover_path):
            cover_img = Image.open(cover_path).convert('RGBA')
            aspect = cover_img.height / cover_img.width
            cover_img = cover_img.resize((output_width, int(output_width * aspect)), Image.LANCZOS)
            images_to_combine.append(cover_img)

    grid_h = rows * cell_h
    grid_img = Image.new('RGBA', (output_width, grid_h), (26, 26, 26, 255))
    
    for idx, fname in enumerate(cells):
        if not fname: continue
        r, c = divmod(idx, cols)
        if r >= rows: break
        
        piece_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        if os.path.exists(piece_path):
            p = Image.open(piece_path).convert('RGBA')
            p = p.resize((cell_w, cell_h), Image.LANCZOS)
            grid_img.paste(p, (c * cell_w, r * cell_h), p)
    
    images_to_combine.append(grid_img)

    total_h = sum(img.height for img in images_to_combine)
    final = Image.new('RGBA', (output_width, total_h))
    current_y = 0
    for img in images_to_combine:
        final.paste(img, (0, current_y), img)
        current_y += img.height

    out_io = io.BytesIO()
    final.convert('RGB').save(out_io, 'JPEG', quality=100, subsampling=0)
    b64 = base64.b64encode(out_io.getvalue()).decode()
    
    return jsonify({'preview': b64})

if __name__ == '__main__':
    # 確保依賴庫已安裝: pip install opencv-python numpy
    app.run(debug=True, port=5000)
