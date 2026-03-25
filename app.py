from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image
import os, uuid, io, base64

app = Flask(__name__)
# 設定上傳暫存目錄
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# [span_4](start_span)定義遊戲截圖中的造型 X 軸座標區間[span_4](end_span)
BOXES_X = [(512, 751), (770, 1009), (1028, 1265), (1285, 1524), (1543, 1781)]
[span_5](start_span)FIXED_HEIGHT = 437  # 裁切的高度[span_5](end_span)

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
        # [span_6](start_span)根據前端傳回的 Y 座標進行裁切[span_6](end_span)
        piece = img.crop((x1, y_top, x2, y_top + FIXED_HEIGHT))
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
    
    output_width = 1200 
    cell_w = output_width // cols
    cell_h = int(cell_w * (FIXED_HEIGHT / (BOXES_X[0][1] - BOXES_X[0][0])))
    
    images_to_combine = []

    # 1. 處理封面
    if cover_name:
        cover_path = os.path.join(app.config['UPLOAD_FOLDER'], cover_name)
        if os.path.exists(cover_path):
            cover_img = Image.open(cover_path).convert('RGBA')
            aspect = cover_img.height / cover_img.width
            cover_img = cover_img.resize((output_width, int(output_width * aspect)), Image.LANCZOS)
            images_to_combine.append(cover_img)

    # 2. 繪製格子區
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

    # 3. 垂直合併所有部分
    total_h = sum(img.height for img in images_to_combine)
    final = Image.new('RGBA', (output_width, total_h))
    current_y = 0
    for img in images_to_combine:
        final.paste(img, (0, current_y), img)
        current_y += img.height

    # 4. 輸出 Base64 供前端顯示
    out_io = io.BytesIO()
    final.convert('RGB').save(out_io, 'JPEG', quality=90)
    b64 = base64.b64encode(out_io.getvalue()).decode()
    
    return jsonify({'preview': b64})

if __name__ == '__main__':
    # Render 部署時會自動覆蓋此 Port 設定
    app.run(debug=True, port=5000)
