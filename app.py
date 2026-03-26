from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image
import os, uuid, io, base64

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 你提供的精確 X 座標與高度
BOXES_X = [[661, 963], [988, 1290], [1315, 1618], [1644, 1946], [1973, 2274]]
FIXED_HEIGHT = 552  # 包含「已擁有」區塊的高度

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
        piece = img.crop((x1, y_top, x2, y_top + FIXED_HEIGHT))
        pname = f'p_{strip_id}_{i}.png'
        piece.save(os.path.join(app.config['UPLOAD_FOLDER'], pname))
        results.append(pname)
    return jsonify({'pieces': results})

# ──────────────────────────────────────────────────
#  核心：合併封面與格子的路由
# ──────────────────────────────────────────────────
@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    cover_name = data.get('cover')
    cells = data.get('cells', [])
    rows = int(data.get('grid_rows', 1))
    cols = int(data.get('grid_cols', 5))
    
    # 1. 決定最終寬度 (以第一張造型卡片的寬度為準，通常約 243px * 5)
    output_width = 1200 
    cell_w = output_width // cols
    cell_h = int(cell_w * (FIXED_HEIGHT / (BOXES_X[0][1] - BOXES_X[0][0])))
    
    images_to_combine = []

    # 2. 處理封面
    if cover_name:
        cover_path = os.path.join(app.config['UPLOAD_FOLDER'], cover_name)
        if os.path.exists(cover_path):
            cover_img = Image.open(cover_path).convert('RGBA')
            # 封面縮放到跟輸出的總寬度一致
            aspect = cover_img.height / cover_img.width
            cover_img = cover_img.resize((output_width, int(output_width * aspect)), Image.LANCZOS)
            images_to_combine.append(cover_img)

    # 3. 繪製格子區
    grid_h = rows * cell_h
    grid_img = Image.new('RGBA', (output_width, grid_h), (26, 26, 26, 255)) # 深色背景
    
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

    # 4. 垂直拼貼
    total_h = sum(img.height for img in images_to_combine)
    final = Image.new('RGBA', (output_width, total_h))
    current_y = 0
    for img in images_to_combine:
        final.paste(img, (0, current_y), img)
        current_y += img.height

    # 5. 回傳 Base64 預覽
    out_io = io.BytesIO()
    final.convert('RGB').save(out_io, 'JPEG', quality=90)
    b64 = base64.b64encode(out_io.getvalue()).decode()
    
    return jsonify({'preview': b64})

if __name__ == '__main__':
    app.run(debug=True, port=5000)