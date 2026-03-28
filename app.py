from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image
import os, uuid, io, base64
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
TAGS_FOLDER = os.path.join(os.path.dirname(__file__), 'filter_tags')
SORT_TAGS_FOLDER = os.path.join(os.path.dirname(__file__), 'sort_tags')
ANCHOR_FOLDER = os.path.join(os.path.dirname(__file__), 'anchor')

# 確保所有必要資料夾都存在
for folder in [UPLOAD_FOLDER, TAGS_FOLDER, SORT_TAGS_FOLDER, ANCHOR_FOLDER]:
    os.makedirs(folder, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TAGS_FOLDER'] = TAGS_FOLDER
app.config['SORT_TAGS_FOLDER'] = SORT_TAGS_FOLDER
app.config['ANCHOR_FOLDER'] = ANCHOR_FOLDER

# 定義裁切座標與高度
BOXES_X = [[661, 963], [988, 1290], [1315, 1618], [1644, 1946], [1973, 2274]]
FIXED_HEIGHT = 552

def cv_imread_unicode(file_path):
    """支援中文路徑與檔名的 OpenCV 讀取方式"""
    try:
        return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"讀取圖檔失敗 ({file_path}): {e}")
        return None

def get_matching_info(image_piece):
    """
    掃描資料夾，同時進行過濾檢查與權重計算
    回傳: (是否應過濾, 權重分數)
    """
    # 轉換 PIL 為 OpenCV 格式
    target_bgr = cv2.cvtColor(np.array(image_piece.convert('RGB')), cv2.COLOR_RGB2BGR)
    h, w = target_bgr.shape[:2]
    # 鎖定標籤常出現的右上角 ROI 區域 (y:0-130, x:40%寬度到最後)
    roi = target_bgr[0:130, int(w*0.4):w]

    # 1. 檢查是否需要過濾 (filter_tags)
    filter_files = [f for f in os.listdir(app.config['TAGS_FOLDER'])
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('.')]

    for tag_name in filter_files:
        template = cv_imread_unicode(os.path.join(app.config['TAGS_FOLDER'], tag_name))
        if template is None: continue
        if template.shape[0] > roi.shape[0] or template.shape[1] > roi.shape[1]: continue

        res = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val > 0.9:
            print(f"命中過濾標籤: {tag_name} ({max_val:.2f})")
            return True, 0

    # 2. 計算權重分數 (sort_tags)
    max_weight = 0
    sort_files = [f for f in os.listdir(app.config['SORT_TAGS_FOLDER'])
                  if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('.')]

    for sf in sort_files:
        template = cv_imread_unicode(os.path.join(app.config['SORT_TAGS_FOLDER'], sf))
        if template is None: continue
        if template.shape[0] > roi.shape[0] or template.shape[1] > roi.shape[1]: continue

        res = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)

        if max_val > 0.8:
            try:
                # 解析檔名 "tag_090.png" 取得 90
                weight = int(sf.split('_')[-1].split('.')[0])
                max_weight = max(max_weight, weight)
            except: pass

    return False, max_weight

@app.route('/')
def index():
    return render_template('index-2622.html')

@app.route('/image/<filename>')
def serve_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/upload_anchor', methods=['POST'])
def upload_anchor():
    """儲存錨點圖片，供 detect_y 做 template matching 用"""
    f = request.files.get('anchor')
    if not f: return jsonify({'error': 'No file'}), 400
    anchor_path = os.path.join(app.config['ANCHOR_FOLDER'], 'anchor.png')
    Image.open(f).convert('RGB').save(anchor_path)
    return jsonify({'ok': True})

@app.route('/detect_y', methods=['POST'])
def detect_y():
    """
    在 strip 圖中尋找錨點，回傳對齊後的 y_top。
    y_top = 錨點頂部 Y + y_offset
    """
    f = request.files.get('strip')
    y_offset = int(request.form.get('y_offset', 0))
    if not f: return jsonify({'error': 'No file'}), 400

    anchor_path = os.path.join(app.config['ANCHOR_FOLDER'], 'anchor.png')
    if not os.path.exists(anchor_path):
        return jsonify({'error': 'No anchor uploaded'}), 400

    # 讀取 strip（來自記憶體流，不存檔）
    img_data = np.frombuffer(f.read(), dtype=np.uint8)
    strip_bgr = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    if strip_bgr is None:
        return jsonify({'error': 'Cannot decode strip image'}), 400

    template = cv_imread_unicode(anchor_path)
    if template is None:
        return jsonify({'error': 'Cannot read anchor'}), 400

    strip_h = strip_bgr.shape[0]
    # 只掃描下半部
    half_y = strip_h // 2
    search_region = strip_bgr[half_y:, :]

    if template.shape[0] > search_region.shape[0] or template.shape[1] > search_region.shape[1]:
        return jsonify({'error': 'Anchor image is larger than search region'}), 400

    res = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)

    if max_val < 0.7:
        return jsonify({'found': False, 'score': round(float(max_val), 3)})

    # max_loc[1] 是相對於下半部的 Y，加回 half_y 換算成全圖座標
    anchor_y = max_loc[1] + half_y
    y_top = int(np.clip(anchor_y + y_offset, 0, strip_h - FIXED_HEIGHT))
    return jsonify({'found': True, 'y_top': y_top, 'score': round(float(max_val), 3)})

@app.route('/upload_cover', methods=['POST'])
def upload_cover():
    f = request.files.get('cover')
    if not f: return jsonify({'error': 'No file'}), 400
    fname = f'cover_{uuid.uuid4().hex}.png'
    f.save(os.path.join(app.config['UPLOAD_FOLDER'], fname))
    return jsonify({'filename': fname})

def _process_piece(args):
    """裁切單一格並進行過濾/權重比對，供多執行緒呼叫。"""
    img, i, x1, x2, y_top, strip_id, upload_folder = args
    piece = img.crop((x1, y_top, x2, y_top + FIXED_HEIGHT))
    is_filter, weight = get_matching_info(piece)
    if is_filter:
        return None
    pname = f'w{weight:03d}_{strip_id}_{i}.png'
    piece.save(os.path.join(upload_folder, pname))
    return pname

@app.route('/upload_strip', methods=['POST'])
def upload_strip():
    f = request.files.get('strip')
    y_top = int(request.form.get('y_top', 0))
    if not f: return jsonify({'error': 'No file'}), 400

    strip_id = uuid.uuid4().hex
    img = Image.open(f).convert('RGBA')
    upload_folder = app.config['UPLOAD_FOLDER']

    args_list = [
        (img, i, x1, x2, y_top, strip_id, upload_folder)
        for i, (x1, x2) in enumerate(BOXES_X)
    ]

    with ThreadPoolExecutor(max_workers=4) as executor:
        raw = list(executor.map(_process_piece, args_list))

    # 過濾掉被捨棄的格（None），並保留原始欄位順序
    results = [r for r in raw if r is not None]
    return jsonify({'pieces': results})

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    cover_name = data.get('cover')
    cells = data.get('cells', [])
    rows = int(data.get('grid_rows', 1))
    cols = int(data.get('grid_cols', 5))

    output_width = 3000
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
    print("伺服器已啟動，請確認資料夾路徑是否有中文...")
    app.run(debug=True, port=5000)
