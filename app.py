from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image
import os, uuid, io, base64, json
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)
UPLOAD_FOLDER       = os.path.join(os.path.dirname(__file__), 'uploads')
TAGS_FOLDER         = os.path.join(os.path.dirname(__file__), 'filter_tags')
SORT_TAGS_FOLDER    = os.path.join(os.path.dirname(__file__), 'sort_tags')
ANCHOR_FOLDER       = os.path.join(os.path.dirname(__file__), 'anchor')
SPECIAL_TAGS_FOLDER = os.path.join(os.path.dirname(__file__), 'special_tags')

for folder in [UPLOAD_FOLDER, TAGS_FOLDER, SORT_TAGS_FOLDER, ANCHOR_FOLDER, SPECIAL_TAGS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

app.config['UPLOAD_FOLDER']       = UPLOAD_FOLDER
app.config['TAGS_FOLDER']         = TAGS_FOLDER
app.config['SORT_TAGS_FOLDER']    = SORT_TAGS_FOLDER
app.config['ANCHOR_FOLDER']       = ANCHOR_FOLDER
app.config['SPECIAL_TAGS_FOLDER'] = SPECIAL_TAGS_FOLDER

BOXES_X      = [[661, 963], [988, 1290], [1315, 1618], [1644, 1946], [1973, 2274]]
FIXED_HEIGHT = 552

# ── 每個標籤的獨立閾值設定 ───────────────────────────
# sort_tags 預設閾值 0.80，可在 UI 對每張標籤單獨調整
# special_tags 因為要置頂、建議用 100~999 的權重值
SORT_TAGS_CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'sort_tags_config.json')
DEFAULT_SORT_THRESHOLD    = 0.80
DEFAULT_FILTER_THRESHOLD  = 0.90
_sort_thresholds = {}   # {filename: threshold}

def _load_sort_config():
    global _sort_thresholds
    if os.path.exists(SORT_TAGS_CONFIG_FILE):
        try:
            with open(SORT_TAGS_CONFIG_FILE, 'r', encoding='utf-8') as f:
                _sort_thresholds = json.load(f)
        except Exception:
            _sort_thresholds = {}

def _save_sort_config():
    with open(SORT_TAGS_CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(_sort_thresholds, f, ensure_ascii=False, indent=2)

_load_sort_config()

# ── Alpha-aware template matching ────────────────────
def match_template(roi_bgr, file_path):
    """
    Template matching 支援 PNG 透明遮罩。
    透明/半透明區域不參與比對，適用空心或半透明標籤。
    回傳 0.0~1.0 的相似度分數。
    """
    raw = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if raw is None:
        return 0.0

    if raw.ndim == 3 and raw.shape[2] == 4:
        mask     = raw[:, :, 3]          # alpha channel
        template = raw[:, :, :3]         # BGR
    else:
        mask     = None
        template = raw if raw.ndim == 3 else cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)

    if template.shape[0] > roi_bgr.shape[0] or template.shape[1] > roi_bgr.shape[1]:
        return 0.0

    try:
        if mask is not None:
            res = cv2.matchTemplate(roi_bgr, template, cv2.TM_CCOEFF_NORMED, mask=mask)
        else:
            res = cv2.matchTemplate(roi_bgr, template, cv2.TM_CCOEFF_NORMED)
    except cv2.error:
        # 舊版 OpenCV 不支援 mask，退回不帶 mask
        res = cv2.matchTemplate(roi_bgr, template, cv2.TM_CCOEFF_NORMED)

    _, max_val, _, _ = cv2.minMaxLoc(res)
    return float(max_val)

# ── 原有工具函式 ─────────────────────────────────────
def cv_imread_unicode(file_path):
    try:
        return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"讀取圖檔失敗 ({file_path}): {e}")
        return None

def get_matching_info(image_piece):
    """
    回傳 (是否過濾, 權重)

    filter_tags  → 固定閾值 0.90，命中就丟棄（不變）
    sort_tags    → 每個標籤各自設閾值，支援 alpha mask
    special_tags → 每個標籤各自設閾值，支援 alpha mask
                   建議用 100~999 的權重，不跟 sort_tags 衝突
    最終權重 = max(sort_weight, special_weight)
    """
    target_bgr = cv2.cvtColor(np.array(image_piece.convert('RGB')), cv2.COLOR_RGB2BGR)
    w = target_bgr.shape[1]
    roi = target_bgr[0:130, int(w * 0.4):w]

    # 1. filter_tags：命中就過濾，閾值固定 0.90
    filter_files = [f for f in os.listdir(app.config['TAGS_FOLDER'])
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('.')]
    for tag_name in filter_files:
        score = match_template(roi, os.path.join(app.config['TAGS_FOLDER'], tag_name))
        if score > DEFAULT_FILTER_THRESHOLD:
            print(f"[filter] 命中 {tag_name} ({score:.2f})")
            return True, 0

    # 2. sort_tags：每個標籤各自閾值，支援 alpha
    max_weight = 0
    sort_files = [f for f in os.listdir(app.config['SORT_TAGS_FOLDER'])
                  if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('.')]
    for sf in sort_files:
        score     = match_template(roi, os.path.join(app.config['SORT_TAGS_FOLDER'], sf))
        threshold = _sort_thresholds.get(sf, DEFAULT_SORT_THRESHOLD)
        if score > threshold:
            try:
                weight = int(sf.split('_')[-1].split('.')[0])
                if weight > max_weight:
                    max_weight = weight
                    print(f"[sort] 命中 {sf} score={score:.2f} threshold={threshold} w={weight}")
            except Exception:
                pass

    # 3. special_tags：建議權重 100~999，不跟 sort_tags 衝突
    special_files = [f for f in os.listdir(app.config['SPECIAL_TAGS_FOLDER'])
                     if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('.')]
    for sf in special_files:
        score     = match_template(roi, os.path.join(app.config['SPECIAL_TAGS_FOLDER'], sf))
        threshold = _sort_thresholds.get(sf, DEFAULT_SORT_THRESHOLD)
        if score > threshold:
            try:
                weight = int(sf.split('_')[-1].split('.')[0])
                if weight > max_weight:
                    max_weight = weight
                    print(f"[special] 命中 {sf} score={score:.2f} threshold={threshold} w={weight}")
            except Exception:
                pass

    return False, max_weight

# ── Routes ───────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index-2622.html')

@app.route('/image/<filename>')
def serve_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/upload_anchor', methods=['POST'])
def upload_anchor():
    f = request.files.get('anchor')
    if not f: return jsonify({'error': 'No file'}), 400
    anchor_path = os.path.join(app.config['ANCHOR_FOLDER'], 'anchor.png')
    Image.open(f).convert('RGB').save(anchor_path)
    return jsonify({'ok': True})

@app.route('/detect_y', methods=['POST'])
def detect_y():
    f = request.files.get('strip')
    y_offset = int(request.form.get('y_offset', 0))
    if not f:
        return jsonify({'error': 'No file'}), 400
    anchor_path = os.path.join(app.config['ANCHOR_FOLDER'], 'anchor.png')
    if not os.path.exists(anchor_path):
        return jsonify({'error': 'No anchor uploaded'}), 400
    img_data = np.frombuffer(f.read(), dtype=np.uint8)
    strip_bgr = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    if strip_bgr is None:
        return jsonify({'error': 'Cannot decode strip image'}), 400
    template = cv_imread_unicode(anchor_path)
    if template is None:
        return jsonify({'error': 'Cannot read anchor'}), 400
    strip_h    = strip_bgr.shape[0]
    half_y     = strip_h // 2
    search_region = strip_bgr[half_y:, :]
    print(f"[detect_y] strip={strip_bgr.shape}, template={template.shape}")
    if template.shape[0] > search_region.shape[0] or template.shape[1] > search_region.shape[1]:
        return jsonify({'error': 'Anchor image is larger than search region'}), 400
    res = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    if max_val < 0.7:
        return jsonify({'found': False, 'score': round(float(max_val), 3)})
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
    results = [r for r in raw if r is not None]
    return jsonify({'pieces': results})

@app.route('/generate', methods=['POST'])
def generate():
    data       = request.json
    cover_name = data.get('cover')
    cells      = data.get('cells', [])
    rows       = int(data.get('grid_rows', 1))
    cols       = int(data.get('grid_cols', 5))
    output_width = 3000
    cell_w = output_width // cols
    cell_h = int(cell_w * (FIXED_HEIGHT / (BOXES_X[0][1] - BOXES_X[0][0])))
    images_to_combine = []
    if cover_name:
        cover_path = os.path.join(app.config['UPLOAD_FOLDER'], cover_name)
        if os.path.exists(cover_path):
            cover_img = Image.open(cover_path).convert('RGBA')
            aspect    = cover_img.height / cover_img.width
            cover_img = cover_img.resize((output_width, int(output_width * aspect)), Image.LANCZOS)
            images_to_combine.append(cover_img)
    grid_h   = rows * cell_h
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
    total_h   = sum(img.height for img in images_to_combine)
    final     = Image.new('RGBA', (output_width, total_h))
    current_y = 0
    for img in images_to_combine:
        final.paste(img, (0, current_y), img)
        current_y += img.height
    out_io = io.BytesIO()
    final.convert('RGB').save(out_io, 'JPEG', quality=100, subsampling=0)
    b64 = base64.b64encode(out_io.getvalue()).decode()
    return jsonify({'preview': b64})

@app.route('/sort_tags_config', methods=['GET'])
def get_sort_tags_config():
    """回傳 sort_tags + special_tags 的標籤清單與各自閾值"""
    result = []
    for folder_key, folder_path in [('sort', app.config['SORT_TAGS_FOLDER']),
                                     ('special', app.config['SPECIAL_TAGS_FOLDER'])]:
        files = sorted(f for f in os.listdir(folder_path)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('.'))
        for f in files:
            try:
                weight = int(f.split('_')[-1].split('.')[0])
            except Exception:
                weight = 0
            result.append({
                'name':      f,
                'folder':    folder_key,
                'weight':    weight,
                'threshold': _sort_thresholds.get(f, DEFAULT_SORT_THRESHOLD)
            })
    return jsonify({'tags': result, 'default': DEFAULT_SORT_THRESHOLD})

@app.route('/sort_tags_config', methods=['POST'])
def set_sort_tags_config():
    global _sort_thresholds
    data = request.json
    _sort_thresholds = {item['name']: float(item['threshold']) for item in data.get('tags', [])}
    _save_sort_config()
    return jsonify({'ok': True})

if __name__ == '__main__':
    print("伺服器已啟動...")
    app.run(debug=False, port=5000)
