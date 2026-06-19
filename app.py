import time

from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image
import os, uuid, io, base64, json
import numpy as np
import cv2
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

app = Flask(__name__)
ACCESS_LOG_FILE   = os.path.join(os.path.dirname(__file__), 'access.log')
ADMIN_CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'admin_config.json')

def _load_admin_config():
    try:
        with open(ADMIN_CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {'ip_mode': 'open', 'allowed_ips': []}

@app.before_request
def before_each_request():
    ip = (request.headers.get('CF-Connecting-IP')
          or request.headers.get('X-Forwarded-For', request.remote_addr))
    if ip:
        ip = ip.split(',')[0].strip()
    ntime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log_line = f"{ntime} | {ip} | {request.method} {request.path}\n"
    print(log_line, end='')
    with open(ACCESS_LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(log_line)
    cfg = _load_admin_config()
    if cfg.get('ip_mode') == 'restricted':
        if ip not in cfg.get('allowed_ips', []):
            return jsonify({'error': '存取被拒絕'}), 403

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
TAGS_FOLDER = os.path.join(os.path.dirname(__file__), 'filter_tags')
SORT_TAGS_FOLDER = os.path.join(os.path.dirname(__file__), 'sort_tags')
ANCHOR_FOLDER = os.path.join(os.path.dirname(__file__), 'anchor')
SPECIAL_TAGS_FOLDER = os.path.join(os.path.dirname(__file__), 'special_tags')

for folder in [UPLOAD_FOLDER, TAGS_FOLDER, SORT_TAGS_FOLDER, ANCHOR_FOLDER, SPECIAL_TAGS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TAGS_FOLDER'] = TAGS_FOLDER
app.config['SORT_TAGS_FOLDER'] = SORT_TAGS_FOLDER
app.config['ANCHOR_FOLDER'] = ANCHOR_FOLDER
app.config['SPECIAL_TAGS_FOLDER'] = SPECIAL_TAGS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 80 * 1024 * 1024

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
SORT_TAGS_NAMES_FILE = os.path.join(os.path.dirname(__file__), 'sort_tags_names.json')

def _load_tag_names():
    try:
        with open(SORT_TAGS_NAMES_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}

def _save_tag_names(names):
    with open(SORT_TAGS_NAMES_FILE, 'w', encoding='utf-8') as f:
        json.dump(names, f, ensure_ascii=False, indent=2)

WIDTH_CONFIGS = {
    1600: {
        'BOXES_X': [[410, 593], [605, 789], [801, 984], [997, 1180], [1193, 1376]],
        'FIXED_HEIGHT': 334,
    },
    1882: {
        'BOXES_X': [[470, 692], [706, 928], [943, 1164], [1179, 1400], [1415, 1637]],
        'FIXED_HEIGHT': 406,
    },
    2048: {
        'BOXES_X': [[511, 752], [768, 1010], [1026, 1267], [1283, 1524], [1540, 1782]],
        'FIXED_HEIGHT': 440,
    },
    2532: {
        'BOXES_X': [[632, 930], [950, 1249], [1268, 1567], [1586, 1885], [1904, 2203]],
        'FIXED_HEIGHT': 548,
    },
    2556: {
        'BOXES_X': [[640, 939], [960, 1259], [1281, 1580], [1601, 1900], [1922, 2221]],
        'FIXED_HEIGHT': 551,
    },
    2622: {
        'BOXES_X': [[657, 965], [985, 1294], [1313, 1621], [1641, 1949], [1969, 2278]],
        'FIXED_HEIGHT': 557,
    },
}

def get_config(width_val):
    try:
        w = int(width_val)
    except (TypeError, ValueError):
        w = 2556
    return WIDTH_CONFIGS.get(w, WIDTH_CONFIGS[2556])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.errorhandler(413)
def request_entity_too_large(e):
    return jsonify({'error': '檔案過大，單次上傳上限為 80MB'}), 413

def cv_imread_unicode(file_path):
    try:
        return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"讀取圖檔失敗 ({file_path}): {e}")
        return None

_template_cache = {'filter': [], 'weight': []}
TEMPLATE_REF_WIDTH = 2556          # 模板擷取時的螢幕寬度（標籤以此尺度製作）
_scaled_cache = {}                 # width -> (filter_list, weight_list)

def _load_templates_from_folder(folder, is_filter=False):
    results = []
    for fname in os.listdir(folder):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')) or fname.startswith('.'):
            continue
        img = cv_imread_unicode(os.path.join(folder, fname))
        if img is None:
            continue
        weight = 0
        if not is_filter:
            try:
                weight = int(fname.split('_')[-1].split('.')[0])
            except Exception:
                pass
        results.append((fname, img, weight))
    return results

def reload_templates():
    _template_cache['filter'] = _load_templates_from_folder(TAGS_FOLDER, is_filter=True)
    _template_cache['weight'] = (
        _load_templates_from_folder(SORT_TAGS_FOLDER) +
        _load_templates_from_folder(SPECIAL_TAGS_FOLDER)
    )
    _scaled_cache.clear()
    print(f"[模板快取] filter={len(_template_cache['filter'])}, weight={len(_template_cache['weight'])}")

reload_templates()

def _worker_init():
    reload_templates()

if multiprocessing.current_process().name == 'MainProcess':
    _executor = ProcessPoolExecutor(
        max_workers=os.cpu_count() or 4,
        initializer=_worker_init
    )
else:
    _executor = None

def _scaled_templates(width):
    """依上傳寬度相對於 TEMPLATE_REF_WIDTH 的比例縮放模板。
    cv2.matchTemplate 不具縮放不變性，模板在 2556 擷取，
    上傳其他寬度（如 1882）時標籤較小，需同步縮放模板才能比中。"""
    try:
        w = int(width)
    except (TypeError, ValueError):
        w = TEMPLATE_REF_WIDTH
    if w == TEMPLATE_REF_WIDTH:
        return _template_cache['filter'], _template_cache['weight']
    cached = _scaled_cache.get(w)
    if cached is not None:
        return cached
    scale = w / float(TEMPLATE_REF_WIDTH)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    def _do(lst):
        out = []
        for fname, img, weight in lst:
            simg = cv2.resize(img, None, fx=scale, fy=scale, interpolation=interp)
            out.append((fname, simg, weight))
        return out
    cached = (_do(_template_cache['filter']), _do(_template_cache['weight']))
    _scaled_cache[w] = cached
    return cached

def get_matching_info(roi, width=TEMPLATE_REF_WIDTH):
    filter_templates, weight_templates = _scaled_templates(width)
    for tag_name, template, _ in filter_templates:
        if template.shape[0] > roi.shape[0] or template.shape[1] > roi.shape[1]:
            continue
        _, max_val, _, _ = cv2.minMaxLoc(cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED))
        if max_val > 0.9:
            current_local_time = time.localtime()
            ntime = time.strftime("%Y-%m-%d %H:%M:%S", current_local_time)
            print(f"{ntime} filter_tag: {tag_name} ({max_val:.2f})")
            return True, 0

    max_weight = 0
    for _, template, weight in weight_templates:
        if template.shape[0] > roi.shape[0] or template.shape[1] > roi.shape[1]:
            continue
        _, max_val, _, _ = cv2.minMaxLoc(cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED))
        if max_val > 0.8:
            max_weight = max(max_weight, weight)

    return False, max_weight

@app.route('/reload_templates', methods=['POST'])
def reload_templates_route():
    reload_templates()
    return jsonify({'filter': len(_template_cache['filter']), 'weight': len(_template_cache['weight'])})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/image/<filename>')
def serve_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/sort_tags_list', methods=['GET'])
def sort_tags_list():
    names = _load_tag_names()
    tags = []
    for fname, _, weight in _template_cache['weight']:
        if not os.path.exists(os.path.join(SORT_TAGS_FOLDER, fname)):
            continue
        default_name = fname.rsplit('_', 1)[0]
        tags.append({'filename': fname, 'name': names.get(fname, default_name), 'weight': weight})
    tags.sort(key=lambda x: (-x['weight'], x['name']))
    return jsonify({'tags': tags})

@app.route('/sort_tag_rename', methods=['POST'])
def sort_tag_rename():
    data = request.json or {}
    filename = os.path.basename(data.get('filename', '').strip())
    new_name = data.get('name', '').strip()
    if not filename or not new_name:
        return jsonify({'error': 'Missing fields'}), 400
    if not os.path.exists(os.path.join(SORT_TAGS_FOLDER, filename)):
        return jsonify({'error': 'File not found'}), 404
    names = _load_tag_names()
    names[filename] = new_name
    _save_tag_names(names)
    return jsonify({'ok': True})

@app.route('/upload_anchor', methods=['POST'])
def upload_anchor():
    f = request.files.get('anchor')
    if not f: return jsonify({'error': 'No file'}), 400
    if not allowed_file(f.filename): return jsonify({'error': '僅支援 PNG / JPG / JPEG / WEBP'}), 400
    width_val = int(request.form.get('width', 2556))
    anchor_path = os.path.join(app.config['ANCHOR_FOLDER'], f'anchor-{width_val}.png')
    Image.open(f).convert('RGB').save(anchor_path)
    return jsonify({'ok': True})

@app.route('/detect_y', methods=['POST'])
def detect_y():
    f = request.files.get('strip')
    y_offset = int(request.form.get('y_offset', 0))
    cfg = get_config(request.form.get('width'))
    FIXED_HEIGHT = cfg['FIXED_HEIGHT']
    width_val = int(request.form.get('width', 2556))

    if not f:
        return jsonify({'error': 'No file'}), 400

    anchor_path = os.path.join(app.config['ANCHOR_FOLDER'], f'anchor-{width_val}.png')
    if not os.path.exists(anchor_path):
        print(f"[detect_y] 400: anchor-{width_val}.png not found at", anchor_path)
        return jsonify({'error': 'No anchor uploaded'}), 400

    img_data = np.frombuffer(f.read(), dtype=np.uint8)
    strip_bgr = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    if strip_bgr is None:
        return jsonify({'error': 'Cannot decode strip image'}), 400

    template = cv_imread_unicode(anchor_path)
    if template is None:
        return jsonify({'error': 'Cannot read anchor'}), 400

    strip_h = strip_bgr.shape[0]
    search_start = strip_h // 3
    search_region = strip_bgr[search_start:, :]
    current_local_time = time.localtime()
    ntime = time.strftime("%Y-%m-%d %H:%M:%S", current_local_time)
    print(f"{ntime} [detect_y] width={width_val} strip={strip_bgr.shape}, search_start={search_start}, template={template.shape}")

    if template.shape[0] > search_region.shape[0] or template.shape[1] > search_region.shape[1]:
        return jsonify({'error': 'Anchor image is larger than search region'}), 400

    res = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)

    if max_val < 0.7:
        return jsonify({'found': False, 'score': round(float(max_val), 3)})

    anchor_y = max_loc[1] + search_start
    y_top = int(np.clip(anchor_y + y_offset, 0, strip_h - FIXED_HEIGHT))
    return jsonify({'found': True, 'y_top': y_top, 'score': round(float(max_val), 3)})

@app.route('/upload_cover', methods=['POST'])
def upload_cover():
    f = request.files.get('cover')
    if not f: return jsonify({'error': 'No file'}), 400
    if not allowed_file(f.filename): return jsonify({'error': '僅支援 PNG / JPG / JPEG / WEBP'}), 400
    fname = f'cover_{uuid.uuid4().hex}.png'
    f.save(os.path.join(app.config['UPLOAD_FOLDER'], fname))
    return jsonify({'filename': fname})

def _process_piece(args):
    raw_bytes, i, x1, x2, y_top, strip_id, upload_folder, fixed_height, width = args
    import io as _io
    img = Image.open(_io.BytesIO(raw_bytes)).convert('RGBA')
    piece = img.crop((x1, y_top, x2, y_top + fixed_height))
    bgr = cv2.cvtColor(np.array(piece.convert('RGB')), cv2.COLOR_RGB2BGR)
    w = bgr.shape[1]
    roi = bgr[0:130, int(w * 0.4):w]
    is_filter, weight = get_matching_info(roi, width)
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
    if not allowed_file(f.filename): return jsonify({'error': '僅支援 PNG / JPG / JPEG / WEBP'}), 400

    cfg = get_config(request.form.get('width'))
    BOXES_X = cfg['BOXES_X']
    FIXED_HEIGHT = cfg['FIXED_HEIGHT']
    width_val = request.form.get('width', TEMPLATE_REF_WIDTH)

    strip_id = uuid.uuid4().hex
    raw_bytes = f.read()
    upload_folder = app.config['UPLOAD_FOLDER']

    args_list = [
        (raw_bytes, i, x1, x2, y_top, strip_id, upload_folder, FIXED_HEIGHT, width_val)
        for i, (x1, x2) in enumerate(BOXES_X)
    ]

    raw = list(_executor.map(_process_piece, args_list))
    results = [r for r in raw if r is not None]
    return jsonify({'pieces': results})

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    cover_name = data.get('cover')
    cells = data.get('cells', [])
    rows = int(data.get('grid_rows', 1))
    cols = int(data.get('grid_cols', 5))

    cfg = get_config(data.get('width'))
    BOXES_X = cfg['BOXES_X']
    FIXED_HEIGHT = cfg['FIXED_HEIGHT']

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
            p = Image.open(piece_path).convert('RGB')
            p = p.resize((cell_w, cell_h), Image.LANCZOS)
            grid_img.paste(p, (c * cell_w, r * cell_h))

    images_to_combine.append(grid_img)

    total_h = sum(img.height for img in images_to_combine)
    final = Image.new('RGBA', (output_width, total_h), (26, 26, 26, 255))
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
    app.run(host='0.0.0.0', port=2000, debug=False)
