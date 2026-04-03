from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image
import os, uuid, io, base64, json, threading
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor

# ── CLIP (HuggingFace) ──────────────────────────────
try:
    from transformers import CLIPProcessor, CLIPModel
    import torch
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("[CLIP] transformers / torch not installed")

_clip_model = None
_clip_processor = None
_clip_lock = threading.Lock()
_clip_text_cache = {}

app = Flask(__name__)
LLM_RULES_FILE = os.path.join(os.path.dirname(__file__), 'llm_rules.json')
_llm_rules = []   # [{"text": str, "weight": int, "threshold": float}]

def _load_llm_rules():
    global _llm_rules
    if os.path.exists(LLM_RULES_FILE):
        try:
            with open(LLM_RULES_FILE, 'r', encoding='utf-8') as f:
                _llm_rules = json.load(f)
        except Exception:
            _llm_rules = []

def _save_llm_rules():
    with open(LLM_RULES_FILE, 'w', encoding='utf-8') as f:
        json.dump(_llm_rules, f, ensure_ascii=False, indent=2)

_load_llm_rules()

UPLOAD_FOLDER    = os.path.join(os.path.dirname(__file__), 'uploads')
TAGS_FOLDER      = os.path.join(os.path.dirname(__file__), 'filter_tags')
SORT_TAGS_FOLDER = os.path.join(os.path.dirname(__file__), 'sort_tags')
ANCHOR_FOLDER    = os.path.join(os.path.dirname(__file__), 'anchor')
SPECIAL_TAGS_FOLDER = os.path.join(os.path.dirname(__file__), 'special_tags')

for folder in [UPLOAD_FOLDER, TAGS_FOLDER, SORT_TAGS_FOLDER, ANCHOR_FOLDER, SPECIAL_TAGS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

app.config['UPLOAD_FOLDER']      = UPLOAD_FOLDER
app.config['TAGS_FOLDER']        = TAGS_FOLDER
app.config['SORT_TAGS_FOLDER']   = SORT_TAGS_FOLDER
app.config['ANCHOR_FOLDER']      = ANCHOR_FOLDER
app.config['SPECIAL_TAGS_FOLDER'] = SPECIAL_TAGS_FOLDER

BOXES_X      = [[661, 963], [988, 1290], [1315, 1618], [1644, 1946], [1973, 2274]]
FIXED_HEIGHT = 552

# ── ORB 參數 ─────────────────────────────────────────
# sort_tags 用 ORB 特徵點比對
# 命中條件：好的配對點數 >= ORB_MIN_MATCHES
# distance < ORB_DIST_THRESH 才算好的配對
ORB_MIN_MATCHES  = 8
ORB_DIST_THRESH  = 55
_orb = cv2.ORB_create(nfeatures=500)

# ── CLIP 工具函式 ────────────────────────────────────

def _get_clip():
    global _clip_model, _clip_processor
    if not CLIP_AVAILABLE:
        return None, None
    with _clip_lock:
        if _clip_model is None:
            print("[CLIP] loading openai/clip-vit-base-patch32 ...")
            _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            _clip_model.eval()
            print("[CLIP] ready")
    return _clip_model, _clip_processor

def _get_text_embedding(text):
    if text in _clip_text_cache:
        return _clip_text_cache[text]
    model, processor = _get_clip()
    if model is None:
        return None
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        vision_out = model.vision_model  # unused here, for text we use text_model
        emb = model.get_text_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    _clip_text_cache[text] = emb
    return emb

def _get_image_embedding(pil_img):
    model, processor = _get_clip()
    if model is None:
        return None
    inputs = processor(images=pil_img.convert("RGB"), return_tensors="pt")
    with torch.no_grad():
        vision_out = model.vision_model(pixel_values=inputs["pixel_values"])
        emb = model.visual_projection(vision_out.pooler_output)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb

def compute_clip_weight(image_piece):
    """CLIP text-based special rules."""
    if not _llm_rules or not CLIP_AVAILABLE:
        return 0
    img_emb = _get_image_embedding(image_piece)
    if img_emb is None:
        return 0
    max_weight = 0
    for rule in _llm_rules:
        text_emb = _get_text_embedding(rule["text"])
        if text_emb is None:
            continue
        sim = float((img_emb @ text_emb.T).item())
        if sim >= rule.get("threshold", 0.25):
            max_weight = max(max_weight, rule["weight"])
            print(f"[CLIP] hit '{rule['text']}' sim={sim:.3f} w={rule['weight']}")
    return max_weight

# ── ORB 比對 ─────────────────────────────────────────

def orb_match(roi_bgr, template_bgr):
    """
    ORB 特徵點比對。
    回傳 True 代表在 roi 中找到 template。
    對半透明、空心標籤都有效，因為比的是輪廓特徵而非像素顏色。
    """
    kp1, des1 = _orb.detectAndCompute(template_bgr, None)
    kp2, des2 = _orb.detectAndCompute(roi_bgr, None)

    # 特徵點太少代表這張 template 很單調（純色圖案），退回 template matching
    if des1 is None or des2 is None or len(kp1) < 5:
        if template_bgr.shape[0] <= roi_bgr.shape[0] and template_bgr.shape[1] <= roi_bgr.shape[1]:
            res = cv2.matchTemplate(roi_bgr, template_bgr, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            return max_val > 0.75
        return False

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    good = [m for m in matches if m.distance < ORB_DIST_THRESH]
    return len(good) >= ORB_MIN_MATCHES

# ── 原有工具函式 ─────────────────────────────────────

def cv_imread_unicode(file_path):
    try:
        return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"read failed ({file_path}): {e}")
        return None

def get_matching_info(image_piece):
    """
    回傳 (是否過濾, 權重)

    過濾層：filter_tags  → template matching (不變)
    排序層：sort_tags    → ORB 特徵點比對   (取代舊 template matching)
            special 規則 → CLIP 文字描述     (獨立，不衝突)
    最終權重 = max(orb_weight, clip_weight)
    """
    target_bgr = cv2.cvtColor(np.array(image_piece.convert('RGB')), cv2.COLOR_RGB2BGR)
    h, w = target_bgr.shape[:2]
    roi = target_bgr[0:130, int(w * 0.4):w]

    # 1. 過濾層：filter_tags — 維持 template matching，動也不動
    filter_files = [f for f in os.listdir(app.config['TAGS_FOLDER'])
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('.')]
    for tag_name in filter_files:
        template = cv_imread_unicode(os.path.join(app.config['TAGS_FOLDER'], tag_name))
        if template is None: continue
        if template.shape[0] > roi.shape[0] or template.shape[1] > roi.shape[1]: continue
        res = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val > 0.9:
            print(f"[filter] hit {tag_name} ({max_val:.2f})")
            return True, 0

    # 2. 排序層 A：sort_tags — ORB 特徵點比對
    orb_weight = 0
    sort_files = [f for f in os.listdir(app.config['SORT_TAGS_FOLDER'])
                  if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('.')]
    for sf in sort_files:
        template = cv_imread_unicode(os.path.join(app.config['SORT_TAGS_FOLDER'], sf))
        if template is None: continue
        if orb_match(roi, template):
            try:
                weight = int(sf.split('_')[-1].split('.')[0])
                if weight > orb_weight:
                    orb_weight = weight
                    print(f"[ORB] hit {sf} w={weight}")
            except Exception:
                pass

    # 3. 排序層 B：special 規則 — CLIP 文字描述（完全獨立，不跟 sort_tags 衝突）
    clip_weight = compute_clip_weight(image_piece)

    return False, max(orb_weight, clip_weight)

# ── Routes ───────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index-2622-llm.html')

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
    strip_h = strip_bgr.shape[0]
    half_y = strip_h // 2
    search_region = strip_bgr[half_y:, :]
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

@app.route('/llm_rules', methods=['GET'])
def get_llm_rules():
    return jsonify({'rules': _llm_rules, 'clip_available': CLIP_AVAILABLE})

@app.route('/llm_rules', methods=['POST'])
def set_llm_rules():
    global _llm_rules, _clip_text_cache
    data = request.json
    _llm_rules = data.get('rules', [])
    _clip_text_cache = {}
    _save_llm_rules()
    if _llm_rules and CLIP_AVAILABLE:
        threading.Thread(target=_get_clip, daemon=True).start()
    return jsonify({'ok': True, 'count': len(_llm_rules)})

@app.route('/orb_config', methods=['GET'])
def get_orb_config():
    return jsonify({'min_matches': ORB_MIN_MATCHES, 'dist_thresh': ORB_DIST_THRESH})

@app.route('/orb_config', methods=['POST'])
def set_orb_config():
    global ORB_MIN_MATCHES, ORB_DIST_THRESH
    data = request.json
    ORB_MIN_MATCHES = int(data.get('min_matches', ORB_MIN_MATCHES))
    ORB_DIST_THRESH = int(data.get('dist_thresh', ORB_DIST_THRESH))
    return jsonify({'ok': True, 'min_matches': ORB_MIN_MATCHES, 'dist_thresh': ORB_DIST_THRESH})

if __name__ == '__main__':
    print("LLM server starting on port 5001")
    app.run(debug=False, port=5001)
