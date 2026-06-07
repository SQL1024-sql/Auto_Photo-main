import json, os, glob, time
from flask import Flask, request, jsonify, render_template

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ADMIN_CONFIG_FILE = os.path.join(BASE_DIR, 'admin_config.json')
ACCESS_LOG_FILE   = os.path.join(BASE_DIR, 'access.log')
GPS_LOG_FILE      = os.path.join(BASE_DIR, 'gps_log.jsonl')
UPLOAD_FOLDER     = os.path.join(BASE_DIR, 'uploads')

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'))

ROUTE_CATEGORIES = [
    ('GET',  '/',                 '首頁'),
    ('POST', '/upload_strip',     '上傳截圖'),
    ('POST', '/upload_cover',     '上傳封面'),
    ('POST', '/upload_anchor',    '上傳錨點'),
    ('POST', '/detect_y',         '偵測Y'),
    ('POST', '/generate',         '生成圖片'),
    ('POST', '/reload_templates', '重載模板'),
    ('GET',  '/sort_tags_list',   '標籤列表'),
    ('POST', '/sort_tag_rename',  '標籤改名'),
]

def categorize(method, path):
    for m, p, label in ROUTE_CATEGORIES:
        if method == m and (path == p or path.startswith(p + '/')):
            return label
    return '其他'

def load_config():
    try:
        with open(ADMIN_CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {'ip_mode': 'open', 'allowed_ips': []}

def save_config(cfg):
    with open(ADMIN_CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

@app.route('/')
def index():
    return render_template('admin.html')

@app.route('/api/config', methods=['GET'])
def api_get_config():
    return jsonify(load_config())

@app.route('/api/config', methods=['POST'])
def api_set_config():
    data = request.json or {}
    cfg = load_config()
    if 'ip_mode' in data:
        cfg['ip_mode'] = data['ip_mode']
    if 'allowed_ips' in data:
        cfg['allowed_ips'] = [ip.strip() for ip in data['allowed_ips'] if ip.strip()]
    save_config(cfg)
    return jsonify({'ok': True})

@app.route('/api/stats')
def api_stats():
    ip_stats = {}
    if os.path.exists(ACCESS_LOG_FILE):
        with open(ACCESS_LOG_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(' | ')
                if len(parts) != 3:
                    continue
                _, ip, method_path = parts
                mp = method_path.split(' ', 1)
                method = mp[0]
                path   = mp[1] if len(mp) > 1 else ''
                cat = categorize(method, path)
                if ip not in ip_stats:
                    ip_stats[ip] = {'__total__': 0}
                ip_stats[ip][cat] = ip_stats[ip].get(cat, 0) + 1
                ip_stats[ip]['__total__'] += 1
    sorted_stats = dict(sorted(ip_stats.items(), key=lambda x: -x[1]['__total__']))
    return jsonify({'stats': sorted_stats})

@app.route('/api/clear_log', methods=['POST'])
def api_clear_log():
    open(ACCESS_LOG_FILE, 'w', encoding='utf-8').close()
    return jsonify({'ok': True})

@app.route('/api/upload_info')
def api_upload_info():
    files = glob.glob(os.path.join(UPLOAD_FOLDER, '*'))
    total_size = sum(os.path.getsize(f) for f in files if os.path.isfile(f))
    return jsonify({'count': len(files), 'size_mb': round(total_size / 1024 / 1024, 2)})

@app.route('/api/gps_log')
def api_gps_log():
    records = []
    if os.path.exists(GPS_LOG_FILE):
        with open(GPS_LOG_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    records.append(json.loads(line.strip()))
                except Exception:
                    pass
    records.reverse()
    return jsonify({'records': records})

@app.route('/api/clear_gps', methods=['POST'])
def api_clear_gps():
    open(GPS_LOG_FILE, 'w', encoding='utf-8').close()
    return jsonify({'ok': True})

@app.route('/api/clean', methods=['POST'])
def api_clean():
    count, errors = 0, 0
    for f in glob.glob(os.path.join(UPLOAD_FOLDER, '*')):
        try:
            os.remove(f)
            count += 1
        except Exception:
            errors += 1
    return jsonify({'deleted': count, 'errors': errors})

if __name__ == '__main__':
    app.run(debug=False, port=8080)
