from flask import Flask, request, jsonify, render_template, send_file
from PIL import Image
import os, uuid, io, base64
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_cover', methods=['POST'])
def upload_cover():
    f = request.files.get('cover')
    if not f:
        return jsonify({'error': 'No file'}), 400
    ext = os.path.splitext(f.filename)[1]
    fname = f'cover_{uuid.uuid4().hex}{ext}'
    f.save(os.path.join(app.config['UPLOAD_FOLDER'], fname))
    return jsonify({'filename': fname})


def find_content_bounds(arr, threshold=40):
    """Find bounding box of non-dark content pixels."""
    h, w = arr.shape[:2]
    col_mean = arr.mean(axis=0).mean(axis=1)
    row_mean = arr.mean(axis=1).mean(axis=1)
    left   = next((i for i in range(w)     if col_mean[i] > threshold), 0)
    right  = next((i for i in range(w-1,0,-1) if col_mean[i] > threshold), w-1)
    top    = next((i for i in range(h)     if row_mean[i] > threshold), 0)
    bottom = next((i for i in range(h-1,0,-1) if row_mean[i] > threshold), h-1)
    return left, top, right, bottom


def cut_grid(img, rows, cols, start_x=0):
    """
    Cut image into rows×cols equal cells.
    start_x: skip left nav bar.
    Finds content boundary automatically, then equal-splits.
    Returns list of (x1,y1,x2,y2).
    """
    arr = np.array(img)
    h, w = arr.shape[:2]

    # Find content boundary starting from start_x
    arr_right = arr[:, start_x:, :]
    _, top, right_rel, bottom = find_content_bounds(arr_right)
    left  = start_x  # user already specified where cards start
    right = start_x + right_rel

    content_w = right - left
    content_h = bottom - top
    cell_w = content_w // cols
    cell_h = content_h // rows

    cards = []
    for r in range(rows):
        for c in range(cols):
            x1 = left  + c * cell_w
            y1 = top   + r * cell_h
            x2 = x1 + cell_w
            y2 = y1 + cell_h
            cards.append((x1, y1, x2, y2))
    return cards


@app.route('/upload_strip', methods=['POST'])
def upload_strip():
    import json as _json
    files    = request.files.getlist('strips')
    # segments: [{y1, y2, cols, startX}]
    segments_raw = request.form.get('segments')
    if segments_raw:
        segments = _json.loads(segments_raw)
    else:
        # legacy fallback
        cols    = int(request.form.get('count', 5))
        rows    = int(request.form.get('rows', 1))
        start_x = int(request.form.get('start_x', 0))
        segments = None

    results = []
    for f in files:
        ext      = os.path.splitext(f.filename)[1]
        strip_id = uuid.uuid4().hex
        fname    = f'strip_{strip_id}{ext}'
        path     = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        f.save(path)
        img = Image.open(path)
        iw, ih = img.size
        arr = np.array(img)

        if segments:
            cards = []
            for seg in segments:
                y1      = int(seg['y1'])
                y2      = min(int(seg['y2']), ih)
                cols    = int(seg['cols'])
                start_x = int(seg.get('startX', 0))
                # find right boundary (first dark col from right)
                arr_row = arr[y1:y2, start_x:, :]
                col_mean = arr_row.mean(axis=0).mean(axis=1)
                right_rel = next(
                    (iw - start_x - 1 - i
                     for i, v in enumerate(reversed(col_mean.tolist()))
                     if v > 30),
                    iw - start_x - 1
                )
                x_right = start_x + right_rel + 1
                cw = (x_right - start_x) // cols
                for c in range(cols):
                    x1c = start_x + c * cw
                    x2c = x1c + cw
                    cards.append((x1c, y1, x2c, y2))
        else:
            cards = cut_grid(img, rows=rows, cols=cols, start_x=start_x)

        for i, (x1, y1, x2, y2) in enumerate(cards):
            piece = img.crop((x1, y1, x2, y2))
            pname = f'piece_{strip_id}_{i}.png'
            piece.save(os.path.join(app.config['UPLOAD_FOLDER'], pname))
            results.append(pname)

    return jsonify({'pieces': results})


@app.route('/image/<filename>')
def serve_image(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))


@app.route('/generate', methods=['POST'])
def generate():
    data         = request.json
    cover_file   = data.get('cover')
    cells        = data.get('cells', [])
    grid_rows    = int(data.get('grid_rows', 1))
    grid_cols    = int(data.get('grid_cols', len(cells)))
    output_width = 1200
    images_to_compose = []

    if cover_file:
        cover_path = os.path.join(app.config['UPLOAD_FOLDER'], cover_file)
        cover_img  = Image.open(cover_path).convert('RGBA')
        ratio      = output_width / cover_img.width
        cover_img  = cover_img.resize((output_width, int(cover_img.height * ratio)), Image.LANCZOS)
        images_to_compose.append(cover_img)

    if cells and grid_cols > 0:
        cell_w = output_width // grid_cols
        sample_fname = next((c for c in cells if c), None)
        if sample_fname:
            sample = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], sample_fname))
            cell_h = int(sample.height * cell_w / sample.width)
        else:
            cell_h = cell_w

        grid_img = Image.new('RGBA', (output_width, cell_h * grid_rows), (255, 255, 255, 255))
        for idx, fname in enumerate(cells):
            row = idx // grid_cols
            col = idx  % grid_cols
            if row >= grid_rows or not fname:
                continue
            piece = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], fname)).convert('RGBA')
            piece = piece.resize((cell_w, cell_h), Image.LANCZOS)
            grid_img.paste(piece, (col * cell_w, row * cell_h))
        images_to_compose.append(grid_img)

    if not images_to_compose:
        return jsonify({'error': 'Nothing to compose'}), 400

    total_h = sum(i.height for i in images_to_compose)
    final   = Image.new('RGBA', (output_width, total_h), (255, 255, 255, 255))
    y = 0
    for img in images_to_compose:
        final.paste(img, (0, y)); y += img.height

    out_name = f'result_{uuid.uuid4().hex}.jpg'
    final.convert('RGB').save(os.path.join(app.config['OUTPUT_FOLDER'], out_name), 'JPEG', quality=95)

    buf = io.BytesIO()
    preview = final.copy(); preview.thumbnail((800, 99999))
    preview.convert('RGB').save(buf, 'JPEG', quality=85)

    return jsonify({'preview': base64.b64encode(buf.getvalue()).decode(), 'filename': out_name})


@app.route('/download/<filename>')
def download(filename):
    return send_file(os.path.join(app.config['OUTPUT_FOLDER'], filename),
                     as_attachment=True, download_name=filename)


if __name__ == '__main__':
    app.run(debug=True, port=5000)