# -*- coding: utf-8 -*-
"""vlm_extract.py — VLM extraction + spline densification."""
import os, sys, json, base64, argparse
from pathlib import Path
import cv2, numpy as np

try:
    from scipy.interpolate import splprep, splev
except ImportError:
    sys.exit("scipy required: python -m pip install scipy")
try:
    import anthropic
except ImportError:
    sys.exit("anthropic required: python -m pip install anthropic")

DEFAULT_MODEL = "claude-opus-4-8"
SYSTEM_PROMPT = ("You are a careful chart-digitization assistant. You output "
    "ONLY a single JSON object: no markdown fences, no prose.")

def draw_grid(img, step=100):
    out = img.copy(); h, w = out.shape[:2]
    grid, label = (210,210,210), (90,90,90)
    f = cv2.FONT_HERSHEY_SIMPLEX
    for x in range(0, w, step):
        cv2.line(out,(x,0),(x,h),grid,1)
        cv2.putText(out,str(x),(x+2,14),f,0.4,label,1,cv2.LINE_AA)
    for y in range(0, h, step):
        cv2.line(out,(0,y),(w,y),grid,1)
        cv2.putText(out,str(y),(2,y-2),f,0.4,label,1,cv2.LINE_AA)
    return out

def encode_png(img):
    ok, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode("utf-8")

def distinct_colors(n):
    if n <= 0: return []
    hues = np.linspace(0,179,n,dtype=np.uint8)
    hsv = np.stack([hues, np.full_like(hues,220), np.full_like(hues,255)], 1)
    bgr = cv2.cvtColor(hsv.reshape(-1,1,3), cv2.COLOR_HSV2BGR).reshape(-1,3)
    return [tuple(int(c) for c in row) for row in bgr]

def draw_overlay(img, dense_lines, sparse_lines):
    out = img.copy()
    colors = distinct_colors(len(dense_lines))
    for idx, (dense, sparse) in enumerate(zip(dense_lines, sparse_lines)):
        c = colors[idx]
        if len(dense) >= 2:
            pts = np.array(dense, dtype=np.int32).reshape(-1,1,2)
            cv2.polylines(out, [pts], False, c, 2, cv2.LINE_AA)
        for x, y in sparse:
            cv2.drawMarker(out, (int(x),int(y)), c, cv2.MARKER_CROSS, 10, 2)
        if sparse:
            lx, ly = min(sparse, key=lambda p: p[0])
            cv2.putText(out, str(idx), (int(lx)-6, int(ly)-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2, cv2.LINE_AA)
    return out

def build_prompt(w, h, n):
    schema = ('{"summary":"...","lines":[{"label":"...",'
              '"points":[[x,y],[x,y]]}]}')
    return (f"Line chart {w}x{h}px with a gray pixel grid (spacing 100px, "
        f"labeled on top/left edges). Origin (0,0) is TOP-LEFT, y increases "
        f"DOWN. Identify every distinct data curve. For each, return ~{n} "
        f"integer [x,y] points ALONG the curve in order from one endpoint to "
        f"the other. Rules: points lie ON the curve; cover the WHOLE curve "
        f"including any steep vertical TAIL — do not stop early; ignore axis "
        f"labels/legends/titles/gridlines/annotations. Output ONLY this JSON, "
        f"no markdown: " + schema)

def parse_response(text):
    s = text.strip()
    if s.startswith("```"):
        parts = s.split("```")
        s = parts[1] if len(parts) >= 2 else text
        if s.lstrip().lower().startswith("json"):
            s = s.lstrip()[4:]
        s = s.strip()
    if not s.startswith("{"):
        i, j = s.find("{"), s.rfind("}")
        if i == -1 or j == -1:
            raise ValueError("No JSON in response: " + text[:300])
        s = s[i:j+1]
    return json.loads(s)

def _linear(pts, n):
    t = np.linspace(0,1,n); u = np.linspace(0,1,len(pts))
    x = np.interp(t,u,pts[:,0]); y = np.interp(t,u,pts[:,1])
    return [[int(round(a)), int(round(b))] for a,b in zip(x,y)]

def densify(points, n_dense=150, smoothing=None):
    pts = np.asarray(points, dtype=float)
    if len(pts) >= 2:
        keep = [0]
        for i in range(1, len(pts)):
            if not np.allclose(pts[i], pts[keep[-1]], atol=0.5):
                keep.append(i)
        pts = pts[keep]
    if len(pts) < 2:
        return [[int(round(p[0])), int(round(p[1]))] for p in pts]
    if len(pts) < 4:
        return _linear(pts, n_dense)
    s = smoothing if smoothing is not None else len(pts)
    try:
        tck, _ = splprep([pts[:,0], pts[:,1]], s=s, k=3)
        u = np.linspace(0,1,n_dense); x,y = splev(u, tck)
        return [[int(round(a)), int(round(b))] for a,b in zip(x,y)]
    except Exception as e:
        print(f"  spline failed ({e}); linear fallback")
        return _linear(pts, n_dense)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--points-per-curve", type=int, default=20)
    ap.add_argument("--dense", type=int, default=150)
    ap.add_argument("--smoothing", type=float, default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--no-ssl-verify", action="store_true")
    ap.add_argument("--no-proxy", action="store_true")
    ap.add_argument("--grid-step", type=int, default=100)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    p = Path(args.image).expanduser()
    if not p.exists(): sys.exit(f"image not found: {p}")
    img = cv2.imread(str(p))
    if img is None: sys.exit(f"could not read image: {p}")
    h, w = img.shape[:2]
    print(f"[1/4] loaded {p.name} ({w}x{h})")

    if args.no_proxy:
        for v in ("HTTP_PROXY","HTTPS_PROXY","http_proxy","https_proxy"):
            os.environ.pop(v, None)
        print("       proxy env vars stripped")

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key: sys.exit("ANTHROPIC_API_KEY not set")

    if args.no_ssl_verify:
        import httpx
        client = anthropic.Anthropic(api_key=key,
            http_client=httpx.Client(verify=False))
    else:
        client = anthropic.Anthropic(api_key=key)

    print(f"[2/4] asking {args.model} for ~{args.points_per_curve} pts/curve...")
    grid = draw_grid(img, step=args.grid_step)
    msg = client.messages.create(
        model=args.model, max_tokens=4096, system=SYSTEM_PROMPT,
        messages=[{"role":"user","content":[
            {"type":"image","source":{"type":"base64",
                "media_type":"image/png","data":encode_png(grid)}},
            {"type":"text","text":build_prompt(w,h,args.points_per_curve)},
        ]}])
    raw = "".join(b.text for b in msg.content if b.type == "text")
    parsed = parse_response(raw)
    print(f"       Claude: {parsed.get('summary','')}")

    sparse, labels = [], []
    for line in parsed.get("lines", []):
        pts = [[int(np.clip(p[0],0,w-1)), int(np.clip(p[1],0,h-1))]
               for p in (line.get("points") or []) if len(p) >= 2]
        if pts:
            sparse.append(pts); labels.append(line.get("label",""))
    if not sparse:
        sys.exit("Claude returned no lines. Raw:\n" + raw[:1000])

    print(f"[3/4] densifying {len(sparse)} curves -> {args.dense} pts each...")
    dense = [densify(l, n_dense=args.dense, smoothing=args.smoothing)
             for l in sparse]

    out_dir = Path(args.out).expanduser() if args.out else p.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    jpath = out_dir / f"{p.stem}_vlm_extract.json"
    ipath = out_dir / f"{p.stem}_vlm_overlay.png"
    payload = {"image":str(p),"width":w,"height":h,"model":args.model,
        "summary":parsed.get("summary",""),
        "lines":[{"label":labels[i],"sparse":sparse[i],"dense":dense[i]}
                 for i in range(len(sparse))]}
    with open(jpath,"w") as f: json.dump(payload, f, indent=2)
    cv2.imwrite(str(ipath), draw_overlay(img, dense, sparse))
    print(f"[4/4] wrote:\n       {jpath}\n       {ipath}")
    for i,(s,d,l) in enumerate(zip(sparse, dense, labels)):
        tag = f" — {l}" if l else ""
        print(f"       line {i}: {len(s)} sparse -> {len(d)} dense{tag}")

if __name__ == "__main__":
    main()
