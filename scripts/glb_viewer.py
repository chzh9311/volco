"""
Minimal web viewer for the .glb files exported by benchmark_contact_rep.visualize_comparison.

    python scripts/glb_viewer.py                      # serves tmp/contact_3dobj on http://127.0.0.1:8000
    python scripts/glb_viewer.py --root <dir> --port 8080 --host 0.0.0.0

<root> must contain one sub-folder per sample with .glb files inside. In the page: pick a sample,
tick the layers to overlay (all files share the same object frame); left/right arrow keys switch samples.
"""
import argparse
import json
import os
import os.path as osp
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

PAGE = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GLB Viewer</title>
<style>
  :root { --bg: #f4f4f2; --panel: #ffffff; --text: #222; --muted: #777; --line: #ddd; }
  html, body { margin: 0; height: 100%; background: var(--bg); color: var(--text);
               font: 13px/1.4 system-ui, sans-serif; }
  #app { display: flex; height: 100%; }
  #side { width: 260px; flex: none; overflow-y: auto; background: var(--panel);
          border-right: 1px solid var(--line); padding: 12px; box-sizing: border-box; }
  #view { flex: 1; position: relative; min-width: 0; }
  canvas { display: block; }
  select, button { width: 100%; margin-bottom: 8px; font: inherit; padding: 4px; }
  .group { margin-top: 10px; }
  .group h4 { margin: 0 0 4px; font-size: 12px; text-transform: uppercase; color: var(--muted); }
  label { display: block; padding: 2px 0; cursor: pointer; white-space: nowrap; }
  #status { position: absolute; left: 10px; bottom: 8px; color: var(--muted); }
  .row { display: flex; gap: 6px; } .row button { flex: 1; }
</style>
<script type="importmap">
{ "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/" } }
</script>
</head>
<body>
<div id="app">
  <div id="side">
    <select id="sample"></select>
    <div class="row"><button id="prev">&larr; prev</button><button id="next">next &rarr;</button></div>
    <div class="row"><button id="reset">Reset view</button><button id="none">Hide all</button></div>
    <div id="layers"></div>
  </div>
  <div id="view"><div id="status"></div></div>
</div>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const view = document.getElementById('view'), statusEl = document.getElementById('status');
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(window.devicePixelRatio);
view.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xffffff);
const camera = new THREE.PerspectiveCamera(35, 1, 0.001, 100);
camera.position.set(0.15, 0.12, 0.25);  // must differ from the orbit target, or OrbitControls has zero radius
scene.add(camera);
scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8a8a, 2.0));
const key = new THREE.DirectionalLight(0xffffff, 1.5);
key.position.set(1, 1, 2);
camera.add(key);  // headlight: follows the camera
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

function resize() {
  const w = view.clientWidth, h = view.clientHeight;
  renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize); resize();
(function loop() { requestAnimationFrame(loop); controls.update(); renderer.render(scene, camera); })();

// The exporter writes display (sRGB) colours, glTF assumes linear: convert so colours match Open3D/matplotlib.
function toLinear(root) {
  root.traverse(o => {
    if (!o.isMesh) return;
    o.material.color.convertSRGBToLinear();
    const c = o.geometry.getAttribute('color');
    if (c) {
      const tmp = new THREE.Color();
      for (let i = 0; i < c.count; i++) {
        tmp.setRGB(c.getX(i), c.getY(i), c.getZ(i)).convertSRGBToLinear();
        c.setXYZ(i, tmp.r, tmp.g, tmp.b);
      }
      c.needsUpdate = true;
    }
  });
}

const loader = new GLTFLoader();
const cache = new Map();   // url -> Promise<Object3D>
const shown = new Map();   // file name -> Object3D currently in the scene
let samples = {}, names = [], current = null, framed = false;
const checked = new Set(['volco_grasp.glb']);  // kept across samples

function load(url) {
  if (!cache.has(url)) cache.set(url, loader.loadAsync(url).then(g => { toLinear(g.scene); return g.scene; }));
  return cache.get(url);
}

function frame() {
  const box = new THREE.Box3();
  shown.forEach(o => box.expandByObject(o));
  if (box.isEmpty()) return;
  const c = box.getCenter(new THREE.Vector3()), r = box.getSize(new THREE.Vector3()).length() / 2;
  const d = r / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2));
  camera.position.copy(c).add(new THREE.Vector3(0.6, 0.5, 1).normalize().multiplyScalar(d));
  camera.near = d / 100; camera.far = d * 100; camera.updateProjectionMatrix();
  controls.target.copy(c); controls.update();
}

async function setLayer(file, on) {
  if (on) checked.add(file); else checked.delete(file);
  if (!on) { if (shown.has(file)) { scene.remove(shown.get(file)); shown.delete(file); } return; }
  const sample = current;
  statusEl.textContent = `loading ${file} ...`;
  const obj = await load(`${encodeURIComponent(sample)}/${encodeURIComponent(file)}`);
  statusEl.textContent = '';
  if (sample !== current || !checked.has(file) || shown.has(file)) return;  // changed meanwhile
  scene.add(obj); shown.set(file, obj);
  if (!framed) { frame(); framed = true; }
}

function showSample(name) {
  current = name;
  document.getElementById('sample').value = name;
  shown.forEach(o => scene.remove(o)); shown.clear();
  const groups = {};
  for (const f of samples[name]) (groups[f.split('_')[0].replace('.glb', '')] ??= []).push(f);
  const box = document.getElementById('layers'); box.innerHTML = '';
  for (const [g, files] of Object.entries(groups)) {
    const div = document.createElement('div'); div.className = 'group';
    div.innerHTML = `<h4>${g}</h4>`;
    for (const f of files) {
      const lab = document.createElement('label'), cb = document.createElement('input');
      cb.type = 'checkbox'; cb.checked = checked.has(f);
      cb.onchange = () => setLayer(f, cb.checked);
      lab.append(cb, ' ' + f.replace('.glb', '').replace(g + '_', ''));
      div.appendChild(lab);
    }
    box.appendChild(div);
  }
  [...checked].filter(f => samples[name].includes(f)).forEach(f => setLayer(f, true));
}

function step(k) { showSample(names[(names.indexOf(current) + k + names.length) % names.length]); }
document.getElementById('prev').onclick = () => step(-1);
document.getElementById('next').onclick = () => step(1);
document.getElementById('reset').onclick = frame;
document.getElementById('none').onclick = () => {
  [...checked].forEach(f => setLayer(f, false));
  document.querySelectorAll('#layers input').forEach(cb => cb.checked = false);
};
document.getElementById('sample').onchange = e => showSample(e.target.value);
window.addEventListener('keydown', e => {
  if (e.target.tagName === 'SELECT') return;
  if (e.key === 'ArrowLeft') step(-1); else if (e.key === 'ArrowRight') step(1);
});

samples = await (await fetch('api/samples')).json();
names = Object.keys(samples);
const sel = document.getElementById('sample');
for (const n of names) sel.add(new Option(n, n));
if (names.length) showSample(names[0]); else statusEl.textContent = 'no samples found';
</script>
</body>
</html>
"""


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split('?', 1)[0]
        if path in ('/', '/index.html'):
            return self._send(PAGE.encode(), 'text/html; charset=utf-8')
        if path == '/api/samples':
            root = self.directory
            samples = {d: sorted(f for f in os.listdir(osp.join(root, d)) if f.endswith('.glb'))
                       for d in sorted(os.listdir(root)) if osp.isdir(osp.join(root, d))}
            return self._send(json.dumps({k: v for k, v in samples.items() if v}).encode(), 'application/json')
        return super().do_GET()

    def _send(self, body, ctype):
        self.send_response(200)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', default=osp.join('tmp', 'contact_3dobj'))
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()

    root = osp.abspath(args.root)
    server = ThreadingHTTPServer((args.host, args.port), partial(Handler, directory=root))
    url = f'http://{args.host}:{args.port}/'
    print(f'Serving {root} at {url} (Ctrl+C to stop)')
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
