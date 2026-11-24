// Rendus produit d'un STL : figurine en resine grise, eclairage studio, 4 vues.
// Usage : node outils/rendu/rendu.mjs <modele.stl> <dossier de sortie> [taille_px]
// Le STL est en Z vers le haut, face vers -Y (sortie de kernel/main.py). Pour un rendu rapide,
// passer une version allegee (~300 k triangles) : la forme visible est la meme.
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { chromium } = require('playwright');
const here = path.dirname(fileURLToPath(import.meta.url));
const [stlPath, outDir, sizeArg] = process.argv.slice(2);
if (!stlPath || !outDir) {
  console.error('Usage : node rendu.mjs <modele.stl> <dossier> [taille_px]');
  process.exit(2);
}
const size = Number(sizeArg || 1200);
fs.mkdirSync(outDir, { recursive: true });

const page = `<!doctype html><html><head><meta charset="utf-8">
<style>html,body{margin:0;background:#fff}canvas{display:block}</style>
<script type="importmap">{"imports":{"three":"/node_modules/three/build/three.module.js",
"three/addons/":"/node_modules/three/examples/jsm/"}}</script></head><body>
<script type="module">
import * as THREE from 'three';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
const S = ${size};
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setSize(S, S);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMappingExposure = 0.85;
document.body.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.environment = new THREE.PMREMGenerator(renderer).fromScene(new RoomEnvironment(), 0.04).texture;
scene.environmentIntensity = 0.35;
// Fond degrade studio
const bg = document.createElement('canvas'); bg.width = 2; bg.height = 256;
const g = bg.getContext('2d'); const grad = g.createLinearGradient(0, 0, 0, 256);
grad.addColorStop(0, '#f4f4f2'); grad.addColorStop(1, '#d9d9d6'); g.fillStyle = grad; g.fillRect(0, 0, 2, 256);
scene.background = new THREE.CanvasTexture(bg); scene.background.colorSpace = THREE.SRGBColorSpace;

const geometry = new STLLoader().parse(await (await fetch('/model.stl')).arrayBuffer());
geometry.rotateX(-Math.PI / 2);                 // Z vers le haut -> Y vers le haut (three.js)
geometry.computeVertexNormals();
geometry.computeBoundingBox();
const box = geometry.boundingBox;
geometry.translate(-(box.min.x + box.max.x) / 2, -box.min.y, -(box.min.z + box.max.z) / 2);
geometry.computeBoundingBox();
const h = geometry.boundingBox.max.y, w = Math.max(geometry.boundingBox.max.x * 2, geometry.boundingBox.max.z * 2);
const mesh = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ color: 0x8c8c88, roughness: 0.55, metalness: 0.0 }));
mesh.castShadow = true;                         // ombre au sol seulement (pas d'acne sur la figurine)
scene.add(mesh);
const ground = new THREE.Mesh(new THREE.PlaneGeometry(h * 20, h * 20), new THREE.ShadowMaterial({ opacity: 0.18 }));
ground.rotation.x = -Math.PI / 2; ground.receiveShadow = true; scene.add(ground);

const key = new THREE.DirectionalLight(0xffffff, 1.6);
key.position.set(h * 1.2, h * 2.2, h * 1.6); key.castShadow = true;
key.shadow.mapSize.set(2048, 2048);
Object.assign(key.shadow.camera, { left: -h * 1.5, right: h * 1.5, top: h * 1.5, bottom: -h * 1.5, near: 0.1, far: h * 8 });
key.shadow.bias = -0.0005; scene.add(key);
const fill = new THREE.DirectionalLight(0xdde6ff, 0.6); fill.position.set(-h * 2, h, h); scene.add(fill);
const rim = new THREE.DirectionalLight(0xffffff, 0.9); rim.position.set(0, h * 1.5, -h * 2.5); scene.add(rim);

const camera = new THREE.PerspectiveCamera(30, 1, h / 100, h * 50);
const extent = Math.max(h, w);
window.renderView = (azimuthDeg, elevationDeg) => {
  const az = THREE.MathUtils.degToRad(azimuthDeg), el = THREE.MathUtils.degToRad(elevationDeg);
  const dist = extent * 0.5 / Math.tan(THREE.MathUtils.degToRad(15)) * 1.25;
  const target = new THREE.Vector3(0, h * 0.5, 0);
  camera.position.set(target.x + dist * Math.cos(el) * Math.sin(az), target.y + dist * Math.sin(el),
                      target.z + dist * Math.cos(el) * Math.cos(az));
  camera.lookAt(target);
  renderer.render(scene, camera);
  return true;
};
window.ready = true;
</script></body></html>`;

const server = http.createServer((req, res) => {
  const url = decodeURIComponent(req.url.split('?')[0]);
  if (url === '/') { res.writeHead(200, { 'content-type': 'text/html' }); return res.end(page); }
  if (url === '/model.stl') { res.writeHead(200); return fs.createReadStream(stlPath).pipe(res); }
  const file = path.join(here, url);
  if (!file.startsWith(path.join(here, 'node_modules')) || !fs.existsSync(file)) { res.writeHead(404); return res.end(); }
  res.writeHead(200, { 'content-type': 'text/javascript' });
  fs.createReadStream(file).pipe(res);
});
await new Promise((ok) => server.listen(0, '127.0.0.1', ok));
const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
try {
  const tab = await browser.newPage({ viewport: { width: size, height: size } });
  tab.on('pageerror', (e) => console.error('Erreur page :', e.message));
  await tab.goto(`http://127.0.0.1:${server.address().port}/`);
  await tab.waitForFunction('window.ready === true', null, { timeout: 300000 });
  const views = { 'rendu-face': [0, 8], 'rendu-3-4': [35, 15], 'rendu-profil': [90, 8], 'rendu-dos': [180, 8] };
  for (const [name, [az, el]] of Object.entries(views)) {
    await tab.evaluate(([a, e]) => window.renderView(a, e), [az, el]);
    await tab.locator('canvas').screenshot({ path: path.join(outDir, `${name}.png`) });
    console.log(path.join(outDir, `${name}.png`));
  }
} finally {
  await browser.close();
  server.close();
}
