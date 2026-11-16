// In-browser demo: decode the photo, preprocess exactly like Python, run the ONNX model with onnxruntime-web.
import { IMAGE_SIZE, diagnose, preprocess } from "./preprocess.js";

const ORT_VERSION = "1.20.1"; // keep in sync with the <script> tag in index.html
const MODEL_URL = "model/plant_fp16.onnx";
const $ = (id) => document.getElementById(id);

let session = null;
let meta = null;
let busy = false;

function setStatus(text, { error = false, progress = null } = {}) {
  const el = $("status");
  el.hidden = !text;
  el.classList.toggle("error", error);
  el.firstChild.textContent = text ?? "";
  $("progress").hidden = progress === null;
  if (progress !== null) $("progress").value = progress;
}

/** Download with a progress bar (8.5 MB matters on a phone connection). */
async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const total = Number(res.headers.get("content-length")) || 0;
  if (!res.body || !total) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress(received / total);
  }
  const out = new Uint8Array(received);
  let offset = 0;
  for (const c of chunks) { out.set(c, offset); offset += c.length; }
  return out;
}

async function loadModel() {
  ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  const [metaRes, bytes] = await Promise.all([
    fetch("model/labels.json").then((r) => r.json()),
    fetchWithProgress(MODEL_URL, (p) => setStatus("Loading the model (8.5 MB)…", { progress: p })),
  ]);
  meta = metaRes;
  setStatus("Starting the model…");
  session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
}

/** Decode any image the browser understands (EXIF rotation applied) into RGBA pixels. */
async function decode(blob) {
  const bitmap = await createImageBitmap(blob);
  const canvas = document.createElement("canvas");
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0);
  bitmap.close?.();
  return ctx.getImageData(0, 0, canvas.width, canvas.height);
}

function drawCrop(rgb) {
  const ctx = $("crop").getContext("2d");
  const img = ctx.createImageData(IMAGE_SIZE, IMAGE_SIZE);
  for (let i = 0, j = 0; i < rgb.length; i += 3, j += 4) {
    img.data[j] = rgb[i]; img.data[j + 1] = rgb[i + 1]; img.data[j + 2] = rgb[i + 2]; img.data[j + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

const pct = (p) => `${(100 * p).toFixed(p < 0.1 ? 1 : 0)}%`;

const ICONS = {
  sure: '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/></svg>',
  unsure: '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><path d="M12 17h.01"/></svg>',
};

/** Status pill in the bar above the panels: loading (orange), ready (green), failed (rust). */
function setState(text, colour) {
  $("hud-state").innerHTML = `<span class="dot dot-${colour}"></span>`;
  $("hud-state").append(text);
}

function render(d, { ms, trueLabel }) {
  $("empty").hidden = true;
  $("output").hidden = false;

  $("verdict-kicker").textContent = d.sure ? "Diagnosis" : "Not sure";
  const verdict = $("verdict");
  verdict.className = `verdict ${d.sure ? "sure" : "unsure"}`;
  verdict.innerHTML = d.sure ? ICONS.sure : ICONS.unsure;
  verdict.append(d.sure ? d.answer : `Maybe ${d.bestGuess}`);

  const truth = $("truth");
  truth.className = trueLabel ? "truth" : "";
  truth.replaceChildren();
  if (trueLabel) {
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = !d.sure ? "Abstained" : d.answer === trueLabel ? "Matches label" : "Differs from label";
    truth.append(tag, `Labelled: ${trueLabel}`);
  }

  document.querySelector(".advice")?.remove();
  if (!d.sure) {
    const tip = document.createElement("p");
    tip.className = "advice";
    tip.innerHTML = "<strong>Closest match, not a diagnosis.</strong> ";
    tip.append("Its confidence is below the line where it is usually right. Try a closer, sharper photo of a " +
      "single leaf in daylight, or ask a local agriculture officer.");
    $("output").querySelector(".seen").after(tip);
  }

  $("fact-conf").textContent = pct(d.confidence);
  $("fact-threshold").textContent = pct(d.threshold);
  $("fact-ms").textContent = `${ms.toFixed(0)} ms`;
  $("hud-ms").textContent = `${ms.toFixed(0)} ms`;

  $("meter-fill").style.width = pct(d.confidence);
  $("meter-fill").classList.toggle("unsure", !d.sure);
  $("meter-mark").style.left = pct(d.threshold);

  $("bars").replaceChildren(...d.top.map(({ label, prob }) => {
    const li = document.createElement("li");
    li.innerHTML = `<span></span><span class="pct"></span><div class="track"><div class="fill"></div></div>`;
    li.children[0].textContent = label;
    li.children[1].textContent = pct(prob);
    li.querySelector(".fill").style.width = pct(prob);
    return li;
  }));

  $("details").textContent =
    `MobileNetV3 (FP16 ONNX, 8.5 MB) ran on your device in ${ms.toFixed(0)} ms. The photo was resized so its shorter ` +
    `side is 255 px and center-cropped to 224×224 (shown above). Scores were divided by a temperature of ` +
    `${meta.temperature.toFixed(2)} so that confidence matches real accuracy on held-out field photos, and anything ` +
    `below ${pct(d.threshold)} confidence is reported as “not sure”.`;
}

async function run(blob, trueLabel = null) {
  if (!session || busy) return;
  busy = true;
  setStatus("Looking at the leaf…");
  try {
    const { data, width, height } = await decode(blob);
    const t0 = performance.now();
    const { tensor, crop } = preprocess(data, width, height, 4);
    const out = await session.run({ image: new ort.Tensor("float32", tensor, [1, 3, IMAGE_SIZE, IMAGE_SIZE]) });
    const ms = performance.now() - t0;
    drawCrop(crop);
    const d = diagnose(out.logits.data, meta);
    render(d, { ms, trueLabel });
    setStatus(null);
    document.body.dataset.lastAnswer = d.answer; // used by the browser smoke test
    if (matchMedia("(max-width: 1023px)").matches) $("result-title").scrollIntoView({ behavior: "smooth" });
  } catch (err) {
    console.error(err);
    setStatus("Couldn't read that image. Try a JPG or PNG photo.", { error: true });
  } finally {
    busy = false;
  }
}

async function loadSamples() {
  const samples = await fetch("samples/samples.json").then((r) => r.json()).catch(() => []);
  const grid = $("samples");
  for (const s of samples) {
    const b = document.createElement("button");
    b.type = "button";
    b.setAttribute("role", "listitem");
    b.setAttribute("aria-pressed", "false");
    b.title = s.label ?? s.name;
    b.innerHTML = `<img loading="lazy" alt="">`;
    b.firstChild.src = s.src;
    b.firstChild.alt = s.label ? `Field photo: ${s.label}` : s.name;
    b.addEventListener("click", async () => {
      grid.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      await run(await fetch(s.src).then((r) => r.blob()), s.label);
    });
    grid.append(b);
  }
}

function wireInputs() {
  const drop = $("drop");
  $("file").addEventListener("change", (e) => e.target.files[0] && run(e.target.files[0]));
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    drop.classList.remove("over");
    const f = [...e.dataTransfer.files].find((x) => x.type.startsWith("image/"));
    if (f) run(f);
  });
  window.addEventListener("paste", (e) => {
    const item = [...e.clipboardData.items].find((x) => x.type.startsWith("image/"));
    if (item) run(item.getAsFile());
  });
}

wireInputs();
loadSamples();
try {
  await loadModel();
  setStatus(null);
  setState("Ready", "green");
  $("empty").hidden = false;
  document.body.dataset.ready = "true";
} catch (err) {
  console.error(err);
  setStatus("The model couldn't load. Check your connection and reload the page.", { error: true });
  setState("Failed", "rust");
}
