// Preprocessing and calibration for the in-browser demo, matching src/lab_to_field/inference.py exactly.
//
// The browser's own canvas resize is not the same as Pillow's, and the model is sensitive to it, so this
// file re-implements Pillow's bilinear resample (Resample.c: antialiased when shrinking, 22-bit fixed point,
// rounded to 8 bits between the horizontal and vertical pass). web/test checks it against Python.

export const IMAGE_SIZE = 224;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
const PRECISION_BITS = 22; // 32 - 8 - 2, as in Pillow
const ONE = 2 ** PRECISION_BITS;

/** Pillow's precompute_coeffs + normalize_coeffs_8bpc for the bilinear filter. */
function coefficients(inSize, outSize) {
  const scale = inSize / outSize;
  const filterscale = Math.max(scale, 1);
  const support = 1.0 * filterscale;
  const ss = 1 / filterscale;
  const bounds = new Int32Array(outSize * 2);
  const weights = [];
  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    const xmin = Math.max(Math.trunc(center - support + 0.5), 0);
    const xmax = Math.min(Math.trunc(center + support + 0.5), inSize) - xmin;
    const k = new Float64Array(xmax);
    let ww = 0;
    for (let x = 0; x < xmax; x++) {
      const d = Math.abs((x + xmin - center + 0.5) * ss);
      k[x] = d < 1 ? 1 - d : 0;
      ww += k[x];
    }
    const ki = new Float64Array(xmax); // integer weights, kept in doubles (exact up to 2^53)
    for (let x = 0; x < xmax; x++) {
      const w = ww !== 0 ? k[x] / ww : k[x];
      ki[x] = w < 0 ? Math.trunc(-0.5 + w * ONE) : Math.trunc(0.5 + w * ONE);
    }
    bounds[xx * 2] = xmin;
    bounds[xx * 2 + 1] = xmax;
    weights.push(ki);
  }
  return { bounds, weights };
}

function clip8(v) {
  if (v >= ONE * 256) return 255;
  if (v <= 0) return 0;
  return Math.floor(v / ONE);
}

/**
 * Resize an RGB(A) pixel buffer like PIL's `img.resize((outW, outH), Image.BILINEAR)`.
 * @param {Uint8Array|Uint8ClampedArray} src  pixels, `channels` bytes each (3 = RGB, 4 = RGBA; alpha ignored)
 * @returns {Uint8Array} RGB pixels of size outW x outH
 */
export function resizeBilinear(src, w, h, outW, outH, channels = 4) {
  // Horizontal pass: (w x h) -> (outW x h)
  let tmp = new Uint8Array(outW * h * 3);
  if (outW === w) {
    for (let i = 0, j = 0; i < w * h; i++, j += channels) tmp.set([src[j], src[j + 1], src[j + 2]], i * 3);
  } else {
    const { bounds, weights } = coefficients(w, outW);
    for (let y = 0; y < h; y++) {
      const row = y * w;
      for (let xx = 0; xx < outW; xx++) {
        const xmin = bounds[xx * 2], k = weights[xx];
        let r = ONE / 2, g = ONE / 2, b = ONE / 2;
        for (let x = 0; x < k.length; x++) {
          const p = (row + xmin + x) * channels;
          r += src[p] * k[x]; g += src[p + 1] * k[x]; b += src[p + 2] * k[x];
        }
        const o = (y * outW + xx) * 3;
        tmp[o] = clip8(r); tmp[o + 1] = clip8(g); tmp[o + 2] = clip8(b);
      }
    }
  }
  if (outH === h) return tmp;
  // Vertical pass: (outW x h) -> (outW x outH)
  const out = new Uint8Array(outW * outH * 3);
  const { bounds, weights } = coefficients(h, outH);
  for (let yy = 0; yy < outH; yy++) {
    const ymin = bounds[yy * 2], k = weights[yy];
    for (let x = 0; x < outW; x++) {
      let r = ONE / 2, g = ONE / 2, b = ONE / 2;
      for (let y = 0; y < k.length; y++) {
        const p = ((ymin + y) * outW + x) * 3;
        r += tmp[p] * k[y]; g += tmp[p + 1] * k[y]; b += tmp[p + 2] * k[y];
      }
      const o = (yy * outW + x) * 3;
      out[o] = clip8(r); out[o + 1] = clip8(g); out[o + 2] = clip8(b);
    }
  }
  return out;
}

/** Python's round(): halves go to the nearest even number. */
function roundHalfEven(x) {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

/**
 * Same as Python `preprocess`: resize the shorter side to 255, center-crop 224, ImageNet-normalise.
 * @returns {{tensor: Float32Array, crop: Uint8Array}} NCHW float32 tensor (1x3x224x224) and the RGB crop
 */
export function preprocess(pixels, w, h, channels = 4, size = IMAGE_SIZE) {
  const short = Math.trunc(size * 1.14);
  const [newW, newH] = w <= h ? [short, Math.trunc((short * h) / w)] : [Math.trunc((short * w) / h), short];
  const resized = resizeBilinear(pixels, w, h, newW, newH, channels);
  const left = roundHalfEven((newW - size) / 2), top = roundHalfEven((newH - size) / 2);
  const plane = size * size;
  const tensor = new Float32Array(3 * plane);
  const crop = new Uint8Array(plane * 3);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const s = ((top + y) * newW + left + x) * 3, i = y * size + x;
      for (let c = 0; c < 3; c++) {
        const v = resized[s + c];
        crop[i * 3 + c] = v;
        // numpy: float32(v / 255) promoted to float64 for the mean/std step, then stored as float32
        tensor[c * plane + i] = (Math.fround(v / 255) - MEAN[c]) / STD[c];
      }
    }
  }
  return { tensor, crop };
}

/** Softmax of logits / temperature, in float64 like the Python code. */
export function softmax(logits, temperature = 1) {
  const z = Array.from(logits, (v) => v / temperature);
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

/** Calibrated diagnosis: the disease name, or "Not sure" below the abstain threshold. */
export function diagnose(logits, { classes, temperature, abstain_threshold: threshold }) {
  const probs = softmax(logits, temperature);
  const order = probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]);
  const [confidence, best] = order[0];
  const sure = confidence >= threshold;
  return {
    answer: sure ? classes[best] : "Not sure",
    bestGuess: classes[best],
    confidence,
    sure,
    threshold,
    top: order.slice(0, 3).map(([p, i]) => ({ label: classes[i], prob: p })),
  };
}
