// Checks web/preprocess.js against the Python pipeline on every bundled photo.
// Run: PYTHONPATH=src python web/test/make_fixtures.py build/web-fixtures && node --test web/test/
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { test } from "node:test";

import { diagnose, preprocess, resizeBilinear, softmax } from "../preprocess.js";

const dir = process.env.WEB_FIXTURES ?? "build/web-fixtures";
const { meta, cases } = JSON.parse(readFileSync(join(dir, "cases.json"), "utf8"));

test("fixtures exist", () => assert.ok(cases.length >= 20));

for (const c of cases) {
  test(`same input tensor as Python: ${c.file}`, () => {
    const rgb = new Uint8Array(readFileSync(join(dir, `${c.id}.rgb`)));
    const buf = readFileSync(join(dir, `${c.id}.tensor`));
    const expected = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
    const { tensor } = preprocess(rgb, c.width, c.height, 3);
    assert.equal(tensor.length, expected.length);
    let worst = 0;
    for (let i = 0; i < tensor.length; i++) worst = Math.max(worst, Math.abs(tensor[i] - expected[i]));
    assert.ok(worst < 1e-6, `max difference ${worst}`);
  });

  test(`same diagnosis as Python: ${c.file}`, () => {
    const d = diagnose(c.logits, meta);
    assert.equal(d.answer, c.answer);
    assert.ok(Math.abs(d.confidence - c.confidence) < 1e-9);
  });
}

test("resize keeps a flat colour flat", () => {
  const src = new Uint8Array(40 * 30 * 4).map((_, i) => [10, 200, 90, 255][i % 4]);
  const out = resizeBilinear(src, 40, 30, 17, 13);
  for (let i = 0; i < out.length; i += 3) assert.deepEqual([...out.subarray(i, i + 3)], [10, 200, 90]);
});

test("temperature changes confidence, not the answer", () => {
  const z = [1.5, -0.3, 3.2, 0.7];
  const [sharp, soft] = [softmax(z, 0.5), softmax(z, 2)];
  assert.equal(sharp.indexOf(Math.max(...sharp)), soft.indexOf(Math.max(...soft)));
  assert.ok(Math.max(...sharp) > Math.max(...soft));
  assert.ok(Math.abs(soft.reduce((a, b) => a + b) - 1) < 1e-12);
});

test("abstains below the threshold", () => {
  const m = { classes: ["a", "b"], temperature: 1, abstain_threshold: 0.9 };
  assert.equal(diagnose([0.2, 0.1], m).answer, "Not sure");
  assert.equal(diagnose([5, 0], m).answer, "a");
});
