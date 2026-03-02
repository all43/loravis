import { useState, useRef, useCallback, useEffect } from "react";

// ══════════════════════════════════════════════════════════════════
// UTILITY HELPERS
// ══════════════════════════════════════════════════════════════════
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
function makeCanvas(w, h) {
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  return c;
}
function drawToCanvas(src, w, h) {
  const c = makeCanvas(w, h);
  c.getContext("2d").drawImage(src, 0, 0, w, h);
  return c;
}
function getPx(canvas) {
  return canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height);
}
function fromPx(data, w, h) {
  const c = makeCanvas(w, h);
  const ctx = c.getContext("2d");
  const id = ctx.createImageData(w, h);
  id.data.set(data);
  ctx.putImageData(id, 0, 0);
  return c;
}
function psnrCalc(a, b, n) {
  let mse = 0;
  for (let i = 0; i < n * 4; i += 4)
    for (let c = 0; c < 3; c++) { const d = a[i + c] - b[i + c]; mse += d * d; }
  mse /= n * 3;
  return mse === 0 ? 99 : (10 * Math.log10(65025 / mse));
}
function kMeansPalette(pixels, k, maxIter = 10) {
  const n = pixels.length / 4;
  const centroids = [];
  for (let i = 0; i < k; i++) {
    const idx = Math.floor((i / k) * n) * 4;
    centroids.push([pixels[idx], pixels[idx + 1], pixels[idx + 2]]);
  }
  let assignments = new Uint8Array(n);
  for (let iter = 0; iter < maxIter; iter++) {
    for (let i = 0; i < n; i++) {
      const r = pixels[i * 4], g = pixels[i * 4 + 1], b = pixels[i * 4 + 2];
      let best = 0, bestD = Infinity;
      for (let c = 0; c < k; c++) {
        const dr = r - centroids[c][0], dg = g - centroids[c][1], db = b - centroids[c][2];
        const d = dr * dr + dg * dg + db * db;
        if (d < bestD) { bestD = d; best = c; }
      }
      assignments[i] = best;
    }
    const sums = Array.from({ length: k }, () => [0, 0, 0, 0]);
    for (let i = 0; i < n; i++) {
      const c = assignments[i];
      sums[c][0] += pixels[i * 4]; sums[c][1] += pixels[i * 4 + 1];
      sums[c][2] += pixels[i * 4 + 2]; sums[c][3]++;
    }
    for (let c = 0; c < k; c++) {
      if (sums[c][3] > 0) {
        centroids[c] = [sums[c][0] / sums[c][3], sums[c][1] / sums[c][3], sums[c][2] / sums[c][3]];
      }
    }
  }
  return { centroids, assignments };
}

function entropyBytes(data) {
  const freq = new Map();
  for (const b of data) freq.set(b, (freq.get(b) || 0) + 1);
  let bits = 0;
  const n = data.length;
  for (const count of freq.values()) {
    const p = count / n;
    bits -= count * Math.log2(p);
  }
  return Math.ceil(bits / 8);
}

// ══════════════════════════════════════════════════════════════════
// IMAGE ANALYSIS ENGINE
// ══════════════════════════════════════════════════════════════════
function analyzeImage(srcCanvas) {
  const W = 64, ratio = srcCanvas.height / srcCanvas.width;
  const H = Math.round(W * ratio);
  const thumb = drawToCanvas(srcCanvas, W, H);
  const px = getPx(thumb).data;
  const N = W * H;

  // Grayscale
  const gray = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    gray[i] = 0.299 * px[i * 4] + 0.587 * px[i * 4 + 1] + 0.114 * px[i * 4 + 2];
  }

  // 1. Edge Density — Sobel, ratio of edge pixels
  let edgeCount = 0;
  const edgeBits = new Uint8Array(N);
  for (let y = 1; y < H - 1; y++) {
    for (let x = 1; x < W - 1; x++) {
      const gx = -gray[(y - 1) * W + (x - 1)] + gray[(y - 1) * W + (x + 1)]
        - 2 * gray[y * W + (x - 1)] + 2 * gray[y * W + (x + 1)]
        - gray[(y + 1) * W + (x - 1)] + gray[(y + 1) * W + (x + 1)];
      const gy = -gray[(y - 1) * W + (x - 1)] - 2 * gray[(y - 1) * W + x] - gray[(y - 1) * W + (x + 1)]
        + gray[(y + 1) * W + (x - 1)] + 2 * gray[(y + 1) * W + x] + gray[(y + 1) * W + (x + 1)];
      const mag = Math.sqrt(gx * gx + gy * gy);
      if (mag > 35) { edgeCount++; edgeBits[y * W + x] = 1; }
    }
  }
  const edgeDensity = clamp(edgeCount / N / 0.25, 0, 1); // normalize: 25% edges → 1.0

  // 2. Color Complexity — unique colors after quantizing to 4-bit per channel
  const colorSet = new Set();
  for (let i = 0; i < N; i++) {
    const r = px[i * 4] >> 4, g = px[i * 4 + 1] >> 4, b = px[i * 4 + 2] >> 4;
    colorSet.add((r << 8) | (g << 4) | b);
  }
  const maxPossible = Math.min(N, 4096);
  const colorComplexity = clamp(colorSet.size / (maxPossible * 0.3), 0, 1);

  // 3. Spatial Frequency — avg gradient magnitude (pixel-to-pixel)
  let gradSum = 0, gradCount = 0;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W - 1; x++) {
      const d = Math.abs(gray[y * W + x] - gray[y * W + x + 1]);
      gradSum += d; gradCount++;
    }
  }
  for (let y = 0; y < H - 1; y++) {
    for (let x = 0; x < W; x++) {
      const d = Math.abs(gray[y * W + x] - gray[(y + 1) * W + x]);
      gradSum += d; gradCount++;
    }
  }
  const avgGrad = gradSum / gradCount;
  const spatialFrequency = clamp(avgGrad / 40, 0, 1); // normalize: avg grad 40 → 1.0

  // 4. Block Uniformity — 1 - (avg intra-block variance / max)
  const blockSize = 4;
  let totalVar = 0, blockCount = 0;
  for (let by = 0; by < Math.floor(H / blockSize); by++) {
    for (let bx = 0; bx < Math.floor(W / blockSize); bx++) {
      let sum = 0, sum2 = 0, cnt = 0;
      for (let dy = 0; dy < blockSize; dy++) {
        for (let dx = 0; dx < blockSize; dx++) {
          const v = gray[(by * blockSize + dy) * W + (bx * blockSize + dx)];
          sum += v; sum2 += v * v; cnt++;
        }
      }
      const mean = sum / cnt;
      const variance = sum2 / cnt - mean * mean;
      totalVar += variance;
      blockCount++;
    }
  }
  const avgBlockVar = totalVar / blockCount;
  const blockUniformity = clamp(1 - avgBlockVar / 2000, 0, 1); // 2000 is high variance

  // 5. Gradient Smoothness — how well a tiny bilinear upscale approximates the original
  const smallW = 8, smallH = Math.max(4, Math.round(8 * ratio));
  const small = drawToCanvas(srcCanvas, smallW, smallH);
  const upscaled = drawToCanvas(small, W, H);
  const upPx = getPx(upscaled).data;
  const smoothPSNR = psnrCalc(px, upPx, N);
  const gradientSmoothness = clamp((smoothPSNR - 10) / 25, 0, 1); // PSNR 10→0, 35→1

  // 6. Pattern Repetition — patch self-similarity via k-means codebook fit
  const patchSize = 4;
  const pW = Math.floor(W / patchSize), pH = Math.floor(H / patchSize);
  const pDim = patchSize * patchSize * 3;
  const patches = [];
  for (let py = 0; py < pH; py++) {
    for (let px2 = 0; px2 < pW; px2++) {
      const patch = new Float32Array(pDim);
      let idx = 0;
      for (let dy = 0; dy < patchSize; dy++) {
        for (let dx = 0; dx < patchSize; dx++) {
          const x = px2 * patchSize + dx, y = py * patchSize + dy;
          const si = (y * W + x) * 4;
          patch[idx++] = px[si]; patch[idx++] = px[si + 1]; patch[idx++] = px[si + 2];
        }
      }
      patches.push(patch);
    }
  }
  // Quick k-means with K=8, measure how well 8 centroids represent all patches
  const K = 8;
  let centroids = [];
  for (let i = 0; i < K; i++) centroids.push(new Float32Array(patches[Math.floor((i / K) * patches.length)]));
  let assignments = new Uint8Array(patches.length);
  for (let iter = 0; iter < 8; iter++) {
    for (let i = 0; i < patches.length; i++) {
      let bestD = Infinity, bestC = 0;
      for (let c = 0; c < K; c++) {
        let d = 0;
        for (let j = 0; j < pDim; j++) { const diff = patches[i][j] - centroids[c][j]; d += diff * diff; }
        if (d < bestD) { bestD = d; bestC = c; }
      }
      assignments[i] = bestC;
    }
    const sums = Array.from({ length: K }, () => new Float32Array(pDim));
    const counts = new Uint32Array(K);
    for (let i = 0; i < patches.length; i++) {
      const c = assignments[i]; counts[c]++;
      for (let j = 0; j < pDim; j++) sums[c][j] += patches[i][j];
    }
    for (let c = 0; c < K; c++) {
      if (counts[c] > 0) for (let j = 0; j < pDim; j++) centroids[c][j] = sums[c][j] / counts[c];
    }
  }
  // Measure reconstruction error from codebook
  let totalPatchError = 0;
  for (let i = 0; i < patches.length; i++) {
    const c = assignments[i];
    for (let j = 0; j < pDim; j++) {
      const diff = patches[i][j] - centroids[c][j];
      totalPatchError += diff * diff;
    }
  }
  const avgPatchMSE = totalPatchError / (patches.length * pDim);
  const patternRepetition = clamp(1 - avgPatchMSE / 3000, 0, 1); // low MSE = high repetition

  return {
    edgeDensity,
    colorComplexity,
    spatialFrequency,
    blockUniformity,
    gradientSmoothness,
    patternRepetition,
  };
}

// ══════════════════════════════════════════════════════════════════
// ALL 7 COMPRESSION STRATEGIES
// ══════════════════════════════════════════════════════════════════

// ── 1. Micro Thumbnail ────────────────────────────────────────────
function encodeMicroThumb(srcCanvas, budget) {
  const k = 8, bpp = 3;
  const headerBytes = k * 3 + 2;
  const availBits = (budget - headerBytes) * 8;
  const maxPixels = Math.floor(availBits / bpp);
  const side = Math.min(Math.floor(Math.sqrt(maxPixels)), 32);
  const thumb = drawToCanvas(srcCanvas, side, side);
  const px = getPx(thumb).data;
  const { centroids, assignments } = kMeansPalette(px, k);
  const packet = [side, side];
  for (const c of centroids) packet.push(Math.round(c[0]), Math.round(c[1]), Math.round(c[2]));
  let buf = 0, bits = 0;
  for (let i = 0; i < assignments.length; i++) {
    buf = (buf << bpp) | assignments[i]; bits += bpp;
    if (bits >= 8) { packet.push((buf >> (bits - 8)) & 0xFF); bits -= 8; }
  }
  if (bits > 0) packet.push((buf << (8 - bits)) & 0xFF);
  const decoded = new Uint8ClampedArray(side * side * 4);
  for (let i = 0; i < assignments.length; i++) {
    const c = centroids[assignments[i]];
    decoded[i * 4] = Math.round(c[0]); decoded[i * 4 + 1] = Math.round(c[1]);
    decoded[i * 4 + 2] = Math.round(c[2]); decoded[i * 4 + 3] = 255;
  }
  const origPx = getPx(drawToCanvas(srcCanvas, side, side)).data;
  return {
    key: "micro_thumb", name: "Micro Thumbnail",
    desc: `${side}x${side}, ${k}-color palette, 3bpp`,
    bytes: packet.length,
    recon: fromPx(decoded, side, side),
    reconW: side, reconH: side,
    psnr: psnrCalc(origPx, decoded, side * side).toFixed(1),
  };
}

// ── 2. Dithered 4-Color ──────────────────────────────────────────
function encodeDithered(srcCanvas, budget) {
  const k = 4;
  const side = Math.min(48, srcCanvas.width);
  const ratio = srcCanvas.height / srcCanvas.width;
  const h = Math.round(side * ratio);
  const thumb = drawToCanvas(srcCanvas, side, h);
  const px = new Uint8ClampedArray(getPx(thumb).data);
  const { centroids } = kMeansPalette(px, k, 12);
  const work = new Float32Array(side * h * 3);
  for (let i = 0; i < side * h; i++) {
    work[i * 3] = px[i * 4]; work[i * 3 + 1] = px[i * 4 + 1]; work[i * 3 + 2] = px[i * 4 + 2];
  }
  const indices = new Uint8Array(side * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < side; x++) {
      const i = y * side + x;
      const r = work[i * 3], g = work[i * 3 + 1], b = work[i * 3 + 2];
      let best = 0, bestD = Infinity;
      for (let c = 0; c < k; c++) {
        const dr = r - centroids[c][0], dg = g - centroids[c][1], db = b - centroids[c][2];
        const d = dr * dr + dg * dg + db * db;
        if (d < bestD) { bestD = d; best = c; }
      }
      indices[i] = best;
      const er = r - centroids[best][0], eg = g - centroids[best][1], eb = b - centroids[best][2];
      const spread = [[x + 1, y, 7 / 16], [x - 1, y + 1, 3 / 16], [x, y + 1, 5 / 16], [x + 1, y + 1, 1 / 16]];
      for (const [nx, ny, w] of spread) {
        if (nx >= 0 && nx < side && ny < h) {
          const ni = ny * side + nx;
          work[ni * 3] += er * w; work[ni * 3 + 1] += eg * w; work[ni * 3 + 2] += eb * w;
        }
      }
    }
  }
  const packet = [side, h];
  for (const c of centroids) packet.push(Math.round(c[0]), Math.round(c[1]), Math.round(c[2]));
  for (let i = 0; i < indices.length; i += 4) {
    let byte = 0;
    for (let j = 0; j < 4 && i + j < indices.length; j++) byte |= (indices[i + j] & 3) << (6 - j * 2);
    packet.push(byte);
  }
  const decoded = new Uint8ClampedArray(side * h * 4);
  for (let i = 0; i < side * h; i++) {
    const c = centroids[indices[i]];
    decoded[i * 4] = clamp(Math.round(c[0]), 0, 255);
    decoded[i * 4 + 1] = clamp(Math.round(c[1]), 0, 255);
    decoded[i * 4 + 2] = clamp(Math.round(c[2]), 0, 255);
    decoded[i * 4 + 3] = 255;
  }
  const origPx = getPx(drawToCanvas(srcCanvas, side, h)).data;
  return {
    key: "dithered", name: "Dithered 4-Color",
    desc: `${side}x${h}, Floyd-Steinberg, 2bpp`,
    bytes: packet.length,
    recon: fromPx(decoded, side, h),
    reconW: side, reconH: h,
    psnr: psnrCalc(origPx, decoded, side * h).toFixed(1),
  };
}

// ── 3. Block Average ──────────────────────────────────────────────
function encodeBlockAverage(srcCanvas, budget) {
  const blockSize = 4;
  const bw = Math.min(16, Math.floor(srcCanvas.width / blockSize));
  const bh = Math.min(16, Math.floor(srcCanvas.height / blockSize));
  const tw = bw * blockSize, th = bh * blockSize;
  const thumb = drawToCanvas(srcCanvas, tw, th);
  const px = getPx(thumb).data;
  const packet = [bw, bh, blockSize];
  const blocks = [];
  for (let by = 0; by < bh; by++) {
    for (let bx = 0; bx < bw; bx++) {
      let r = 0, g = 0, b = 0;
      for (let dy = 0; dy < blockSize; dy++) {
        for (let dx = 0; dx < blockSize; dx++) {
          const idx = ((by * blockSize + dy) * tw + (bx * blockSize + dx)) * 4;
          r += px[idx]; g += px[idx + 1]; b += px[idx + 2];
        }
      }
      const n = blockSize * blockSize;
      const qr = Math.round(r / n / 8), qg = Math.round(g / n / 8), qb = Math.round(b / n / 8);
      blocks.push([qr, qg, qb]);
      packet.push((qr << 2) | (qg >> 3));
      packet.push(((qg & 7) << 5) | qb);
    }
  }
  const decoded = new Uint8ClampedArray(tw * th * 4);
  let bi = 0;
  for (let by = 0; by < bh; by++) {
    for (let bx = 0; bx < bw; bx++) {
      const [qr, qg, qb] = blocks[bi++];
      for (let dy = 0; dy < blockSize; dy++) {
        for (let dx = 0; dx < blockSize; dx++) {
          const idx = ((by * blockSize + dy) * tw + (bx * blockSize + dx)) * 4;
          decoded[idx] = qr * 8; decoded[idx + 1] = qg * 8; decoded[idx + 2] = qb * 8; decoded[idx + 3] = 255;
        }
      }
    }
  }
  const origPx = getPx(drawToCanvas(srcCanvas, tw, th)).data;
  return {
    key: "block_avg", name: "Block Average",
    desc: `${bw}x${bh} blocks, RGB555`,
    bytes: packet.length,
    recon: fromPx(decoded, tw, th),
    reconW: tw, reconH: th,
    psnr: psnrCalc(origPx, decoded, tw * th).toFixed(1),
  };
}

// ── 4. Edge Sketch ────────────────────────────────────────────────
function encodeEdgeSketch(srcCanvas, budget) {
  const side = Math.min(64, srcCanvas.width);
  const ratio = srcCanvas.height / srcCanvas.width;
  const h = Math.round(side * ratio);
  const thumb = drawToCanvas(srcCanvas, side, h);
  const px = getPx(thumb).data;
  const gray = new Float32Array(side * h);
  for (let i = 0; i < side * h; i++) gray[i] = 0.299 * px[i * 4] + 0.587 * px[i * 4 + 1] + 0.114 * px[i * 4 + 2];
  const edges = new Uint8Array(side * h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < side - 1; x++) {
      const gx = -gray[(y - 1) * side + (x - 1)] + gray[(y - 1) * side + (x + 1)]
        - 2 * gray[y * side + (x - 1)] + 2 * gray[y * side + (x + 1)]
        - gray[(y + 1) * side + (x - 1)] + gray[(y + 1) * side + (x + 1)];
      const gy = -gray[(y - 1) * side + (x - 1)] - 2 * gray[(y - 1) * side + x] - gray[(y - 1) * side + (x + 1)]
        + gray[(y + 1) * side + (x - 1)] + 2 * gray[(y + 1) * side + x] + gray[(y + 1) * side + (x + 1)];
      edges[y * side + x] = Math.sqrt(gx * gx + gy * gy) > 40 ? 1 : 0;
    }
  }
  const packet = [side, h];
  let run = 0, current = 0;
  for (let i = 0; i < edges.length; i++) {
    if (edges[i] === current && run < 255) run++;
    else { packet.push(run); run = 1; current = edges[i]; }
  }
  packet.push(run);
  let avgR = 0, avgG = 0, avgB = 0;
  for (let i = 0; i < px.length; i += 4) { avgR += px[i]; avgG += px[i + 1]; avgB += px[i + 2]; }
  const np = px.length / 4;
  avgR = Math.round(avgR / np); avgG = Math.round(avgG / np); avgB = Math.round(avgB / np);
  packet.push(avgR, avgG, avgB);
  const decoded = new Uint8ClampedArray(side * h * 4);
  for (let i = 0; i < edges.length; i++) {
    if (edges[i]) { decoded[i * 4] = 255; decoded[i * 4 + 1] = 255; decoded[i * 4 + 2] = 255; }
    else {
      decoded[i * 4] = Math.round(avgR * 0.15);
      decoded[i * 4 + 1] = Math.round(avgG * 0.15);
      decoded[i * 4 + 2] = Math.round(avgB * 0.15);
    }
    decoded[i * 4 + 3] = 255;
  }
  const origPx = getPx(drawToCanvas(srcCanvas, side, h)).data;
  return {
    key: "edge_sketch", name: "Edge Sketch",
    desc: `${side}x${h} Sobel edges, RLE`,
    bytes: Math.min(packet.length, entropyBytes(new Uint8Array(packet))),
    recon: fromPx(decoded, side, h),
    reconW: side, reconH: h,
    psnr: psnrCalc(origPx, decoded, side * h).toFixed(1),
  };
}

// ── 5. Vector Primitives ──────────────────────────────────────────
function encodeVectorPrimitives(srcCanvas, budget) {
  const side = 64;
  const ratio = srcCanvas.height / srcCanvas.width;
  const h = Math.round(side * ratio);
  const thumb = drawToCanvas(srcCanvas, side, h);
  const px = getPx(thumb).data;
  const bytesPerPrim = 5, headerBytes = 3;
  const maxPrims = Math.floor((budget - headerBytes) / bytesPerPrim);
  const numPrims = Math.min(maxPrims, 80);
  const target = new Float32Array(side * h * 3);
  for (let i = 0; i < side * h; i++) {
    target[i * 3] = px[i * 4]; target[i * 3 + 1] = px[i * 4 + 1]; target[i * 3 + 2] = px[i * 4 + 2];
  }
  const current = new Float32Array(side * h * 3);
  let avgR = 0, avgG = 0, avgB = 0;
  for (let i = 0; i < side * h; i++) { avgR += target[i * 3]; avgG += target[i * 3 + 1]; avgB += target[i * 3 + 2]; }
  avgR /= side * h; avgG /= side * h; avgB /= side * h;
  for (let i = 0; i < side * h; i++) { current[i * 3] = avgR; current[i * 3 + 1] = avgG; current[i * 3 + 2] = avgB; }
  const primitives = [];
  for (let p = 0; p < numPrims; p++) {
    let bestScore = -Infinity, bestPrim = null;
    for (let t = 0; t < 60; t++) {
      const cx = Math.floor(Math.random() * side), cy = Math.floor(Math.random() * h);
      const cr = Math.floor(Math.random() * 12) + 2;
      let sr = 0, sg = 0, sb = 0, count = 0;
      const x0 = Math.max(0, cx - cr), x1 = Math.min(side - 1, cx + cr);
      const y0 = Math.max(0, cy - cr), y1 = Math.min(h - 1, cy + cr);
      for (let y = y0; y <= y1; y++) {
        for (let x = x0; x <= x1; x++) {
          if ((x - cx) * (x - cx) + (y - cy) * (y - cy) <= cr * cr) {
            const idx = y * side + x;
            sr += target[idx * 3] - current[idx * 3];
            sg += target[idx * 3 + 1] - current[idx * 3 + 1];
            sb += target[idx * 3 + 2] - current[idx * 3 + 2];
            count++;
          }
        }
      }
      if (count === 0) continue;
      sr /= count; sg /= count; sb /= count;
      let score = 0;
      for (let y = y0; y <= y1; y++) {
        for (let x = x0; x <= x1; x++) {
          if ((x - cx) * (x - cx) + (y - cy) * (y - cy) <= cr * cr) {
            const idx = y * side + x;
            for (let c = 0; c < 3; c++) {
              const diff = target[idx * 3 + c] - current[idx * 3 + c];
              const newDiff = diff - [sr, sg, sb][c];
              score += diff * diff - newDiff * newDiff;
            }
          }
        }
      }
      if (score > bestScore) {
        bestScore = score;
        bestPrim = { cx, cy, cr, r: clamp(Math.round(sr), -128, 127), g: clamp(Math.round(sg), -128, 127), b: clamp(Math.round(sb), -128, 127) };
      }
    }
    if (!bestPrim) break;
    primitives.push(bestPrim);
    const { cx, cy, cr, r, g, b } = bestPrim;
    const x0 = Math.max(0, cx - cr), x1 = Math.min(side - 1, cx + cr);
    const y0 = Math.max(0, cy - cr), y1 = Math.min(h - 1, cy + cr);
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        if ((x - cx) * (x - cx) + (y - cy) * (y - cy) <= cr * cr) {
          const idx = y * side + x;
          current[idx * 3] = clamp(current[idx * 3] + r, 0, 255);
          current[idx * 3 + 1] = clamp(current[idx * 3 + 1] + g, 0, 255);
          current[idx * 3 + 2] = clamp(current[idx * 3 + 2] + b, 0, 255);
        }
      }
    }
  }
  const decoded = new Uint8ClampedArray(side * h * 4);
  for (let i = 0; i < side * h; i++) {
    decoded[i * 4] = clamp(Math.round(current[i * 3]), 0, 255);
    decoded[i * 4 + 1] = clamp(Math.round(current[i * 3 + 1]), 0, 255);
    decoded[i * 4 + 2] = clamp(Math.round(current[i * 3 + 2]), 0, 255);
    decoded[i * 4 + 3] = 255;
  }
  const origPx = getPx(drawToCanvas(srcCanvas, side, h)).data;
  return {
    key: "vector_prims", name: "Vector Primitives",
    desc: `${primitives.length} circles on ${side}x${h}`,
    bytes: headerBytes + primitives.length * bytesPerPrim,
    recon: fromPx(decoded, side, h),
    reconW: side, reconH: h,
    psnr: psnrCalc(origPx, decoded, side * h).toFixed(1),
  };
}

// ── 6. Hybrid Layered ─────────────────────────────────────────────
function encodeHybridLayered(srcCanvas, budget) {
  const W = 48, ratio = srcCanvas.height / srcCanvas.width;
  const H = Math.round(W * ratio);
  const thumb = drawToCanvas(srcCanvas, W, H);
  const px = getPx(thumb).data;
  const N = W * H;
  const gridW = 4, gridH = 4;
  const cellW = Math.floor(W / gridW), cellH = Math.floor(H / gridH);
  const colorMap = [];
  for (let gy = 0; gy < gridH; gy++) {
    for (let gx = 0; gx < gridW; gx++) {
      let r = 0, g = 0, b = 0, cnt = 0;
      for (let dy = 0; dy < cellH; dy++) {
        for (let dx = 0; dx < cellW; dx++) {
          const x = gx * cellW + dx, y = gy * cellH + dy;
          if (x < W && y < H) { const i = (y * W + x) * 4; r += px[i]; g += px[i + 1]; b += px[i + 2]; cnt++; }
        }
      }
      colorMap.push([Math.round(r / cnt), Math.round(g / cnt), Math.round(b / cnt)]);
    }
  }
  const layer1 = new Float32Array(N * 3);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const gx = (x + 0.5) / cellW - 0.5, gy = (y + 0.5) / cellH - 0.5;
      const gx0 = clamp(Math.floor(gx), 0, gridW - 1), gx1 = clamp(gx0 + 1, 0, gridW - 1);
      const gy0 = clamp(Math.floor(gy), 0, gridH - 1), gy1 = clamp(gy0 + 1, 0, gridH - 1);
      const fx = clamp(gx - gx0, 0, 1), fy = clamp(gy - gy0, 0, 1);
      const idx = y * W + x;
      for (let c = 0; c < 3; c++) {
        const v00 = colorMap[gy0 * gridW + gx0][c], v10 = colorMap[gy0 * gridW + gx1][c];
        const v01 = colorMap[gy1 * gridW + gx0][c], v11 = colorMap[gy1 * gridW + gx1][c];
        layer1[idx * 3 + c] = v00 * (1 - fx) * (1 - fy) + v10 * fx * (1 - fy) + v01 * (1 - fx) * fy + v11 * fx * fy;
      }
    }
  }
  const layer1Bytes = gridW * gridH * 3;
  const gray = new Float32Array(N);
  for (let i = 0; i < N; i++) gray[i] = 0.299 * px[i * 4] + 0.587 * px[i * 4 + 1] + 0.114 * px[i * 4 + 2];
  const edgeBits = new Uint8Array(N);
  for (let y = 1; y < H - 1; y++) {
    for (let x = 1; x < W - 1; x++) {
      const gx = -gray[(y - 1) * W + (x - 1)] + gray[(y - 1) * W + (x + 1)]
        - 2 * gray[y * W + (x - 1)] + 2 * gray[y * W + (x + 1)]
        - gray[(y + 1) * W + (x - 1)] + gray[(y + 1) * W + (x + 1)];
      const gy2 = -gray[(y - 1) * W + (x - 1)] - 2 * gray[(y - 1) * W + x] - gray[(y - 1) * W + (x + 1)]
        + gray[(y + 1) * W + (x - 1)] + 2 * gray[(y + 1) * W + x] + gray[(y + 1) * W + (x + 1)];
      edgeBits[y * W + x] = Math.sqrt(gx * gx + gy2 * gy2) > 35 ? 1 : 0;
    }
  }
  const edgeRLE = [];
  let run = 0, cur = 0;
  for (let i = 0; i < N; i++) {
    if (edgeBits[i] === cur && run < 255) run++;
    else { edgeRLE.push(run); run = 1; cur = edgeBits[i]; }
  }
  edgeRLE.push(run);
  const layer2Bytes = Math.min(edgeRLE.length, Math.ceil(N / 8) + 2);
  const residual = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    residual[i * 3] = px[i * 4] - layer1[i * 3];
    residual[i * 3 + 1] = px[i * 4 + 1] - layer1[i * 3 + 1];
    residual[i * 3 + 2] = px[i * 4 + 2] - layer1[i * 3 + 2];
  }
  const rBlockW = Math.ceil(W / 4), rBlockH = Math.ceil(H / 4);
  const residBlocks = [];
  for (let by = 0; by < rBlockH; by++) {
    for (let bx = 0; bx < rBlockW; bx++) {
      let sr = 0, sg = 0, sb = 0, cnt = 0;
      for (let dy = 0; dy < 4; dy++) {
        for (let dx = 0; dx < 4; dx++) {
          const x = bx * 4 + dx, y = by * 4 + dy;
          if (x < W && y < H) { const i = y * W + x; sr += residual[i * 3]; sg += residual[i * 3 + 1]; sb += residual[i * 3 + 2]; cnt++; }
        }
      }
      residBlocks.push([
        clamp(Math.round(sr / cnt / 4), -31, 31),
        clamp(Math.round(sg / cnt / 4), -31, 31),
        clamp(Math.round(sb / cnt / 4), -31, 31),
      ]);
    }
  }
  const remainingBudget = budget - layer1Bytes - layer2Bytes - 4;
  const residBlockBytes = residBlocks.length * 2;
  const useResidual = remainingBudget >= residBlockBytes;
  const layer3Bytes = useResidual ? residBlockBytes : 0;
  const finalPx = new Float32Array(N * 3);
  for (let i = 0; i < N * 3; i++) finalPx[i] = layer1[i];
  if (useResidual) {
    for (let by = 0; by < rBlockH; by++) {
      for (let bx = 0; bx < rBlockW; bx++) {
        const blk = residBlocks[by * rBlockW + bx];
        for (let dy = 0; dy < 4; dy++) {
          for (let dx = 0; dx < 4; dx++) {
            const x = bx * 4 + dx, y = by * 4 + dy;
            if (x < W && y < H) {
              const i = y * W + x;
              finalPx[i * 3] += blk[0] * 4; finalPx[i * 3 + 1] += blk[1] * 4; finalPx[i * 3 + 2] += blk[2] * 4;
            }
          }
        }
      }
    }
  }
  for (let i = 0; i < N; i++) {
    if (edgeBits[i]) {
      const origLum = 0.299 * px[i * 4] + 0.587 * px[i * 4 + 1] + 0.114 * px[i * 4 + 2];
      const curLum = 0.299 * finalPx[i * 3] + 0.587 * finalPx[i * 3 + 1] + 0.114 * finalPx[i * 3 + 2];
      const shift = (origLum - curLum) * 0.4;
      finalPx[i * 3] += shift; finalPx[i * 3 + 1] += shift; finalPx[i * 3 + 2] += shift;
    }
  }
  const decoded = new Uint8ClampedArray(N * 4);
  for (let i = 0; i < N; i++) {
    decoded[i * 4] = clamp(Math.round(finalPx[i * 3]), 0, 255);
    decoded[i * 4 + 1] = clamp(Math.round(finalPx[i * 3 + 1]), 0, 255);
    decoded[i * 4 + 2] = clamp(Math.round(finalPx[i * 3 + 2]), 0, 255);
    decoded[i * 4 + 3] = 255;
  }
  const totalBytes = 4 + layer1Bytes + layer2Bytes + layer3Bytes;
  const origPx = getPx(drawToCanvas(srcCanvas, W, H)).data;
  return {
    key: "hybrid", name: "Hybrid Layered",
    desc: `3-layer: color map + edges + residual`,
    bytes: totalBytes,
    recon: fromPx(decoded, W, H),
    reconW: W, reconH: H,
    psnr: psnrCalc(origPx, decoded, N).toFixed(1),
  };
}

// ── 7. VQ Patch Codebook ──────────────────────────────────────────
function encodeVQPatch(srcCanvas, budget) {
  const W = 48, ratio = srcCanvas.height / srcCanvas.width;
  const H = Math.round(W * ratio);
  const thumb = drawToCanvas(srcCanvas, W, H);
  const px = getPx(thumb).data;
  const patchSize = 4, K = 16;
  const pW = Math.floor(W / patchSize), pH = Math.floor(H / patchSize);
  const pDim = patchSize * patchSize * 3;
  const patches = [];
  for (let py = 0; py < pH; py++) {
    for (let px2 = 0; px2 < pW; px2++) {
      const patch = new Float32Array(pDim);
      let idx = 0;
      for (let dy = 0; dy < patchSize; dy++) {
        for (let dx = 0; dx < patchSize; dx++) {
          const x = px2 * patchSize + dx, y = py * patchSize + dy;
          const si = (y * W + x) * 4;
          patch[idx++] = px[si]; patch[idx++] = px[si + 1]; patch[idx++] = px[si + 2];
        }
      }
      patches.push(patch);
    }
  }
  let centroids = [];
  for (let i = 0; i < K; i++) centroids.push(new Float32Array(patches[Math.floor((i / K) * patches.length)]));
  let assignments = new Uint8Array(patches.length);
  for (let iter = 0; iter < 15; iter++) {
    for (let i = 0; i < patches.length; i++) {
      let bestD = Infinity, bestC = 0;
      for (let c = 0; c < K; c++) {
        let d = 0;
        for (let j = 0; j < pDim; j++) { const diff = patches[i][j] - centroids[c][j]; d += diff * diff; }
        if (d < bestD) { bestD = d; bestC = c; }
      }
      assignments[i] = bestC;
    }
    const sums = Array.from({ length: K }, () => new Float32Array(pDim));
    const counts = new Uint32Array(K);
    for (let i = 0; i < patches.length; i++) {
      const c = assignments[i]; counts[c]++;
      for (let j = 0; j < pDim; j++) sums[c][j] += patches[i][j];
    }
    for (let c = 0; c < K; c++) {
      if (counts[c] > 0) for (let j = 0; j < pDim; j++) centroids[c][j] = sums[c][j] / counts[c];
    }
  }
  const bitsPerIndex = Math.ceil(Math.log2(K));
  const qCodebook = centroids.map(c => {
    const q = new Uint8Array(pDim);
    for (let j = 0; j < pDim; j++) q[j] = clamp(Math.round(c[j] / 8), 0, 31);
    return q;
  });
  const qCodebookBytes = Math.ceil(K * pDim * 5 / 8);
  const indexBytes = Math.ceil(patches.length * bitsPerIndex / 8);
  const totalBytes = 4 + qCodebookBytes + indexBytes;
  const reconData = new Uint8ClampedArray(W * H * 4);
  for (let i = 0; i < W * H * 4; i += 4) { reconData[i] = 0; reconData[i + 1] = 0; reconData[i + 2] = 0; reconData[i + 3] = 255; }
  for (let py = 0; py < pH; py++) {
    for (let px2 = 0; px2 < pW; px2++) {
      const ci = assignments[py * pW + px2];
      const cent = qCodebook[ci];
      let idx = 0;
      for (let dy = 0; dy < patchSize; dy++) {
        for (let dx = 0; dx < patchSize; dx++) {
          const x = px2 * patchSize + dx, y = py * patchSize + dy;
          const di = (y * W + x) * 4;
          reconData[di] = cent[idx++] * 8; reconData[di + 1] = cent[idx++] * 8; reconData[di + 2] = cent[idx++] * 8;
        }
      }
    }
  }
  const origPx = getPx(drawToCanvas(srcCanvas, W, H)).data;
  return {
    key: "vq_patch", name: "VQ Patch Codebook",
    desc: `K=${K}, ${patchSize}x${patchSize} patches`,
    bytes: totalBytes,
    recon: fromPx(reconData, W, H),
    reconW: W, reconH: H,
    psnr: psnrCalc(origPx, reconData, W * H).toFixed(1),
  };
}

// ══════════════════════════════════════════════════════════════════
// STRATEGY SCORING
// ══════════════════════════════════════════════════════════════════
const STRATEGY_PROFILES = {
  micro_thumb: {
    name: "Micro Thumbnail",
    weights: { edgeDensity: -0.3, colorComplexity: -0.5, spatialFrequency: -0.3, blockUniformity: 0.6, gradientSmoothness: 0.3, patternRepetition: 0.2 },
    baseScore: 40,
    reason: (f) => {
      const parts = [];
      if (f.blockUniformity > 0.5) parts.push(`high uniformity (${f.blockUniformity.toFixed(2)})`);
      if (f.colorComplexity < 0.4) parts.push(`few colors (${f.colorComplexity.toFixed(2)})`);
      if (parts.length === 0) parts.push("simple scenes with few details");
      return `Best for ${parts.join(", ")}`;
    },
  },
  dithered: {
    name: "Dithered 4-Color",
    weights: { edgeDensity: -0.1, colorComplexity: 0.1, spatialFrequency: 0.0, blockUniformity: 0.1, gradientSmoothness: 0.6, patternRepetition: 0.1 },
    baseScore: 48,
    reason: (f) => {
      const parts = [];
      if (f.gradientSmoothness > 0.4) parts.push(`smooth gradients (${f.gradientSmoothness.toFixed(2)})`);
      if (f.colorComplexity > 0.3 && f.colorComplexity < 0.7) parts.push("moderate color range");
      if (parts.length === 0) parts.push("natural scenes with smooth tones");
      return `Best for ${parts.join(", ")}`;
    },
  },
  block_avg: {
    name: "Block Average",
    weights: { edgeDensity: -0.4, colorComplexity: -0.2, spatialFrequency: -0.3, blockUniformity: 0.7, gradientSmoothness: 0.2, patternRepetition: 0.3 },
    baseScore: 42,
    reason: (f) => {
      const parts = [];
      if (f.blockUniformity > 0.6) parts.push(`very uniform blocks (${f.blockUniformity.toFixed(2)})`);
      if (f.edgeDensity < 0.3) parts.push("few edges");
      if (parts.length === 0) parts.push("geometric or uniform content");
      return `Best for ${parts.join(", ")}`;
    },
  },
  edge_sketch: {
    name: "Edge Sketch",
    weights: { edgeDensity: 0.8, colorComplexity: -0.3, spatialFrequency: 0.3, blockUniformity: -0.2, gradientSmoothness: -0.3, patternRepetition: -0.1 },
    baseScore: 35,
    reason: (f) => {
      const parts = [];
      if (f.edgeDensity > 0.5) parts.push(`high edge density (${f.edgeDensity.toFixed(2)})`);
      if (f.colorComplexity < 0.4) parts.push("color isn't critical");
      if (parts.length === 0) parts.push("structural/shape information priority");
      return `Best for ${parts.join(", ")}`;
    },
  },
  vector_prims: {
    name: "Vector Primitives",
    weights: { edgeDensity: -0.2, colorComplexity: 0.1, spatialFrequency: -0.4, blockUniformity: 0.2, gradientSmoothness: 0.6, patternRepetition: 0.0 },
    baseScore: 44,
    reason: (f) => {
      const parts = [];
      if (f.gradientSmoothness > 0.5) parts.push(`smooth regions (${f.gradientSmoothness.toFixed(2)})`);
      if (f.spatialFrequency < 0.4) parts.push("low spatial frequency");
      if (parts.length === 0) parts.push("smooth scenes with color blobs");
      return `Best for ${parts.join(", ")}`;
    },
  },
  hybrid: {
    name: "Hybrid Layered",
    weights: { edgeDensity: 0.3, colorComplexity: 0.2, spatialFrequency: 0.1, blockUniformity: 0.1, gradientSmoothness: 0.2, patternRepetition: 0.1 },
    baseScore: 55,
    reason: (f) => {
      const parts = [];
      if (f.edgeDensity > 0.3 && f.colorComplexity > 0.3) parts.push("balanced edges + color");
      else parts.push("versatile general-purpose approach");
      return `${parts[0]} — works well across diverse images`;
    },
  },
  vq_patch: {
    name: "VQ Patch Codebook",
    weights: { edgeDensity: -0.1, colorComplexity: 0.2, spatialFrequency: 0.1, blockUniformity: -0.1, gradientSmoothness: -0.1, patternRepetition: 0.8 },
    baseScore: 42,
    reason: (f) => {
      const parts = [];
      if (f.patternRepetition > 0.5) parts.push(`high pattern repetition (${f.patternRepetition.toFixed(2)})`);
      if (f.colorComplexity > 0.4) parts.push("rich colors");
      if (parts.length === 0) parts.push("textures and repeating patterns");
      return `Best for ${parts.join(", ")}`;
    },
  },
};

function scoreStrategies(features, results, budget) {
  const scored = [];
  for (const result of results) {
    const profile = STRATEGY_PROFILES[result.key];
    if (!profile) continue;
    let score = profile.baseScore;
    for (const [feat, weight] of Object.entries(profile.weights)) {
      score += (features[feat] || 0) * weight * 50;
    }
    // Budget fitness: bonus if within budget, penalty if over
    if (result.bytes <= budget) {
      score += 10;
      // Bonus for using budget efficiently (closer to budget = better)
      score += (result.bytes / budget) * 5;
    } else {
      score -= 20;
    }
    // PSNR bonus
    const psnrVal = parseFloat(result.psnr);
    if (psnrVal > 20) score += (psnrVal - 15) * 0.5;

    scored.push({
      ...result,
      score: Math.round(score),
      reason: profile.reason(features),
      withinBudget: result.bytes <= budget,
    });
  }
  scored.sort((a, b) => b.score - a.score);
  return scored;
}

// ══════════════════════════════════════════════════════════════════
// TEST SCENE GENERATORS
// ══════════════════════════════════════════════════════════════════
function generateLandscape() {
  const c = makeCanvas(128, 128), ctx = c.getContext("2d");
  const sky = ctx.createLinearGradient(0, 0, 0, 65);
  sky.addColorStop(0, "#0f2744"); sky.addColorStop(1, "#4a8db7");
  ctx.fillStyle = sky; ctx.fillRect(0, 0, 128, 65);
  const gnd = ctx.createLinearGradient(0, 65, 0, 128);
  gnd.addColorStop(0, "#5a9e4b"); gnd.addColorStop(1, "#2d6a1e");
  ctx.fillStyle = gnd; ctx.fillRect(0, 65, 128, 63);
  ctx.fillStyle = "#ffd700"; ctx.beginPath(); ctx.arc(100, 22, 14, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = "#8b4513"; ctx.fillRect(25, 48, 38, 32);
  ctx.fillStyle = "#cc6633"; ctx.beginPath(); ctx.moveTo(20, 48); ctx.lineTo(44, 26); ctx.lineTo(68, 48); ctx.fill();
  ctx.fillStyle = "#3366cc"; ctx.fillRect(36, 58, 12, 14);
  ctx.fillStyle = "#1a5c1a"; ctx.beginPath(); ctx.arc(95, 50, 14, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = "#663300"; ctx.fillRect(93, 62, 5, 16);
  ctx.fillStyle = "rgba(255,255,255,0.6)";
  [35, 46, 57].forEach((x, i) => { ctx.beginPath(); ctx.arc(x, 14, 7 + i % 2 * 3, 0, Math.PI * 2); ctx.fill(); });
  return c;
}

function generateHighEdge() {
  const c = makeCanvas(128, 128), ctx = c.getContext("2d");
  ctx.fillStyle = "#0a0a1a"; ctx.fillRect(0, 0, 128, 128);
  ctx.strokeStyle = "#00ff88"; ctx.lineWidth = 2;
  // Grid
  for (let i = 0; i < 128; i += 16) {
    ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, 128); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, i); ctx.lineTo(128, i); ctx.stroke();
  }
  // Triangles
  ctx.strokeStyle = "#ff4466"; ctx.lineWidth = 3;
  ctx.beginPath(); ctx.moveTo(32, 24); ctx.lineTo(16, 56); ctx.lineTo(48, 56); ctx.closePath(); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(96, 24); ctx.lineTo(72, 56); ctx.lineTo(120, 56); ctx.closePath(); ctx.stroke();
  // Circle
  ctx.strokeStyle = "#44aaff"; ctx.lineWidth = 3;
  ctx.beginPath(); ctx.arc(64, 96, 24, 0, Math.PI * 2); ctx.stroke();
  // Cross
  ctx.strokeStyle = "#ffcc00"; ctx.lineWidth = 3;
  ctx.beginPath(); ctx.moveTo(52, 84); ctx.lineTo(76, 108); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(76, 84); ctx.lineTo(52, 108); ctx.stroke();
  return c;
}

function generateSmoothGradient() {
  const c = makeCanvas(128, 128), ctx = c.getContext("2d");
  // Radial sunset gradient
  const grad1 = ctx.createRadialGradient(64, 100, 5, 64, 80, 100);
  grad1.addColorStop(0, "#ff6b35"); grad1.addColorStop(0.3, "#ff4466");
  grad1.addColorStop(0.6, "#6b2fa0"); grad1.addColorStop(1, "#0f0f2a");
  ctx.fillStyle = grad1; ctx.fillRect(0, 0, 128, 128);
  // Overlay glow
  const grad2 = ctx.createRadialGradient(30, 40, 5, 30, 40, 50);
  grad2.addColorStop(0, "rgba(255,200,100,0.4)"); grad2.addColorStop(1, "rgba(255,200,100,0)");
  ctx.fillStyle = grad2; ctx.fillRect(0, 0, 128, 128);
  return c;
}

function generateTexture() {
  const c = makeCanvas(128, 128), ctx = c.getContext("2d");
  // Checkerboard-like pattern with noise
  for (let y = 0; y < 128; y += 8) {
    for (let x = 0; x < 128; x += 8) {
      const base = ((x / 8 + y / 8) % 2 === 0) ? 60 : 30;
      const r = base + Math.floor(Math.random() * 30) + 30;
      const g = base + Math.floor(Math.random() * 50) + 60;
      const b = base + Math.floor(Math.random() * 20);
      ctx.fillStyle = `rgb(${r},${g},${b})`; ctx.fillRect(x, y, 8, 8);
      // Inner detail
      const r2 = r + 20, g2 = g - 10, b2 = b + 15;
      ctx.fillStyle = `rgb(${clamp(r2, 0, 255)},${clamp(g2, 0, 255)},${clamp(b2, 0, 255)})`;
      ctx.fillRect(x + 2, y + 2, 4, 4);
    }
  }
  return c;
}

function generateGeometric() {
  const c = makeCanvas(128, 128), ctx = c.getContext("2d");
  ctx.fillStyle = "#e8e0d4"; ctx.fillRect(0, 0, 128, 128);
  // Building-like blocks
  ctx.fillStyle = "#8b8b8b"; ctx.fillRect(10, 40, 30, 88);
  ctx.fillStyle = "#a0a0a0"; ctx.fillRect(45, 20, 35, 108);
  ctx.fillStyle = "#7a7a7a"; ctx.fillRect(85, 50, 35, 78);
  // Windows - uniform grid
  ctx.fillStyle = "#4a90d9";
  for (let bx of [10, 45, 85]) {
    const bw = bx === 45 ? 35 : bx === 85 ? 35 : 30;
    const by = bx === 10 ? 40 : bx === 45 ? 20 : 50;
    for (let wy = by + 5; wy < 128 - 8; wy += 12) {
      for (let wx = bx + 4; wx < bx + bw - 6; wx += 10) {
        ctx.fillRect(wx, wy, 6, 8);
      }
    }
  }
  // Sky gradient
  const sky = ctx.createLinearGradient(0, 0, 0, 40);
  sky.addColorStop(0, "#87ceeb"); sky.addColorStop(1, "#e8e0d4");
  ctx.fillStyle = sky; ctx.fillRect(0, 0, 128, 20);
  return c;
}

const TEST_SCENES = [
  { name: "Landscape", desc: "House, tree, sky — mixed content", gen: generateLandscape },
  { name: "High Edge", desc: "Grid, shapes — strong edges, minimal color", gen: generateHighEdge },
  { name: "Smooth Gradient", desc: "Sunset — smooth color transitions", gen: generateSmoothGradient },
  { name: "Texture", desc: "Repeating checkerboard — pattern repetition", gen: generateTexture },
  { name: "Geometric", desc: "Buildings — uniform blocks, structured", gen: generateGeometric },
];

// ══════════════════════════════════════════════════════════════════
// UI COMPONENTS
// ══════════════════════════════════════════════════════════════════

function CanvasView({ source, width = 140, height = 140, pixelated = true, border }) {
  const ref = useRef(null);
  useEffect(() => {
    if (!ref.current || !source) return;
    const ctx = ref.current.getContext("2d");
    ctx.imageSmoothingEnabled = !pixelated;
    ctx.clearRect(0, 0, width, height);
    const sw = source.width, sh = source.height;
    const scale = Math.min(width / sw, height / sh);
    const dw = sw * scale, dh = sh * scale;
    ctx.drawImage(source, (width - dw) / 2, (height - dh) / 2, dw, dh);
  }, [source, width, height, pixelated]);
  return (
    <canvas ref={ref} width={width} height={height}
      style={{ imageRendering: pixelated ? "pixelated" : "auto", width, height, borderRadius: 6, border: border || "1px solid #262d37", background: "#0a0e14" }} />
  );
}

// ── SVG Radar Chart ───────────────────────────────────────────────
function RadarChart({ features, size = 200 }) {
  const labels = [
    { key: "edgeDensity", label: "Edges", color: "#f0883e" },
    { key: "colorComplexity", label: "Color", color: "#bc8cff" },
    { key: "spatialFrequency", label: "Detail", color: "#58a6ff" },
    { key: "blockUniformity", label: "Uniform", color: "#3fb950" },
    { key: "gradientSmoothness", label: "Smooth", color: "#f778ba" },
    { key: "patternRepetition", label: "Pattern", color: "#ffd700" },
  ];
  const cx = size / 2, cy = size / 2, r = size / 2 - 30;
  const n = labels.length;
  const angleStep = (2 * Math.PI) / n;

  // Polygon points for each ring
  const rings = [0.25, 0.5, 0.75, 1.0];
  const getPoint = (i, val) => {
    const angle = -Math.PI / 2 + i * angleStep;
    return [cx + Math.cos(angle) * r * val, cy + Math.sin(angle) * r * val];
  };

  const dataPoints = labels.map((l, i) => getPoint(i, features[l.key] || 0));
  const dataPath = dataPoints.map((p, i) => `${i === 0 ? "M" : "L"}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" ") + " Z";

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} style={{ display: "block" }}>
      {/* Background rings */}
      {rings.map((rv, ri) => (
        <polygon key={ri}
          points={labels.map((_, i) => getPoint(i, rv).join(",")).join(" ")}
          fill="none" stroke="#21262d" strokeWidth={1} />
      ))}
      {/* Axis lines */}
      {labels.map((_, i) => {
        const [ex, ey] = getPoint(i, 1);
        return <line key={i} x1={cx} y1={cy} x2={ex} y2={ey} stroke="#21262d" strokeWidth={1} />;
      })}
      {/* Data polygon */}
      <polygon points={dataPoints.map(p => p.join(",")).join(" ")}
        fill="rgba(88,166,255,0.15)" stroke="#58a6ff" strokeWidth={2} />
      {/* Data points */}
      {dataPoints.map((p, i) => (
        <circle key={i} cx={p[0]} cy={p[1]} r={4}
          fill={labels[i].color} stroke="#0d1117" strokeWidth={2} />
      ))}
      {/* Labels */}
      {labels.map((l, i) => {
        const [lx, ly] = getPoint(i, 1.22);
        return (
          <text key={i} x={lx} y={ly} textAnchor="middle" dominantBaseline="middle"
            fill={l.color} fontSize={10} fontFamily="'JetBrains Mono', monospace" fontWeight={700}>
            {l.label}
          </text>
        );
      })}
      {/* Value labels */}
      {labels.map((l, i) => {
        const val = (features[l.key] || 0).toFixed(2);
        const [lx, ly] = getPoint(i, 1.42);
        return (
          <text key={`v${i}`} x={lx} y={ly} textAnchor="middle" dominantBaseline="middle"
            fill="#6b7b8d" fontSize={8} fontFamily="'JetBrains Mono', monospace">
            {val}
          </text>
        );
      })}
    </svg>
  );
}

// ── Strategy Ranking ──────────────────────────────────────────────
function StrategyRanking({ scored }) {
  const maxScore = Math.max(...scored.map(s => s.score), 1);
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      {scored.map((s, i) => {
        const isWinner = i === 0;
        const barPct = (s.score / maxScore) * 100;
        return (
          <div key={s.key} style={{
            display: "flex", alignItems: "center", gap: 10,
            padding: "8px 12px",
            background: isWinner ? "#0d2818" : "#0d1117",
            border: isWinner ? "1px solid #238636" : "1px solid #21262d",
            borderRadius: 8,
          }}>
            <div style={{
              width: 22, height: 22, borderRadius: "50%",
              background: isWinner ? "#238636" : "#21262d",
              display: "flex", alignItems: "center", justifyContent: "center",
              fontSize: 11, fontWeight: 700, color: isWinner ? "#fff" : "#8b949e",
              fontFamily: "'JetBrains Mono', monospace",
              flexShrink: 0,
            }}>
              {i + 1}
            </div>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 3 }}>
                <span style={{
                  fontSize: 12, fontWeight: 600,
                  color: isWinner ? "#3fb950" : "#e6edf3",
                  fontFamily: "'JetBrains Mono', monospace",
                }}>
                  {isWinner && "\u2605 "}{s.name}
                </span>
                <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                  <span style={{
                    fontSize: 10, color: s.withinBudget ? "#8b949e" : "#f85149",
                    fontFamily: "monospace",
                  }}>
                    {s.bytes}B {!s.withinBudget && "(over)"}
                  </span>
                  <span style={{
                    fontFamily: "'JetBrains Mono', monospace",
                    fontSize: 13, fontWeight: 700,
                    color: isWinner ? "#3fb950" : "#58a6ff",
                  }}>
                    {s.score}
                  </span>
                </div>
              </div>
              <div style={{ position: "relative", height: 4, background: "#21262d", borderRadius: 2 }}>
                <div style={{
                  position: "absolute", left: 0, top: 0, height: "100%",
                  width: `${barPct}%`,
                  background: isWinner
                    ? "linear-gradient(90deg, #238636, #3fb950)"
                    : "linear-gradient(90deg, #21262d, #484f58)",
                  borderRadius: 2,
                  transition: "width 0.4s ease",
                }} />
              </div>
              <div style={{ fontSize: 9, color: "#6b7b8d", marginTop: 3 }}>{s.reason}</div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ── Result Card (compact) ─────────────────────────────────────────
function ResultCard({ result, budget, isWinner }) {
  const overBudget = result.bytes > budget;
  const pct = ((result.bytes / budget) * 100).toFixed(0);
  return (
    <div style={{
      background: isWinner ? "#0d2818" : "#0d1117",
      border: `1px solid ${isWinner ? "#238636" : overBudget ? "#f8514966" : "#21262d"}`,
      borderRadius: 8, padding: 14,
      display: "flex", flexDirection: "column", gap: 8,
      flex: "1 1 200px", minWidth: 180, maxWidth: 240,
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
        <div>
          <div style={{
            fontFamily: "'JetBrains Mono', monospace", fontWeight: 700, fontSize: 11,
            color: isWinner ? "#3fb950" : "#e6edf3",
          }}>
            {isWinner && "\u2605 "}{result.name}
          </div>
          <div style={{ fontSize: 9, color: "#6b7b8d", marginTop: 1 }}>{result.desc}</div>
        </div>
        <div style={{
          fontFamily: "'JetBrains Mono', monospace", fontSize: 14, fontWeight: 700,
          color: overBudget ? "#f85149" : "#58a6ff",
        }}>
          {result.bytes}<span style={{ fontSize: 9, color: "#6b7b8d" }}>B</span>
        </div>
      </div>
      <div style={{ display: "flex", justifyContent: "center" }}>
        <CanvasView source={result.recon} width={130} height={130}
          border={isWinner ? "2px solid #23863666" : "1px solid #21262d"} />
      </div>
      <div style={{ position: "relative", height: 4, background: "#21262d", borderRadius: 2 }}>
        <div style={{
          position: "absolute", left: 0, top: 0, height: "100%",
          width: `${Math.min(100, parseFloat(pct))}%`,
          background: overBudget
            ? "linear-gradient(90deg, #f85149, #da3633)"
            : "linear-gradient(90deg, #238636, #3fb950)",
          borderRadius: 2,
        }} />
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 9, color: "#6b7b8d" }}>
        <span>PSNR: {result.psnr} dB</span>
        <span>{pct}% budget</span>
      </div>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════
// MAIN APP
// ══════════════════════════════════════════════════════════════════
export default function App() {
  const [srcCanvas, setSrc] = useState(null);
  const [srcUrl, setSrcUrl] = useState(null);
  const [budget, setBudget] = useState(200);
  const [features, setFeatures] = useState(null);
  const [scored, setScored] = useState(null);
  const [showAll, setShowAll] = useState(false);
  const [processing, setProcessing] = useState(false);
  const fileRef = useRef(null);

  const load = useCallback((src) => {
    if (src instanceof HTMLCanvasElement) { setSrc(src); setSrcUrl(src.toDataURL()); return; }
    const img = new Image();
    img.onload = () => {
      let w = img.width, h = img.height;
      const s = 192 / Math.max(w, h);
      if (s < 1) { w = Math.round(w * s); h = Math.round(h * s); }
      const c = drawToCanvas(img, w, h);
      setSrc(c); setSrcUrl(c.toDataURL());
    };
    img.src = src;
  }, []);

  const onFile = useCallback((e) => {
    const f = e.target.files?.[0];
    if (f) { const r = new FileReader(); r.onload = ev => load(ev.target.result); r.readAsDataURL(f); }
  }, [load]);

  useEffect(() => {
    if (!srcCanvas) return;
    setProcessing(true);
    const t = setTimeout(() => {
      const feat = analyzeImage(srcCanvas);
      const results = [
        encodeMicroThumb(srcCanvas, budget),
        encodeDithered(srcCanvas, budget),
        encodeBlockAverage(srcCanvas, budget),
        encodeEdgeSketch(srcCanvas, budget),
        encodeVectorPrimitives(srcCanvas, budget),
        encodeHybridLayered(srcCanvas, budget),
        encodeVQPatch(srcCanvas, budget),
      ];
      const sc = scoreStrategies(feat, results, budget);
      setFeatures(feat);
      setScored(sc);
      setProcessing(false);
    }, 50);
    return () => clearTimeout(t);
  }, [srcCanvas, budget]);

  const winner = scored?.[0];

  return (
    <div style={{
      minHeight: "100vh", background: "#010409", color: "#e6edf3",
      fontFamily: "'DM Sans', -apple-system, sans-serif",
    }}>
      <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet" />

      {/* ── Top Bar ── */}
      <div style={{
        borderBottom: "1px solid #21262d", padding: "12px 24px",
        display: "flex", alignItems: "center", gap: 12,
        background: "#0d1117",
      }}>
        <div style={{ width: 8, height: 8, borderRadius: "50%", background: "#3fb950", boxShadow: "0 0 6px #3fb95066" }} />
        <span style={{
          fontFamily: "'JetBrains Mono', monospace", fontSize: 12, fontWeight: 700,
          color: "#58a6ff", letterSpacing: 1.5, textTransform: "uppercase",
        }}>LoRaVis</span>
        <span style={{ fontSize: 11, color: "#484f58" }}>Adaptive Mode Selector</span>
        <div style={{ flex: 1 }} />
        {processing && <span style={{ fontSize: 11, color: "#58a6ff", fontFamily: "monospace" }}>Analyzing...</span>}
        <span style={{
          fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: "#6b7b8d",
          padding: "3px 10px", background: "#161b22", borderRadius: 4,
        }}>Budget: {budget}B</span>
      </div>

      <div style={{ maxWidth: 960, margin: "0 auto", padding: "24px 20px" }}>
        {/* ── Controls ── */}
        <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap", marginBottom: 20 }}>
          <input ref={fileRef} type="file" accept="image/*" onChange={onFile} style={{ display: "none" }} />
          <button onClick={() => fileRef.current?.click()} style={btnStyle}>Upload Image</button>

          {/* Test scene dropdown */}
          <div style={{ position: "relative", display: "inline-block" }}>
            <select
              onChange={e => { if (e.target.value !== "") { load(TEST_SCENES[+e.target.value].gen()); e.target.value = ""; } }}
              defaultValue=""
              style={{
                padding: "8px 14px", background: "transparent", color: "#58a6ff",
                border: "1px solid #58a6ff33", borderRadius: 6,
                fontFamily: "'DM Sans', sans-serif", fontSize: 13, fontWeight: 500, cursor: "pointer",
                appearance: "none", paddingRight: 28,
              }}
            >
              <option value="" disabled>Test Scenes</option>
              {TEST_SCENES.map((s, i) => (
                <option key={i} value={i} style={{ background: "#0d1117", color: "#e6edf3" }}>
                  {s.name} — {s.desc}
                </option>
              ))}
            </select>
            <span style={{
              position: "absolute", right: 10, top: "50%", transform: "translateY(-50%)",
              fontSize: 10, color: "#58a6ff", pointerEvents: "none",
            }}>&#9662;</span>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: 8, marginLeft: "auto" }}>
            <label style={{ fontSize: 10, color: "#6b7b8d", fontFamily: "monospace", letterSpacing: 1 }}>BYTE BUDGET</label>
            <input type="range" min={80} max={500} step={10} value={budget}
              onChange={e => setBudget(+e.target.value)}
              style={{ width: 120, accentColor: "#58a6ff" }} />
            <span style={{ fontFamily: "'JetBrains Mono', monospace", fontWeight: 700, fontSize: 16, color: "#58a6ff", minWidth: 50 }}>
              {budget}
            </span>
          </div>
        </div>

        {/* ── Empty state ── */}
        {!srcCanvas && (
          <div style={{
            height: 320, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
            border: "2px dashed #21262d", borderRadius: 12, color: "#30363d", gap: 8,
          }}>
            <div style={{ fontSize: 40, opacity: 0.3 }}>&#x1F4E1;</div>
            <div style={{ fontSize: 14 }}>Upload an image or pick a test scene to begin</div>
            <div style={{ fontSize: 11, color: "#484f58", marginTop: 4 }}>
              The analyzer will examine image properties and recommend the best compression strategy
            </div>
          </div>
        )}

        {srcCanvas && features && scored && (
          <>
            {/* ── Source Info ── */}
            <div style={{
              display: "flex", alignItems: "center", gap: 16, marginBottom: 24,
              padding: "12px 16px", background: "#0d1117", borderRadius: 10, border: "1px solid #21262d",
            }}>
              <img src={srcUrl} alt="" style={{
                width: 56, height: 56, borderRadius: 6, objectFit: "cover",
                imageRendering: "pixelated", border: "1px solid #21262d",
              }} />
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: 13, fontWeight: 600 }}>Source: {srcCanvas.width}x{srcCanvas.height}px</div>
                <div style={{ fontSize: 11, color: "#6b7b8d" }}>
                  Raw: {(srcCanvas.width * srcCanvas.height * 3).toLocaleString()}B
                  {" \u2192 "}Target: <b style={{ color: "#58a6ff" }}>{budget}B</b>
                  {" ("}
                  <span style={{ color: "#f0883e" }}>{((srcCanvas.width * srcCanvas.height * 3) / budget).toFixed(0)}:1</span>
                  {" ratio)"}
                </div>
              </div>
            </div>

            {/* ── Analysis Section ── */}
            <div style={{
              background: "#0d1117", border: "1px solid #21262d", borderRadius: 12,
              padding: 24, marginBottom: 20,
            }}>
              <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 20 }}>
                <div style={{
                  width: 28, height: 28, borderRadius: 6,
                  background: "linear-gradient(135deg, #21262d, #30363d)",
                  display: "flex", alignItems: "center", justifyContent: "center", fontSize: 14,
                }}>&#x1F50D;</div>
                <div>
                  <div style={{ fontFamily: "'JetBrains Mono', monospace", fontWeight: 700, fontSize: 14, color: "#e6edf3" }}>
                    Image Analysis
                  </div>
                  <div style={{ fontSize: 11, color: "#6b7b8d" }}>
                    6 features extracted to score each compression strategy
                  </div>
                </div>
              </div>

              <div style={{ display: "flex", gap: 24, flexWrap: "wrap", alignItems: "flex-start" }}>
                {/* Radar Chart */}
                <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                  <RadarChart features={features} size={220} />
                </div>

                {/* Strategy Ranking */}
                <div style={{ flex: 1, minWidth: 300 }}>
                  <div style={{
                    fontSize: 10, color: "#6b7b8d", fontFamily: "monospace",
                    letterSpacing: 1, marginBottom: 8, textTransform: "uppercase",
                  }}>Strategy Ranking</div>
                  <StrategyRanking scored={scored} />
                </div>
              </div>
            </div>

            {/* ── Recommendation Section ── */}
            {winner && (
              <div style={{
                background: "#0d2818", border: "1px solid #238636", borderRadius: 12,
                padding: 24, marginBottom: 20,
              }}>
                <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16 }}>
                  <div style={{
                    width: 28, height: 28, borderRadius: 6, background: "#238636",
                    display: "flex", alignItems: "center", justifyContent: "center",
                    fontSize: 14, color: "#fff",
                  }}>&#x2605;</div>
                  <div>
                    <div style={{ fontFamily: "'JetBrains Mono', monospace", fontWeight: 700, fontSize: 14, color: "#3fb950" }}>
                      Recommended: {winner.name}
                    </div>
                    <div style={{ fontSize: 11, color: "#8b949e" }}>{winner.reason}</div>
                  </div>
                  <div style={{ marginLeft: "auto", textAlign: "right" }}>
                    <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 20, fontWeight: 700, color: "#3fb950" }}>
                      {winner.score}<span style={{ fontSize: 11, color: "#6b7b8d" }}>pts</span>
                    </div>
                  </div>
                </div>

                <div style={{ display: "flex", gap: 20, alignItems: "center", flexWrap: "wrap" }}>
                  {/* Original */}
                  <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
                    <CanvasView source={srcCanvas} width={160} height={160} border="2px solid #30363d" />
                    <span style={{ fontSize: 10, color: "#6b7b8d" }}>Original</span>
                  </div>

                  <div style={{ fontSize: 24, color: "#238636" }}>{"\u2192"}</div>

                  {/* Reconstruction */}
                  <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
                    <CanvasView source={winner.recon} width={160} height={160} border="2px solid #238636" />
                    <span style={{ fontSize: 10, color: "#3fb950" }}>Reconstruction</span>
                  </div>

                  {/* Stats */}
                  <div style={{ flex: 1, minWidth: 140 }}>
                    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                      <div style={{
                        padding: "8px 12px", background: "#161b22", borderRadius: 6,
                        border: "1px solid #21262d",
                      }}>
                        <div style={{ fontSize: 9, color: "#6b7b8d", textTransform: "uppercase", letterSpacing: 0.5 }}>Size</div>
                        <div style={{ fontFamily: "'JetBrains Mono', monospace", fontWeight: 700, fontSize: 18, color: winner.withinBudget ? "#3fb950" : "#f85149" }}>
                          {winner.bytes}<span style={{ fontSize: 10, color: "#6b7b8d" }}>B</span>
                          <span style={{ fontSize: 10, color: "#6b7b8d", fontWeight: 400 }}> / {budget}B</span>
                        </div>
                      </div>
                      <div style={{
                        padding: "8px 12px", background: "#161b22", borderRadius: 6,
                        border: "1px solid #21262d",
                      }}>
                        <div style={{ fontSize: 9, color: "#6b7b8d", textTransform: "uppercase", letterSpacing: 0.5 }}>Quality</div>
                        <div style={{ fontFamily: "'JetBrains Mono', monospace", fontWeight: 700, fontSize: 18, color: "#58a6ff" }}>
                          {winner.psnr}<span style={{ fontSize: 10, color: "#6b7b8d" }}> dB</span>
                        </div>
                      </div>
                      <div style={{
                        padding: "8px 12px", background: "#161b22", borderRadius: 6,
                        border: "1px solid #21262d",
                      }}>
                        <div style={{ fontSize: 9, color: "#6b7b8d", textTransform: "uppercase", letterSpacing: 0.5 }}>Compression</div>
                        <div style={{ fontFamily: "'JetBrains Mono', monospace", fontWeight: 700, fontSize: 18, color: "#f0883e" }}>
                          {((srcCanvas.width * srcCanvas.height * 3) / winner.bytes).toFixed(0)}<span style={{ fontSize: 10, color: "#6b7b8d" }}>:1</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* ── Show All Toggle ── */}
            <button
              onClick={() => setShowAll(!showAll)}
              style={{
                width: "100%", padding: "10px 16px", marginBottom: 16,
                background: "#0d1117", color: "#58a6ff",
                border: "1px solid #21262d", borderRadius: 8,
                fontFamily: "'JetBrains Mono', monospace", fontSize: 12,
                cursor: "pointer", textAlign: "center",
              }}
            >
              {showAll ? "Hide all strategies \u25B2" : "Show all 7 strategies \u25BC"}
            </button>

            {/* ── All Strategies Grid ── */}
            {showAll && (
              <div style={{
                display: "flex", flexWrap: "wrap", gap: 12, marginBottom: 24,
                padding: 20, background: "#0d1117", borderRadius: 12,
                border: "1px solid #21262d",
              }}>
                <div style={{
                  width: "100%", fontSize: 10, color: "#6b7b8d", fontFamily: "monospace",
                  letterSpacing: 1, textTransform: "uppercase", marginBottom: 4,
                }}>All Strategies Comparison</div>
                {scored.map((r, i) => (
                  <ResultCard key={r.key} result={r} budget={budget} isWinner={i === 0} />
                ))}
              </div>
            )}

            {/* ── Footer Note ── */}
            <div style={{
              padding: "14px 18px", background: "#0d1117", borderRadius: 10,
              border: "1px solid #21262d", fontSize: 11, color: "#6b7b8d", lineHeight: 1.7,
            }}>
              <div style={{
                fontFamily: "'JetBrains Mono', monospace", fontWeight: 700, color: "#e6edf3",
                marginBottom: 6, fontSize: 11,
              }}>How Adaptive Selection Works</div>
              <div>
                The analyzer extracts 6 image features: <b style={{ color: "#f0883e" }}>edge density</b>,{" "}
                <b style={{ color: "#bc8cff" }}>color complexity</b>,{" "}
                <b style={{ color: "#58a6ff" }}>spatial frequency</b>,{" "}
                <b style={{ color: "#3fb950" }}>block uniformity</b>,{" "}
                <b style={{ color: "#f778ba" }}>gradient smoothness</b>, and{" "}
                <b style={{ color: "#ffd700" }}>pattern repetition</b>.
                Each strategy has a feature-weight profile tuned to its strengths.
                The final score combines feature fitness, budget compliance, and PSNR quality.
              </div>
            </div>
          </>
        )}
      </div>

      <style>{`
        * { box-sizing: border-box; }
        button:hover { filter: brightness(1.2); }
        input[type="range"] { height: 4px; }
        select:hover { filter: brightness(1.1); }
      `}</style>
    </div>
  );
}

const btnStyle = {
  padding: "8px 18px", background: "#21262d", color: "#e6edf3",
  border: "1px solid #30363d", borderRadius: 6,
  fontFamily: "'DM Sans', sans-serif", fontSize: 13, fontWeight: 500, cursor: "pointer",
};
