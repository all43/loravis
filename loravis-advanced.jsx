import { useState, useRef, useCallback, useEffect, useMemo } from "react";

// ── Helpers ──────────────────────────────────────────────────────
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
function psnr(a, b, n) {
  let mse = 0;
  for (let i = 0; i < n * 4; i += 4)
    for (let c = 0; c < 3; c++) { const d = a[i+c] - b[i+c]; mse += d*d; }
  mse /= n * 3;
  return mse === 0 ? 99 : (10 * Math.log10(65025 / mse));
}

function generateTestImage() {
  const c = makeCanvas(128, 128), ctx = c.getContext("2d");
  const sky = ctx.createLinearGradient(0, 0, 0, 65);
  sky.addColorStop(0, "#0f2744"); sky.addColorStop(1, "#4a8db7");
  ctx.fillStyle = sky; ctx.fillRect(0, 0, 128, 65);
  const gnd = ctx.createLinearGradient(0, 65, 0, 128);
  gnd.addColorStop(0, "#5a9e4b"); gnd.addColorStop(1, "#2d6a1e");
  ctx.fillStyle = gnd; ctx.fillRect(0, 65, 128, 63);
  ctx.fillStyle = "#ffd700"; ctx.beginPath(); ctx.arc(100, 22, 14, 0, Math.PI*2); ctx.fill();
  ctx.fillStyle = "#8b4513"; ctx.fillRect(25, 48, 38, 32);
  ctx.fillStyle = "#cc6633"; ctx.beginPath(); ctx.moveTo(20, 48); ctx.lineTo(44, 26); ctx.lineTo(68, 48); ctx.fill();
  ctx.fillStyle = "#3366cc"; ctx.fillRect(36, 58, 12, 14);
  ctx.fillStyle = "#ffcc00"; ctx.fillRect(38, 60, 3, 3); ctx.fillRect(43, 60, 3, 3);
  ctx.fillStyle = "#1a5c1a"; ctx.beginPath(); ctx.arc(95, 50, 14, 0, Math.PI*2); ctx.fill();
  ctx.fillStyle = "#663300"; ctx.fillRect(93, 62, 5, 16);
  ctx.fillStyle = "rgba(255,255,255,0.6)";
  [35,46,57].forEach((x,i) => { ctx.beginPath(); ctx.arc(x, 14, 7+i%2*3, 0, Math.PI*2); ctx.fill(); });
  // small flowers
  ctx.fillStyle = "#ff6b8a";
  [[15,90],[40,100],[75,85],[110,95]].forEach(([x,y]) => { ctx.beginPath(); ctx.arc(x, y, 2, 0, Math.PI*2); ctx.fill(); });
  return c;
}

// ── Canvas Renderer ──────────────────────────────────────────────
function CanvasView({ source, width = 160, height = 160, pixelated = true, label, border }) {
  const ref = useRef(null);
  useEffect(() => {
    if (!ref.current || !source) return;
    const ctx = ref.current.getContext("2d");
    ctx.imageSmoothingEnabled = !pixelated;
    ctx.clearRect(0, 0, width, height);
    const sw = source.width || source.videoWidth || width;
    const sh = source.height || source.videoHeight || height;
    const scale = Math.min(width / sw, height / sh);
    const dw = sw * scale, dh = sh * scale;
    ctx.drawImage(source, (width - dw) / 2, (height - dh) / 2, dw, dh);
  }, [source, width, height, pixelated]);

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
      <canvas
        ref={ref}
        width={width}
        height={height}
        style={{
          imageRendering: pixelated ? "pixelated" : "auto",
          width, height,
          borderRadius: 6,
          border: border || "1px solid #262d37",
          background: "#0a0e14",
        }}
      />
      {label && <span style={{ fontSize: 10, color: "#6b7b8d", fontFamily: "'JetBrains Mono', monospace" }}>{label}</span>}
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════
// STRATEGY 1: HYBRID LAYERED (Scene Descriptor + Residual)
// ══════════════════════════════════════════════════════════════════
function encodeHybridLayered(srcCanvas, budget) {
  const W = 48, ratio = srcCanvas.height / srcCanvas.width;
  const H = Math.round(W * ratio);
  const thumb = drawToCanvas(srcCanvas, W, H);
  const px = getPx(thumb).data;
  const N = W * H;

  // ── LAYER 1: Spatial Color Map (4×4 grid of average RGB) ──
  const gridW = 4, gridH = 4;
  const cellW = Math.floor(W / gridW), cellH = Math.floor(H / gridH);
  const colorMap = [];
  for (let gy = 0; gy < gridH; gy++) {
    for (let gx = 0; gx < gridW; gx++) {
      let r = 0, g = 0, b = 0, cnt = 0;
      for (let dy = 0; dy < cellH; dy++) {
        for (let dx = 0; dx < cellW; dx++) {
          const x = gx * cellW + dx, y = gy * cellH + dy;
          if (x < W && y < H) {
            const i = (y * W + x) * 4;
            r += px[i]; g += px[i+1]; b += px[i+2]; cnt++;
          }
        }
      }
      colorMap.push([Math.round(r/cnt), Math.round(g/cnt), Math.round(b/cnt)]);
    }
  }

  // Reconstruct from spatial color map (bilinear interpolation)
  const layer1 = new Float32Array(N * 3);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const gx = (x + 0.5) / cellW - 0.5, gy = (y + 0.5) / cellH - 0.5;
      const gx0 = clamp(Math.floor(gx), 0, gridW-1), gx1 = clamp(gx0+1, 0, gridW-1);
      const gy0 = clamp(Math.floor(gy), 0, gridH-1), gy1 = clamp(gy0+1, 0, gridH-1);
      const fx = clamp(gx - gx0, 0, 1), fy = clamp(gy - gy0, 0, 1);
      const idx = y * W + x;
      for (let c = 0; c < 3; c++) {
        const v00 = colorMap[gy0*gridW+gx0][c], v10 = colorMap[gy0*gridW+gx1][c];
        const v01 = colorMap[gy1*gridW+gx0][c], v11 = colorMap[gy1*gridW+gx1][c];
        layer1[idx*3+c] = v00*(1-fx)*(1-fy) + v10*fx*(1-fy) + v01*(1-fx)*fy + v11*fx*fy;
      }
    }
  }

  const layer1Canvas = (() => {
    const d = new Uint8ClampedArray(N * 4);
    for (let i = 0; i < N; i++) {
      d[i*4] = clamp(Math.round(layer1[i*3]), 0, 255);
      d[i*4+1] = clamp(Math.round(layer1[i*3+1]), 0, 255);
      d[i*4+2] = clamp(Math.round(layer1[i*3+2]), 0, 255);
      d[i*4+3] = 255;
    }
    return fromPx(d, W, H);
  })();
  const layer1Bytes = gridW * gridH * 3; // 48 bytes for 4x4 RGB

  // ── LAYER 2: Edge structure (compact binary edge map) ──
  const gray = new Float32Array(N);
  for (let i = 0; i < N; i++) gray[i] = 0.299*px[i*4] + 0.587*px[i*4+1] + 0.114*px[i*4+2];

  const edgeBits = new Uint8Array(N);
  for (let y = 1; y < H-1; y++) {
    for (let x = 1; x < W-1; x++) {
      const gx = -gray[(y-1)*W+(x-1)] + gray[(y-1)*W+(x+1)]
                -2*gray[y*W+(x-1)] + 2*gray[y*W+(x+1)]
                -gray[(y+1)*W+(x-1)] + gray[(y+1)*W+(x+1)];
      const gy2 = -gray[(y-1)*W+(x-1)] - 2*gray[(y-1)*W+x] - gray[(y-1)*W+(x+1)]
                  +gray[(y+1)*W+(x-1)] + 2*gray[(y+1)*W+x] + gray[(y+1)*W+(x+1)];
      edgeBits[y*W+x] = Math.sqrt(gx*gx + gy2*gy2) > 35 ? 1 : 0;
    }
  }

  // Pack bits + RLE
  const edgeRLE = [];
  let run = 0, cur = 0;
  for (let i = 0; i < N; i++) {
    if (edgeBits[i] === cur && run < 255) run++;
    else { edgeRLE.push(run); run = 1; cur = edgeBits[i]; }
  }
  edgeRLE.push(run);
  const layer2Bytes = Math.min(edgeRLE.length, Math.ceil(N / 8) + 2);

  const edgeCanvas = (() => {
    const d = new Uint8ClampedArray(N * 4);
    for (let i = 0; i < N; i++) {
      const v = edgeBits[i] ? 255 : 0;
      d[i*4] = v; d[i*4+1] = v; d[i*4+2] = v; d[i*4+3] = 255;
    }
    return fromPx(d, W, H);
  })();

  // ── LAYER 3: Quantized residual (DCT-like, budget permitting) ──
  const residual = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    residual[i*3]   = px[i*4]   - layer1[i*3];
    residual[i*3+1] = px[i*4+1] - layer1[i*3+1];
    residual[i*3+2] = px[i*4+2] - layer1[i*3+2];
  }

  // Simple block-DCT-like: average residual per 4×4 block, quantized
  const rBlockW = Math.ceil(W / 4), rBlockH = Math.ceil(H / 4);
  const residBlocks = [];
  for (let by = 0; by < rBlockH; by++) {
    for (let bx = 0; bx < rBlockW; bx++) {
      let sr = 0, sg = 0, sb = 0, cnt = 0;
      for (let dy = 0; dy < 4; dy++) {
        for (let dx = 0; dx < 4; dx++) {
          const x = bx*4+dx, y = by*4+dy;
          if (x < W && y < H) {
            const i = y*W+x;
            sr += residual[i*3]; sg += residual[i*3+1]; sb += residual[i*3+2]; cnt++;
          }
        }
      }
      residBlocks.push([
        clamp(Math.round(sr/cnt/4), -31, 31),
        clamp(Math.round(sg/cnt/4), -31, 31),
        clamp(Math.round(sb/cnt/4), -31, 31),
      ]);
    }
  }

  const remainingBudget = budget - layer1Bytes - layer2Bytes - 4; // 4 = header
  const residBlockBytes = residBlocks.length * 2; // pack as 6-bit signed x3
  const useResidual = remainingBudget >= residBlockBytes;
  const layer3Bytes = useResidual ? residBlockBytes : 0;

  // Apply residual correction
  const finalPx = new Float32Array(N * 3);
  for (let i = 0; i < N * 3; i++) finalPx[i] = layer1[i];

  if (useResidual) {
    for (let by = 0; by < rBlockH; by++) {
      for (let bx = 0; bx < rBlockW; bx++) {
        const blk = residBlocks[by * rBlockW + bx];
        for (let dy = 0; dy < 4; dy++) {
          for (let dx = 0; dx < 4; dx++) {
            const x = bx*4+dx, y = by*4+dy;
            if (x < W && y < H) {
              const i = y*W+x;
              finalPx[i*3]   += blk[0] * 4;
              finalPx[i*3+1] += blk[1] * 4;
              finalPx[i*3+2] += blk[2] * 4;
            }
          }
        }
      }
    }
  }

  // Edge enhancement on final
  for (let i = 0; i < N; i++) {
    if (edgeBits[i]) {
      // Sharpen edges by pulling toward original luminance
      const origLum = 0.299*px[i*4] + 0.587*px[i*4+1] + 0.114*px[i*4+2];
      const curLum = 0.299*finalPx[i*3] + 0.587*finalPx[i*3+1] + 0.114*finalPx[i*3+2];
      const shift = (origLum - curLum) * 0.4;
      finalPx[i*3]   += shift;
      finalPx[i*3+1] += shift;
      finalPx[i*3+2] += shift;
    }
  }

  const finalCanvas = (() => {
    const d = new Uint8ClampedArray(N * 4);
    for (let i = 0; i < N; i++) {
      d[i*4]   = clamp(Math.round(finalPx[i*3]), 0, 255);
      d[i*4+1] = clamp(Math.round(finalPx[i*3+1]), 0, 255);
      d[i*4+2] = clamp(Math.round(finalPx[i*3+2]), 0, 255);
      d[i*4+3] = 255;
    }
    return fromPx(d, W, H);
  })();

  const residualVis = (() => {
    const d = new Uint8ClampedArray(N * 4);
    for (let i = 0; i < N; i++) {
      d[i*4]   = clamp(Math.round(residual[i*3] + 128), 0, 255);
      d[i*4+1] = clamp(Math.round(residual[i*3+1] + 128), 0, 255);
      d[i*4+2] = clamp(Math.round(residual[i*3+2] + 128), 0, 255);
      d[i*4+3] = 255;
    }
    return fromPx(d, W, H);
  })();

  const totalBytes = 4 + layer1Bytes + layer2Bytes + layer3Bytes;
  const origPx = getPx(drawToCanvas(srcCanvas, W, H)).data;
  const reconPx = getPx(finalCanvas).data;
  const q = psnr(origPx, reconPx, N);

  return {
    name: "Hybrid Layered",
    totalBytes,
    psnr: q.toFixed(1),
    W, H,
    layers: [
      { name: "Spatial Color Map", desc: `${gridW}×${gridH} bilinear RGB grid`, bytes: layer1Bytes, canvas: layer1Canvas },
      { name: "Edge Structure", desc: `Sobel edge map, RLE packed`, bytes: layer2Bytes, canvas: edgeCanvas },
      { name: "Residual Correction", desc: useResidual ? `${rBlockW}×${rBlockH} signed blocks` : "Skipped (over budget)", bytes: layer3Bytes, canvas: residualVis },
    ],
    finalCanvas,
    budget,
  };
}

// ══════════════════════════════════════════════════════════════════
// STRATEGY 2: VQ-VAE–Style Patch Codebook
// ══════════════════════════════════════════════════════════════════
function encodeVQVAE(srcCanvas, budget, codebookSize = 16, patchSize = 4) {
  const W = 48, ratio = srcCanvas.height / srcCanvas.width;
  const H = Math.round(W * ratio);
  const thumb = drawToCanvas(srcCanvas, W, H);
  const px = getPx(thumb).data;

  const pW = Math.floor(W / patchSize), pH = Math.floor(H / patchSize);
  const pDim = patchSize * patchSize * 3; // Dimensions per patch

  // Extract all patches as vectors
  const patches = [];
  for (let py = 0; py < pH; py++) {
    for (let px2 = 0; px2 < pW; px2++) {
      const patch = new Float32Array(pDim);
      let idx = 0;
      for (let dy = 0; dy < patchSize; dy++) {
        for (let dx = 0; dx < patchSize; dx++) {
          const x = px2 * patchSize + dx, y = py * patchSize + dy;
          const si = (y * W + x) * 4;
          patch[idx++] = px[si]; patch[idx++] = px[si+1]; patch[idx++] = px[si+2];
        }
      }
      patches.push(patch);
    }
  }

  // K-means clustering on patches → codebook
  const K = codebookSize;
  let centroids = [];
  // Init: pick K evenly spaced patches
  for (let i = 0; i < K; i++) {
    const idx = Math.floor((i / K) * patches.length);
    centroids.push(new Float32Array(patches[idx]));
  }

  let assignments = new Uint8Array(patches.length);

  for (let iter = 0; iter < 15; iter++) {
    // Assign patches to nearest centroid
    for (let i = 0; i < patches.length; i++) {
      let bestD = Infinity, bestC = 0;
      for (let c = 0; c < K; c++) {
        let d = 0;
        for (let j = 0; j < pDim; j++) { const diff = patches[i][j] - centroids[c][j]; d += diff * diff; }
        if (d < bestD) { bestD = d; bestC = c; }
      }
      assignments[i] = bestC;
    }
    // Update centroids
    const sums = Array.from({length: K}, () => new Float32Array(pDim));
    const counts = new Uint32Array(K);
    for (let i = 0; i < patches.length; i++) {
      const c = assignments[i]; counts[c]++;
      for (let j = 0; j < pDim; j++) sums[c][j] += patches[i][j];
    }
    for (let c = 0; c < K; c++) {
      if (counts[c] > 0) {
        for (let j = 0; j < pDim; j++) centroids[c][j] = sums[c][j] / counts[c];
      }
    }
  }

  // Compute sizes
  const bitsPerIndex = Math.ceil(Math.log2(K));
  const codebookBytes = K * patchSize * patchSize * 3; // Full RGB codebook
  // Quantize codebook to 5-bit per channel to save space
  const qCodebook = centroids.map(c => {
    const q = new Uint8Array(pDim);
    for (let j = 0; j < pDim; j++) q[j] = clamp(Math.round(c[j] / 8), 0, 31);
    return q;
  });
  const qCodebookBytes = Math.ceil(K * pDim * 5 / 8);
  const indexBytes = Math.ceil(patches.length * bitsPerIndex / 8);
  const headerBytes = 4; // W, H, K, patchSize
  const totalBytes = headerBytes + qCodebookBytes + indexBytes;

  // Decode: reconstruct from quantized codebook + assignments
  const reconData = new Uint8ClampedArray(W * H * 4);
  // Fill with dark first
  for (let i = 0; i < W * H * 4; i += 4) { reconData[i] = 0; reconData[i+1] = 0; reconData[i+2] = 0; reconData[i+3] = 255; }

  for (let py = 0; py < pH; py++) {
    for (let px2 = 0; px2 < pW; px2++) {
      const ci = assignments[py * pW + px2];
      const cent = qCodebook[ci];
      let idx = 0;
      for (let dy = 0; dy < patchSize; dy++) {
        for (let dx = 0; dx < patchSize; dx++) {
          const x = px2 * patchSize + dx, y = py * patchSize + dy;
          const di = (y * W + x) * 4;
          reconData[di]   = cent[idx++] * 8;
          reconData[di+1] = cent[idx++] * 8;
          reconData[di+2] = cent[idx++] * 8;
          reconData[di+3] = 255;
        }
      }
    }
  }

  const reconCanvas = fromPx(reconData, W, H);

  // Build codebook visualization
  const cbVizSize = Math.ceil(Math.sqrt(K));
  const cbCanvas = makeCanvas(cbVizSize * patchSize * 2, cbVizSize * patchSize * 2);
  const cbCtx = cbCanvas.getContext("2d");
  cbCtx.fillStyle = "#111"; cbCtx.fillRect(0, 0, cbCanvas.width, cbCanvas.height);
  for (let c = 0; c < K; c++) {
    const cx = (c % cbVizSize) * patchSize * 2, cy = Math.floor(c / cbVizSize) * patchSize * 2;
    const patchCanvas = makeCanvas(patchSize, patchSize);
    const pctx = patchCanvas.getContext("2d");
    const pid = pctx.createImageData(patchSize, patchSize);
    let idx = 0;
    for (let dy = 0; dy < patchSize; dy++) {
      for (let dx = 0; dx < patchSize; dx++) {
        const pi = (dy * patchSize + dx) * 4;
        pid.data[pi]   = qCodebook[c][idx++] * 8;
        pid.data[pi+1] = qCodebook[c][idx++] * 8;
        pid.data[pi+2] = qCodebook[c][idx++] * 8;
        pid.data[pi+3] = 255;
      }
    }
    pctx.putImageData(pid, 0, 0);
    cbCtx.imageSmoothingEnabled = false;
    cbCtx.drawImage(patchCanvas, cx, cy, patchSize * 2, patchSize * 2);
  }

  // Index map visualization
  const indexCanvas = makeCanvas(pW, pH);
  const ictx = indexCanvas.getContext("2d");
  const iid = ictx.createImageData(pW, pH);
  for (let i = 0; i < pW * pH; i++) {
    const hue = (assignments[i] / K) * 360;
    const [r, g, b] = hslToRgb(hue, 80, 55);
    iid.data[i*4] = r; iid.data[i*4+1] = g; iid.data[i*4+2] = b; iid.data[i*4+3] = 255;
  }
  ictx.putImageData(iid, 0, 0);

  // Frequency per codebook entry
  const freq = new Uint32Array(K);
  for (const a of assignments) freq[a]++;

  const origPx2 = getPx(drawToCanvas(srcCanvas, W, H)).data;
  const reconPx = getPx(reconCanvas).data;
  const q = psnr(origPx2, reconPx, W * H);

  return {
    name: "VQ Patch Codebook",
    totalBytes,
    psnr: q.toFixed(1),
    W, H, K, patchSize,
    codebookBytes: qCodebookBytes,
    indexBytes,
    reconCanvas,
    codebookCanvas: cbCanvas,
    indexCanvas,
    freq,
    assignments,
    budget,
  };
}

function hslToRgb(h, s, l) {
  s /= 100; l /= 100;
  const c = (1 - Math.abs(2*l - 1)) * s;
  const x = c * (1 - Math.abs((h / 60) % 2 - 1));
  const m = l - c / 2;
  let r = 0, g = 0, b = 0;
  if (h < 60) { r = c; g = x; }
  else if (h < 120) { r = x; g = c; }
  else if (h < 180) { g = c; b = x; }
  else if (h < 240) { g = x; b = c; }
  else if (h < 300) { r = x; b = c; }
  else { r = c; b = x; }
  return [Math.round((r+m)*255), Math.round((g+m)*255), Math.round((b+m)*255)];
}

// ── Hex Dump ─────────────────────────────────────────────────────
function HexDump({ bytes, label }) {
  const hex = Array.from({length: Math.min(bytes, 80)}, (_, i) =>
    Math.floor(Math.random() * 256).toString(16).padStart(2, "0")
  ).join(" ");
  return (
    <div>
      {label && <div style={{ fontSize: 9, color: "#4a5568", fontFamily: "monospace", marginBottom: 3 }}>{label}</div>}
      <div style={{
        fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
        fontSize: 8.5, color: "#7fcc7f",
        background: "#060d06", padding: "5px 7px", borderRadius: 4,
        overflowX: "auto", whiteSpace: "pre-wrap", wordBreak: "break-all",
        lineHeight: 1.6, border: "1px solid #152015", maxHeight: 60, overflow: "hidden",
      }}>
        {hex}{bytes > 80 ? ` … +${bytes - 80}B` : ""}
      </div>
    </div>
  );
}

// ── Progress meter ───────────────────────────────────────────────
function ByteMeter({ used, total, segments }) {
  const pct = Math.min(100, (used / total) * 100);
  const over = used > total;
  let cumPct = 0;
  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, marginBottom: 3, color: "#6b7b8d" }}>
        <span>{used} / {total} bytes</span>
        <span style={{ color: over ? "#f85149" : "#3fb950" }}>{over ? "OVER BUDGET" : `${(total - used)}B remaining`}</span>
      </div>
      <div style={{ position: "relative", height: 10, background: "#161b22", borderRadius: 5, overflow: "hidden" }}>
        {segments?.map((seg, i) => {
          const segPct = (seg.bytes / total) * 100;
          const left = cumPct;
          cumPct += segPct;
          return (
            <div key={i} title={seg.name} style={{
              position: "absolute", left: `${left}%`, top: 0, height: "100%",
              width: `${Math.min(segPct, 100 - left)}%`,
              background: seg.color,
              borderRight: i < segments.length - 1 ? "1px solid #0d1117" : "none",
            }} />
          );
        })}
        {!segments && (
          <div style={{
            height: "100%", width: `${pct}%`,
            background: over ? "linear-gradient(90deg, #f85149, #da3633)" : "linear-gradient(90deg, #238636, #3fb950)",
            borderRadius: 5, transition: "width 0.3s",
          }} />
        )}
      </div>
      {segments && (
        <div style={{ display: "flex", gap: 10, marginTop: 5, flexWrap: "wrap" }}>
          {segments.map((seg, i) => (
            <div key={i} style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 9, color: "#8b949e" }}>
              <div style={{ width: 8, height: 8, borderRadius: 2, background: seg.color }} />
              {seg.name}: {seg.bytes}B
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Section wrapper ──────────────────────────────────────────────
function Section({ title, subtitle, icon, children }) {
  return (
    <div style={{
      background: "#0d1117",
      border: "1px solid #21262d",
      borderRadius: 12,
      padding: 24,
      marginBottom: 20,
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16 }}>
        <div style={{
          width: 32, height: 32, borderRadius: 8,
          background: "linear-gradient(135deg, #21262d, #30363d)",
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: 16,
        }}>
          {icon}
        </div>
        <div>
          <div style={{ fontFamily: "'JetBrains Mono', monospace", fontWeight: 700, fontSize: 15, color: "#e6edf3" }}>
            {title}
          </div>
          {subtitle && <div style={{ fontSize: 11, color: "#6b7b8d", marginTop: 1 }}>{subtitle}</div>}
        </div>
      </div>
      {children}
    </div>
  );
}

// ── Stat pill ────────────────────────────────────────────────────
function Stat({ label, value, unit, color }) {
  return (
    <div style={{
      padding: "6px 12px",
      background: "#161b22",
      borderRadius: 6,
      border: "1px solid #21262d",
      display: "inline-flex", alignItems: "baseline", gap: 5,
    }}>
      <span style={{ fontSize: 10, color: "#6b7b8d", textTransform: "uppercase", letterSpacing: 0.5 }}>{label}</span>
      <span style={{ fontFamily: "'JetBrains Mono', monospace", fontWeight: 700, fontSize: 16, color: color || "#58a6ff" }}>
        {value}
      </span>
      {unit && <span style={{ fontSize: 10, color: "#6b7b8d" }}>{unit}</span>}
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
  const [cbSize, setCbSize] = useState(16);
  const [patchSz, setPatchSz] = useState(4);
  const [hybrid, setHybrid] = useState(null);
  const [vqvae, setVqvae] = useState(null);
  const fileRef = useRef(null);

  const load = useCallback((src) => {
    if (src instanceof HTMLCanvasElement) { setSrc(src); setSrcUrl(src.toDataURL()); return; }
    const img = new Image();
    img.onload = () => {
      let w = img.width, h = img.height;
      const s = 192 / Math.max(w, h);
      if (s < 1) { w = Math.round(w*s); h = Math.round(h*s); }
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
    const t = setTimeout(() => {
      setHybrid(encodeHybridLayered(srcCanvas, budget));
      setVqvae(encodeVQVAE(srcCanvas, budget, cbSize, patchSz));
    }, 30);
    return () => clearTimeout(t);
  }, [srcCanvas, budget, cbSize, patchSz]);

  return (
    <div style={{
      minHeight: "100vh", background: "#010409", color: "#e6edf3",
      fontFamily: "'DM Sans', -apple-system, sans-serif",
    }}>
      <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet" />

      {/* ── Top Bar ── */}
      <div style={{
        borderBottom: "1px solid #21262d",
        padding: "12px 24px",
        display: "flex", alignItems: "center", gap: 12,
        background: "#0d1117",
      }}>
        <div style={{
          width: 8, height: 8, borderRadius: "50%",
          background: "#3fb950", boxShadow: "0 0 6px #3fb95066",
        }} />
        <span style={{
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: 12, fontWeight: 700, color: "#58a6ff",
          letterSpacing: 1.5, textTransform: "uppercase",
        }}>
          LoRaVis
        </span>
        <span style={{ fontSize: 11, color: "#484f58" }}>Advanced Compression Lab</span>
        <div style={{ flex: 1 }} />
        <span style={{
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: 11, color: "#6b7b8d",
          padding: "3px 10px", background: "#161b22", borderRadius: 4,
        }}>
          Budget: {budget}B
        </span>
      </div>

      <div style={{ maxWidth: 900, margin: "0 auto", padding: "24px 20px" }}>

        {/* ── Source Controls ── */}
        <div style={{
          display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap", marginBottom: 20,
        }}>
          <input ref={fileRef} type="file" accept="image/*" onChange={onFile} style={{ display: "none" }} />
          <button onClick={() => fileRef.current?.click()} style={btnStyle}>
            Upload Image
          </button>
          <button onClick={() => load(generateTestImage())} style={{...btnStyle, background: "transparent", color: "#58a6ff", border: "1px solid #58a6ff33"}}>
            Test Scene
          </button>
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

        {!srcCanvas && (
          <div style={{
            height: 320, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
            border: "2px dashed #21262d", borderRadius: 12, color: "#30363d", gap: 8,
          }}>
            <div style={{ fontSize: 40, opacity: 0.3 }}>📡</div>
            <div style={{ fontSize: 14 }}>Upload an image or use the test scene</div>
          </div>
        )}

        {srcCanvas && hybrid && vqvae && (
          <>
            {/* ── Source Info ── */}
            <div style={{
              display: "flex", alignItems: "center", gap: 16, marginBottom: 24,
              padding: "12px 16px", background: "#0d1117", borderRadius: 10, border: "1px solid #21262d",
            }}>
              <img src={srcUrl} alt="" style={{ width: 64, height: 64, borderRadius: 6, objectFit: "cover", imageRendering: "pixelated", border: "1px solid #21262d" }} />
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: 13, fontWeight: 600 }}>Source: {srcCanvas.width}×{srcCanvas.height}px</div>
                <div style={{ fontSize: 11, color: "#6b7b8d" }}>
                  Raw: {(srcCanvas.width * srcCanvas.height * 3).toLocaleString()}B →
                  Target: <b style={{color: "#58a6ff"}}>{budget}B</b> ({((srcCanvas.width * srcCanvas.height * 3) / budget).toFixed(0)}:1 ratio)
                </div>
              </div>
            </div>

            {/* ═══ HYBRID LAYERED ═══ */}
            <Section title="Hybrid Layered Encoding" subtitle="Scene descriptor → edge map → residual correction" icon="🧅">
              <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginBottom: 16 }}>
                <Stat label="Total" value={hybrid.totalBytes} unit="bytes" color={hybrid.totalBytes > budget ? "#f85149" : "#3fb950"} />
                <Stat label="PSNR" value={hybrid.psnr} unit="dB" />
                <Stat label="Layers" value={hybrid.layers.filter(l => l.bytes > 0).length} />
              </div>

              <ByteMeter used={hybrid.totalBytes} total={budget} segments={[
                { name: "Header", bytes: 4, color: "#484f58" },
                ...hybrid.layers.map((l, i) => ({
                  name: l.name, bytes: l.bytes,
                  color: ["#58a6ff", "#f0883e", "#bc8cff"][i],
                })),
              ]} />

              {/* Pipeline visualization */}
              <div style={{ marginTop: 20 }}>
                <div style={{ fontSize: 10, color: "#6b7b8d", fontFamily: "monospace", letterSpacing: 1, marginBottom: 10, textTransform: "uppercase" }}>
                  Encoding Pipeline
                </div>
                <div style={{ display: "flex", gap: 6, alignItems: "flex-start", flexWrap: "wrap" }}>
                  {hybrid.layers.map((layer, i) => (
                    <div key={i} style={{ display: "flex", alignItems: "center", gap: 6 }}>
                      <div style={{
                        background: "#161b22", border: "1px solid #21262d", borderRadius: 8, padding: 10,
                        display: "flex", flexDirection: "column", alignItems: "center", gap: 6, minWidth: 130,
                      }}>
                        <div style={{ fontSize: 10, fontWeight: 700, color: ["#58a6ff", "#f0883e", "#bc8cff"][i] }}>
                          Layer {i + 1}: {layer.name}
                        </div>
                        <CanvasView source={layer.canvas} width={120} height={120} label={layer.desc} />
                        <div style={{
                          fontSize: 10, fontFamily: "monospace",
                          color: layer.bytes > 0 ? "#e6edf3" : "#484f58",
                          background: "#0d1117", padding: "2px 8px", borderRadius: 3,
                        }}>
                          {layer.bytes}B
                        </div>
                      </div>
                      {i < hybrid.layers.length - 1 && (
                        <div style={{ color: "#30363d", fontSize: 18, flexShrink: 0 }}>→</div>
                      )}
                    </div>
                  ))}
                  <div style={{ color: "#30363d", fontSize: 18, display: "flex", alignItems: "center" }}>=</div>
                  <div style={{
                    background: "#161b22", border: "2px solid #3fb950", borderRadius: 8, padding: 10,
                    display: "flex", flexDirection: "column", alignItems: "center", gap: 6,
                  }}>
                    <div style={{ fontSize: 10, fontWeight: 700, color: "#3fb950" }}>Reconstruction</div>
                    <CanvasView source={hybrid.finalCanvas} width={120} height={120} border="1px solid #3fb95044" />
                    <div style={{ fontSize: 10, color: "#3fb950", fontFamily: "monospace" }}>
                      PSNR: {hybrid.psnr} dB
                    </div>
                  </div>
                </div>
              </div>

              <div style={{ marginTop: 14, fontSize: 11, color: "#6b7b8d", lineHeight: 1.6 }}>
                The spatial color map captures broad color regions with bilinear interpolation.
                Edge structure preserves important boundaries. The residual corrects what the
                first two layers missed. Each layer is independently useful — even Layer 1 alone
                gives a recognizable color impression.
              </div>
            </Section>

            {/* ═══ VQ-VAE ═══ */}
            <Section title="VQ Patch Codebook" subtitle="Vector-quantized patches — the core idea behind VQ-VAE" icon="🧬">
              {/* Controls */}
              <div style={{ display: "flex", gap: 16, flexWrap: "wrap", marginBottom: 16, alignItems: "center" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  <label style={sliderLabel}>Codebook K</label>
                  <input type="range" min={4} max={64} step={4} value={cbSize}
                    onChange={e => setCbSize(+e.target.value)} style={{ width: 80, accentColor: "#bc8cff" }} />
                  <span style={sliderVal}>{cbSize}</span>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  <label style={sliderLabel}>Patch</label>
                  <select value={patchSz} onChange={e => setPatchSz(+e.target.value)}
                    style={{ background: "#161b22", color: "#e6edf3", border: "1px solid #30363d", borderRadius: 4, padding: "3px 6px", fontSize: 12 }}>
                    <option value={2}>2×2</option>
                    <option value={4}>4×4</option>
                    <option value={8}>8×8</option>
                  </select>
                </div>
              </div>

              <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginBottom: 16 }}>
                <Stat label="Total" value={vqvae.totalBytes} unit="bytes" color={vqvae.totalBytes > budget ? "#f85149" : "#3fb950"} />
                <Stat label="PSNR" value={vqvae.psnr} unit="dB" />
                <Stat label="Codebook" value={vqvae.codebookBytes} unit="B" color="#bc8cff" />
                <Stat label="Indices" value={vqvae.indexBytes} unit="B" color="#f0883e" />
              </div>

              <ByteMeter used={vqvae.totalBytes} total={budget} segments={[
                { name: "Header", bytes: 4, color: "#484f58" },
                { name: "Codebook", bytes: vqvae.codebookBytes, color: "#bc8cff" },
                { name: "Indices", bytes: vqvae.indexBytes, color: "#f0883e" },
              ]} />

              {/* VQ Pipeline */}
              <div style={{ marginTop: 20 }}>
                <div style={{ fontSize: 10, color: "#6b7b8d", fontFamily: "monospace", letterSpacing: 1, marginBottom: 10, textTransform: "uppercase" }}>
                  VQ Pipeline
                </div>
                <div style={{ display: "flex", gap: 12, alignItems: "flex-start", flexWrap: "wrap" }}>
                  {/* Codebook */}
                  <div style={{
                    background: "#161b22", border: "1px solid #21262d", borderRadius: 8, padding: 10,
                    display: "flex", flexDirection: "column", alignItems: "center", gap: 6,
                  }}>
                    <div style={{ fontSize: 10, fontWeight: 700, color: "#bc8cff" }}>
                      Codebook ({vqvae.K} entries)
                    </div>
                    <CanvasView source={vqvae.codebookCanvas} width={120} height={120} label={`${vqvae.codebookBytes}B transmitted`} />
                    {/* Frequency bars */}
                    <div style={{ display: "flex", gap: 1, alignItems: "flex-end", height: 30, width: 120 }}>
                      {Array.from(vqvae.freq).map((f, i) => {
                        const maxF = Math.max(...vqvae.freq);
                        const h = maxF > 0 ? (f / maxF) * 28 + 2 : 2;
                        return (
                          <div key={i} style={{
                            flex: 1, height: h,
                            background: `hsl(${(i / vqvae.K) * 360}, 70%, 55%)`,
                            borderRadius: "2px 2px 0 0",
                            minWidth: 2,
                          }} />
                        );
                      })}
                    </div>
                    <div style={{ fontSize: 8, color: "#484f58" }}>Usage frequency</div>
                  </div>

                  <div style={{ color: "#30363d", fontSize: 18, alignSelf: "center" }}>+</div>

                  {/* Index map */}
                  <div style={{
                    background: "#161b22", border: "1px solid #21262d", borderRadius: 8, padding: 10,
                    display: "flex", flexDirection: "column", alignItems: "center", gap: 6,
                  }}>
                    <div style={{ fontSize: 10, fontWeight: 700, color: "#f0883e" }}>
                      Index Map
                    </div>
                    <CanvasView source={vqvae.indexCanvas} width={120} height={120} label={`${vqvae.indexBytes}B transmitted`} />
                    <div style={{ fontSize: 9, color: "#6b7b8d" }}>
                      {Math.ceil(Math.log2(vqvae.K))} bits/patch
                    </div>
                  </div>

                  <div style={{ color: "#30363d", fontSize: 18, alignSelf: "center" }}>=</div>

                  {/* Reconstruction */}
                  <div style={{
                    background: "#161b22", border: "2px solid #3fb950", borderRadius: 8, padding: 10,
                    display: "flex", flexDirection: "column", alignItems: "center", gap: 6,
                  }}>
                    <div style={{ fontSize: 10, fontWeight: 700, color: "#3fb950" }}>Reconstruction</div>
                    <CanvasView source={vqvae.reconCanvas} width={120} height={120} border="1px solid #3fb95044" />
                    <div style={{ fontSize: 10, color: "#3fb950", fontFamily: "monospace" }}>
                      PSNR: {vqvae.psnr} dB
                    </div>
                  </div>
                </div>
              </div>

              <div style={{ marginTop: 14, fontSize: 11, color: "#6b7b8d", lineHeight: 1.6 }}>
                The image is split into {patchSz}×{patchSz} patches. K-means finds {cbSize} representative
                patches (the codebook). Each patch is replaced by its nearest codebook entry. The
                tradeoff is clear: larger K = better quality but bigger codebook overhead. Smaller patches =
                more spatial detail but more indices to transmit. Try adjusting K and patch size to see the sweet spot.
              </div>

              <div style={{
                marginTop: 12, padding: "10px 14px", background: "#161b22", borderRadius: 6,
                border: "1px solid #21262d", fontSize: 11, color: "#8b949e", lineHeight: 1.6,
              }}>
                <b style={{ color: "#bc8cff" }}>Real VQ-VAE difference:</b> In a true VQ-VAE, a neural network
                encoder learns to map image patches to a latent space where quantization works optimally.
                The codebook entries aren't raw pixel patches — they're learned embeddings. A neural decoder
                reconstructs from these embeddings. This means the codebook can be pre-shared (trained on
                your domain), so only the indices need transmission. With K=256 and pre-shared codebook,
                you'd need only {Math.ceil(Math.floor(48 / patchSz) * Math.floor(Math.round(48 * srcCanvas.height / srcCanvas.width) / patchSz) * 8 / 8)}B for indices alone.
              </div>
            </Section>

            {/* ═══ HEAD-TO-HEAD ═══ */}
            <Section title="Comparison" subtitle="Side-by-side at current budget" icon="⚖️">
              <div style={{ display: "flex", gap: 20, justifyContent: "center", flexWrap: "wrap" }}>
                {[
                  { label: "Original", canvas: drawToCanvas(srcCanvas, 48, Math.round(48 * srcCanvas.height / srcCanvas.width)), psnrV: "∞", bytes: srcCanvas.width * srcCanvas.height * 3 },
                  { label: "Hybrid Layered", canvas: hybrid.finalCanvas, psnrV: hybrid.psnr, bytes: hybrid.totalBytes },
                  { label: "VQ Codebook", canvas: vqvae.reconCanvas, psnrV: vqvae.psnr, bytes: vqvae.totalBytes },
                ].map((item, i) => (
                  <div key={i} style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 6 }}>
                    <CanvasView source={item.canvas} width={150} height={150}
                      border={i === 0 ? "2px solid #30363d" : `2px solid ${item.bytes <= budget ? "#3fb950" : "#f85149"}`} />
                    <div style={{ fontFamily: "'JetBrains Mono', monospace", fontWeight: 700, fontSize: 12, color: "#e6edf3" }}>
                      {item.label}
                    </div>
                    <div style={{ fontSize: 10, color: "#6b7b8d" }}>
                      {item.bytes.toLocaleString()}B • PSNR {item.psnrV} dB
                    </div>
                  </div>
                ))}
              </div>
            </Section>

            {/* Architecture note */}
            <div style={{
              padding: "16px 20px", background: "#0d1117", borderRadius: 10,
              border: "1px solid #21262d", fontSize: 11, color: "#6b7b8d", lineHeight: 1.7,
            }}>
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontWeight: 700, color: "#e6edf3", marginBottom: 8, fontSize: 12 }}>
                💡 Next Steps for a Real Library
              </div>
              <div style={{ marginBottom: 4 }}><b style={{ color: "#58a6ff" }}>Pre-shared codebook:</b> Train VQ-VAE on your domain (security cams, agriculture, etc). Only transmit indices → 10-50B for a full frame.</div>
              <div style={{ marginBottom: 4 }}><b style={{ color: "#f0883e" }}>Temporal encoding:</b> Send full codebook once, then only changed indices per frame. Perfect for periodic imaging.</div>
              <div style={{ marginBottom: 4 }}><b style={{ color: "#bc8cff" }}>Hybrid fusion:</b> Use VQ-VAE for spatial content + edge layer for sharp boundaries. Best of both worlds.</div>
              <div><b style={{ color: "#3fb950" }}>Adaptive mode selection:</b> Analyze image characteristics → pick best strategy per frame. Scene description for simple views, VQ for complex ones.</div>
            </div>
          </>
        )}
      </div>

      <style>{`
        * { box-sizing: border-box; }
        button:hover { filter: brightness(1.2); }
        input[type="range"] { height: 4px; }
      `}</style>
    </div>
  );
}

const btnStyle = {
  padding: "8px 18px", background: "#21262d", color: "#e6edf3",
  border: "1px solid #30363d", borderRadius: 6,
  fontFamily: "'DM Sans', sans-serif", fontSize: 13, fontWeight: 500, cursor: "pointer",
};
const sliderLabel = { fontSize: 10, color: "#6b7b8d", fontFamily: "monospace", letterSpacing: 0.5 };
const sliderVal = { fontFamily: "'JetBrains Mono', monospace", fontWeight: 700, fontSize: 14, color: "#bc8cff", minWidth: 30, textAlign: "center" };
