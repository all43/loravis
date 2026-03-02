import { useState, useRef, useCallback, useEffect } from "react";

// ── Utility helpers ──────────────────────────────────────────────
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

function getPixels(canvas) {
  const ctx = canvas.getContext("2d");
  return ctx.getImageData(0, 0, canvas.width, canvas.height);
}

function makeCanvas(w, h) {
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  return c;
}

function drawImageToCanvas(img, w, h) {
  const c = makeCanvas(w, h);
  c.getContext("2d").drawImage(img, 0, 0, w, h);
  return c;
}

function canvasFromPixels(data, w, h) {
  const c = makeCanvas(w, h);
  const ctx = c.getContext("2d");
  const id = ctx.createImageData(w, h);
  id.data.set(data);
  ctx.putImageData(id, 0, 0);
  return c;
}

function psnr(orig, recon, w, h) {
  let mse = 0;
  for (let i = 0; i < w * h * 4; i += 4) {
    for (let c = 0; c < 3; c++) {
      const d = orig[i + c] - recon[i + c];
      mse += d * d;
    }
  }
  mse /= (w * h * 3);
  if (mse === 0) return 99;
  return 10 * Math.log10(255 * 255 / mse);
}

// Simple RLE on byte array → returns packed length
function rleSize(bytes) {
  if (bytes.length === 0) return 0;
  let size = 0, run = 1;
  for (let i = 1; i < bytes.length; i++) {
    if (bytes[i] === bytes[i - 1] && run < 255) { run++; }
    else { size += 2; run = 1; }
  }
  size += 2;
  return size;
}

// Entropy-based size estimate (bits)
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

function kMeansPalette(pixels, k, maxIter = 10) {
  // Initialize centroids from evenly spaced pixels
  const n = pixels.length / 4;
  const centroids = [];
  for (let i = 0; i < k; i++) {
    const idx = Math.floor((i / k) * n) * 4;
    centroids.push([pixels[idx], pixels[idx+1], pixels[idx+2]]);
  }
  let assignments = new Uint8Array(n);
  for (let iter = 0; iter < maxIter; iter++) {
    // Assign
    for (let i = 0; i < n; i++) {
      const r = pixels[i*4], g = pixels[i*4+1], b = pixels[i*4+2];
      let best = 0, bestD = Infinity;
      for (let c = 0; c < k; c++) {
        const dr = r - centroids[c][0], dg = g - centroids[c][1], db = b - centroids[c][2];
        const d = dr*dr + dg*dg + db*db;
        if (d < best || d < bestD) { bestD = d; best = c; }
      }
      assignments[i] = best;
    }
    // Update
    const sums = Array.from({length: k}, () => [0,0,0,0]);
    for (let i = 0; i < n; i++) {
      const c = assignments[i];
      sums[c][0] += pixels[i*4];
      sums[c][1] += pixels[i*4+1];
      sums[c][2] += pixels[i*4+2];
      sums[c][3]++;
    }
    for (let c = 0; c < k; c++) {
      if (sums[c][3] > 0) {
        centroids[c] = [sums[c][0]/sums[c][3], sums[c][1]/sums[c][3], sums[c][2]/sums[c][3]];
      }
    }
  }
  return { centroids, assignments };
}

// ── STRATEGY 1: Micro Thumbnail ──────────────────────────────────
function encodeMicroThumb(srcCanvas, budget) {
  // Determine size that fits in budget: palette (k*3) + indices (w*h * bpp/8)
  const k = 8; // 8 colors = 3 bits per pixel
  const bpp = 3;
  const headerBytes = k * 3 + 2; // palette + width/height bytes
  const availBits = (budget - headerBytes) * 8;
  const maxPixels = Math.floor(availBits / bpp);
  const side = Math.min(Math.floor(Math.sqrt(maxPixels)), 32);
  
  const thumb = drawImageToCanvas(srcCanvas, side, side);
  const px = getPixels(thumb).data;
  const { centroids, assignments } = kMeansPalette(px, k);
  
  // Build encoded packet
  const packet = [side, side]; // w, h
  for (const c of centroids) packet.push(Math.round(c[0]), Math.round(c[1]), Math.round(c[2]));
  // Pack indices: 3 bits each
  let buf = 0, bits = 0;
  for (let i = 0; i < assignments.length; i++) {
    buf = (buf << bpp) | assignments[i];
    bits += bpp;
    if (bits >= 8) { packet.push((buf >> (bits - 8)) & 0xFF); bits -= 8; }
  }
  if (bits > 0) packet.push((buf << (8 - bits)) & 0xFF);

  // Decode
  const decoded = new Uint8ClampedArray(side * side * 4);
  for (let i = 0; i < assignments.length; i++) {
    const c = centroids[assignments[i]];
    decoded[i*4] = Math.round(c[0]);
    decoded[i*4+1] = Math.round(c[1]);
    decoded[i*4+2] = Math.round(c[2]);
    decoded[i*4+3] = 255;
  }
  
  return {
    name: "Micro Thumbnail",
    desc: `${side}×${side}, ${k}-color palette`,
    bytes: packet.length,
    recon: canvasFromPixels(decoded, side, side),
    reconW: side, reconH: side,
    packet
  };
}

// ── STRATEGY 2: Edge Sketch ──────────────────────────────────────
function encodeEdgeSketch(srcCanvas, budget) {
  const side = Math.min(64, srcCanvas.width);
  const ratio = srcCanvas.height / srcCanvas.width;
  const h = Math.round(side * ratio);
  const thumb = drawImageToCanvas(srcCanvas, side, h);
  const px = getPixels(thumb).data;
  
  // Grayscale
  const gray = new Float32Array(side * h);
  for (let i = 0; i < side * h; i++) {
    gray[i] = 0.299 * px[i*4] + 0.587 * px[i*4+1] + 0.114 * px[i*4+2];
  }
  
  // Sobel edge detection
  const edges = new Uint8Array(side * h);
  const threshold = 40;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < side - 1; x++) {
      const gx = -gray[(y-1)*side+(x-1)] + gray[(y-1)*side+(x+1)]
                -2*gray[y*side+(x-1)] + 2*gray[y*side+(x+1)]
                -gray[(y+1)*side+(x-1)] + gray[(y+1)*side+(x+1)];
      const gy = -gray[(y-1)*side+(x-1)] - 2*gray[(y-1)*side+x] - gray[(y-1)*side+(x+1)]
                +gray[(y+1)*side+(x-1)] + 2*gray[(y+1)*side+x] + gray[(y+1)*side+(x+1)];
      const mag = Math.sqrt(gx*gx + gy*gy);
      edges[y * side + x] = mag > threshold ? 1 : 0;
    }
  }
  
  // RLE encode the binary edge map
  const packet = [side, h];
  let run = 0, current = 0;
  for (let i = 0; i < edges.length; i++) {
    if (edges[i] === current && run < 255) {
      run++;
    } else {
      packet.push(run);
      run = 1;
      current = edges[i];
    }
  }
  packet.push(run);
  
  // Also compute average color for tinting
  let avgR = 0, avgG = 0, avgB = 0;
  for (let i = 0; i < px.length; i += 4) {
    avgR += px[i]; avgG += px[i+1]; avgB += px[i+2];
  }
  const np = px.length / 4;
  avgR = Math.round(avgR/np); avgG = Math.round(avgG/np); avgB = Math.round(avgB/np);
  packet.push(avgR, avgG, avgB);
  
  // Decode
  const decoded = new Uint8ClampedArray(side * h * 4);
  for (let i = 0; i < edges.length; i++) {
    if (edges[i]) {
      decoded[i*4] = 255; decoded[i*4+1] = 255; decoded[i*4+2] = 255;
    } else {
      decoded[i*4] = Math.round(avgR * 0.15);
      decoded[i*4+1] = Math.round(avgG * 0.15);
      decoded[i*4+2] = Math.round(avgB * 0.15);
    }
    decoded[i*4+3] = 255;
  }
  
  return {
    name: "Edge Sketch",
    desc: `${side}×${h} Sobel edges, RLE compressed`,
    bytes: Math.min(packet.length, entropyBytes(new Uint8Array(packet))),
    recon: canvasFromPixels(decoded, side, h),
    reconW: side, reconH: h,
    packet
  };
}

// ── STRATEGY 3: Vector Primitives (greedy circles) ───────────────
function encodeVectorPrimitives(srcCanvas, budget) {
  const side = 64;
  const ratio = srcCanvas.height / srcCanvas.width;
  const h = Math.round(side * ratio);
  const thumb = drawImageToCanvas(srcCanvas, side, h);
  const px = getPixels(thumb).data;
  
  // Each primitive: x(6bit), y(6bit), r(4bit), R(5bit), G(5bit), B(5bit) = ~31 bits ≈ 4 bytes
  const bytesPerPrim = 5;
  const headerBytes = 3;
  const maxPrims = Math.floor((budget - headerBytes) / bytesPerPrim);
  const numPrims = Math.min(maxPrims, 80);
  
  // Target image as float
  const target = new Float32Array(side * h * 3);
  for (let i = 0; i < side * h; i++) {
    target[i*3] = px[i*4]; target[i*3+1] = px[i*4+1]; target[i*3+2] = px[i*4+2];
  }
  
  // Current reconstruction
  const current = new Float32Array(side * h * 3);
  // Start with average color
  let avgR = 0, avgG = 0, avgB = 0;
  for (let i = 0; i < side*h; i++) { avgR += target[i*3]; avgG += target[i*3+1]; avgB += target[i*3+2]; }
  avgR /= side*h; avgG /= side*h; avgB /= side*h;
  for (let i = 0; i < side*h; i++) { current[i*3] = avgR; current[i*3+1] = avgG; current[i*3+2] = avgB; }
  
  const primitives = [];
  
  for (let p = 0; p < numPrims; p++) {
    let bestScore = -Infinity, bestPrim = null;
    
    // Try random candidates
    const trials = 60;
    for (let t = 0; t < trials; t++) {
      const cx = Math.floor(Math.random() * side);
      const cy = Math.floor(Math.random() * h);
      const cr = Math.floor(Math.random() * 12) + 2;
      
      // Find average error color in this circle
      let sr = 0, sg = 0, sb = 0, count = 0;
      const x0 = Math.max(0, cx - cr), x1 = Math.min(side - 1, cx + cr);
      const y0 = Math.max(0, cy - cr), y1 = Math.min(h - 1, cy + cr);
      for (let y = y0; y <= y1; y++) {
        for (let x = x0; x <= x1; x++) {
          if ((x-cx)*(x-cx) + (y-cy)*(y-cy) <= cr*cr) {
            const idx = y * side + x;
            sr += target[idx*3] - current[idx*3];
            sg += target[idx*3+1] - current[idx*3+1];
            sb += target[idx*3+2] - current[idx*3+2];
            count++;
          }
        }
      }
      if (count === 0) continue;
      sr /= count; sg /= count; sb /= count;
      
      // Score: reduction in MSE
      let score = 0;
      for (let y = y0; y <= y1; y++) {
        for (let x = x0; x <= x1; x++) {
          if ((x-cx)*(x-cx) + (y-cy)*(y-cy) <= cr*cr) {
            const idx = y * side + x;
            for (let c = 0; c < 3; c++) {
              const diff = target[idx*3+c] - current[idx*3+c];
              const newDiff = diff - [sr,sg,sb][c];
              score += diff*diff - newDiff*newDiff;
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
    
    // Apply primitive
    const { cx, cy, cr, r, g, b } = bestPrim;
    const x0 = Math.max(0, cx - cr), x1 = Math.min(side - 1, cx + cr);
    const y0 = Math.max(0, cy - cr), y1 = Math.min(h - 1, cy + cr);
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        if ((x-cx)*(x-cx) + (y-cy)*(y-cy) <= cr*cr) {
          const idx = y * side + x;
          current[idx*3] = clamp(current[idx*3] + r, 0, 255);
          current[idx*3+1] = clamp(current[idx*3+1] + g, 0, 255);
          current[idx*3+2] = clamp(current[idx*3+2] + b, 0, 255);
        }
      }
    }
  }
  
  const decoded = new Uint8ClampedArray(side * h * 4);
  for (let i = 0; i < side * h; i++) {
    decoded[i*4] = clamp(Math.round(current[i*3]), 0, 255);
    decoded[i*4+1] = clamp(Math.round(current[i*3+1]), 0, 255);
    decoded[i*4+2] = clamp(Math.round(current[i*3+2]), 0, 255);
    decoded[i*4+3] = 255;
  }
  
  return {
    name: "Vector Primitives",
    desc: `${primitives.length} circles on ${side}×${h}`,
    bytes: headerBytes + primitives.length * bytesPerPrim,
    recon: canvasFromPixels(decoded, side, h),
    reconW: side, reconH: h,
    primitives
  };
}

// ── STRATEGY 4: Block DCT-ish (simplified) ───────────────────────
function encodeBlockAverage(srcCanvas, budget) {
  const blockSize = 4;
  const bw = Math.min(16, Math.floor(srcCanvas.width / blockSize));
  const bh = Math.min(16, Math.floor(srcCanvas.height / blockSize));
  const tw = bw * blockSize, th = bh * blockSize;
  const thumb = drawImageToCanvas(srcCanvas, tw, th);
  const px = getPixels(thumb).data;
  
  // For each block, store average color (quantized to 5-5-5 = 15 bits = 2 bytes)
  const packet = [bw, bh, blockSize];
  const blocks = [];
  
  for (let by = 0; by < bh; by++) {
    for (let bx = 0; bx < bw; bx++) {
      let r = 0, g = 0, b = 0;
      for (let dy = 0; dy < blockSize; dy++) {
        for (let dx = 0; dx < blockSize; dx++) {
          const idx = ((by * blockSize + dy) * tw + (bx * blockSize + dx)) * 4;
          r += px[idx]; g += px[idx+1]; b += px[idx+2];
        }
      }
      const n = blockSize * blockSize;
      const qr = Math.round(r / n / 8); // 5 bit
      const qg = Math.round(g / n / 8);
      const qb = Math.round(b / n / 8);
      blocks.push([qr, qg, qb]);
      packet.push((qr << 2) | (qg >> 3));
      packet.push(((qg & 7) << 5) | qb);
    }
  }
  
  // Decode
  const decoded = new Uint8ClampedArray(tw * th * 4);
  let bi = 0;
  for (let by = 0; by < bh; by++) {
    for (let bx = 0; bx < bw; bx++) {
      const [qr, qg, qb] = blocks[bi++];
      for (let dy = 0; dy < blockSize; dy++) {
        for (let dx = 0; dx < blockSize; dx++) {
          const idx = ((by * blockSize + dy) * tw + (bx * blockSize + dx)) * 4;
          decoded[idx] = qr * 8; decoded[idx+1] = qg * 8; decoded[idx+2] = qb * 8; decoded[idx+3] = 255;
        }
      }
    }
  }
  
  return {
    name: "Block Average",
    desc: `${bw}×${bh} blocks of ${blockSize}px, RGB555`,
    bytes: packet.length,
    recon: canvasFromPixels(decoded, tw, th),
    reconW: tw, reconH: th,
    packet
  };
}

// ── STRATEGY 5: Dithered Palette ─────────────────────────────────
function encodeDithered(srcCanvas, budget) {
  const k = 4;
  const side = Math.min(48, srcCanvas.width);
  const ratio = srcCanvas.height / srcCanvas.width;
  const h = Math.round(side * ratio);
  const thumb = drawImageToCanvas(srcCanvas, side, h);
  const px = new Uint8ClampedArray(getPixels(thumb).data);
  
  const { centroids } = kMeansPalette(px, k, 12);
  
  // Floyd-Steinberg dithering
  const work = new Float32Array(side * h * 3);
  for (let i = 0; i < side * h; i++) {
    work[i*3] = px[i*4]; work[i*3+1] = px[i*4+1]; work[i*3+2] = px[i*4+2];
  }
  
  const indices = new Uint8Array(side * h);
  
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < side; x++) {
      const i = y * side + x;
      const r = work[i*3], g = work[i*3+1], b = work[i*3+2];
      
      let best = 0, bestD = Infinity;
      for (let c = 0; c < k; c++) {
        const dr = r - centroids[c][0], dg = g - centroids[c][1], db = b - centroids[c][2];
        const d = dr*dr + dg*dg + db*db;
        if (d < bestD) { bestD = d; best = c; }
      }
      indices[i] = best;
      
      const er = r - centroids[best][0], eg = g - centroids[best][1], eb = b - centroids[best][2];
      const spread = [[x+1,y,7/16],[x-1,y+1,3/16],[x,y+1,5/16],[x+1,y+1,1/16]];
      for (const [nx, ny, w] of spread) {
        if (nx >= 0 && nx < side && ny < h) {
          const ni = ny * side + nx;
          work[ni*3] += er * w;
          work[ni*3+1] += eg * w;
          work[ni*3+2] += eb * w;
        }
      }
    }
  }
  
  // Encode: palette + packed 2-bit indices
  const packet = [side, h];
  for (const c of centroids) packet.push(Math.round(c[0]), Math.round(c[1]), Math.round(c[2]));
  
  for (let i = 0; i < indices.length; i += 4) {
    let byte = 0;
    for (let j = 0; j < 4 && i + j < indices.length; j++) {
      byte |= (indices[i+j] & 3) << (6 - j * 2);
    }
    packet.push(byte);
  }
  
  // Decode
  const decoded = new Uint8ClampedArray(side * h * 4);
  for (let i = 0; i < side * h; i++) {
    const c = centroids[indices[i]];
    decoded[i*4] = clamp(Math.round(c[0]), 0, 255);
    decoded[i*4+1] = clamp(Math.round(c[1]), 0, 255);
    decoded[i*4+2] = clamp(Math.round(c[2]), 0, 255);
    decoded[i*4+3] = 255;
  }
  
  return {
    name: "Dithered 4-Color",
    desc: `${side}×${h}, Floyd-Steinberg, 2bpp`,
    bytes: packet.length,
    recon: canvasFromPixels(decoded, side, h),
    reconW: side, reconH: h,
    packet
  };
}

// ── Generate a test image ────────────────────────────────────────
function generateTestImage() {
  const c = makeCanvas(128, 128);
  const ctx = c.getContext("2d");
  // Sky
  const sky = ctx.createLinearGradient(0, 0, 0, 70);
  sky.addColorStop(0, "#1a3a5c"); sky.addColorStop(1, "#5b8fb9");
  ctx.fillStyle = sky; ctx.fillRect(0, 0, 128, 70);
  // Ground
  const ground = ctx.createLinearGradient(0, 70, 0, 128);
  ground.addColorStop(0, "#4a7c3f"); ground.addColorStop(1, "#2d5a27");
  ctx.fillStyle = ground; ctx.fillRect(0, 70, 128, 58);
  // Sun
  ctx.fillStyle = "#ffd700"; ctx.beginPath(); ctx.arc(100, 25, 14, 0, Math.PI*2); ctx.fill();
  // House
  ctx.fillStyle = "#8b4513"; ctx.fillRect(30, 50, 35, 30);
  ctx.fillStyle = "#d2691e"; ctx.beginPath(); ctx.moveTo(25, 50); ctx.lineTo(47, 30); ctx.lineTo(70, 50); ctx.fill();
  ctx.fillStyle = "#4169e1"; ctx.fillRect(40, 60, 10, 12);
  // Tree
  ctx.fillStyle = "#2d5a27"; ctx.beginPath(); ctx.arc(95, 55, 12, 0, Math.PI*2); ctx.fill();
  ctx.fillStyle = "#5c3317"; ctx.fillRect(93, 65, 5, 15);
  // Cloud
  ctx.fillStyle = "rgba(255,255,255,0.7)";
  ctx.beginPath(); ctx.arc(40, 18, 8, 0, Math.PI*2); ctx.fill();
  ctx.beginPath(); ctx.arc(50, 15, 10, 0, Math.PI*2); ctx.fill();
  ctx.beginPath(); ctx.arc(60, 18, 7, 0, Math.PI*2); ctx.fill();
  return c;
}

// ── Hex dump display ─────────────────────────────────────────────
function HexDump({ data, maxBytes = 64 }) {
  const bytes = data.slice(0, maxBytes);
  const hex = bytes.map(b => b.toString(16).padStart(2, "0")).join(" ");
  return (
    <div style={{
      fontFamily: "'IBM Plex Mono', 'Courier New', monospace",
      fontSize: 9,
      color: "#7fcc7f",
      background: "#0a1a0a",
      padding: "6px 8px",
      borderRadius: 4,
      overflowX: "auto",
      whiteSpace: "pre-wrap",
      wordBreak: "break-all",
      lineHeight: 1.6,
      border: "1px solid #1a3a1a"
    }}>
      {hex}{data.length > maxBytes ? ` … +${data.length - maxBytes} more` : ""}
    </div>
  );
}

// ── Result card ──────────────────────────────────────────────────
function ResultCard({ result, originalCanvas, budget }) {
  const canvasRef = useRef(null);
  const [psnrVal, setPsnr] = useState(null);
  
  useEffect(() => {
    if (!canvasRef.current || !result) return;
    const ctx = canvasRef.current.getContext("2d");
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, 160, 160);
    const scale = Math.min(160 / result.reconW, 160 / result.reconH);
    const dw = result.reconW * scale, dh = result.reconH * scale;
    ctx.drawImage(result.recon, (160 - dw)/2, (160 - dh)/2, dw, dh);
    
    // Compute PSNR against original at same resolution
    if (originalCanvas) {
      const origSmall = drawImageToCanvas(originalCanvas, result.reconW, result.reconH);
      const origPx = getPixels(origSmall).data;
      const reconPx = getPixels(result.recon).data;
      setPsnr(psnr(origPx, reconPx, result.reconW, result.reconH).toFixed(1));
    }
  }, [result, originalCanvas]);
  
  if (!result) return null;
  
  const overBudget = result.bytes > budget;
  const pct = ((result.bytes / budget) * 100).toFixed(0);
  
  return (
    <div style={{
      background: "#0d1117",
      border: `1px solid ${overBudget ? "#f8514966" : "#30363d"}`,
      borderRadius: 8,
      padding: 16,
      display: "flex",
      flexDirection: "column",
      gap: 10,
      flex: "1 1 280px",
      minWidth: 250,
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
        <div>
          <div style={{ fontFamily: "'Space Mono', monospace", fontWeight: 700, fontSize: 14, color: "#e6edf3" }}>
            {result.name}
          </div>
          <div style={{ fontSize: 11, color: "#8b949e", marginTop: 2 }}>{result.desc}</div>
        </div>
        <div style={{ textAlign: "right" }}>
          <div style={{
            fontFamily: "'Space Mono', monospace",
            fontSize: 20,
            fontWeight: 700,
            color: overBudget ? "#f85149" : "#58a6ff",
          }}>
            {result.bytes}
            <span style={{ fontSize: 11, color: "#8b949e", fontWeight: 400 }}> B</span>
          </div>
          <div style={{ fontSize: 10, color: overBudget ? "#f85149" : "#8b949e" }}>
            {pct}% of budget
          </div>
        </div>
      </div>
      
      <div style={{
        background: "#161b22",
        borderRadius: 6,
        padding: 8,
        display: "flex",
        justifyContent: "center",
        border: "1px solid #21262d",
      }}>
        <canvas
          ref={canvasRef}
          width={160}
          height={160}
          style={{ imageRendering: "pixelated", width: 160, height: 160 }}
        />
      </div>
      
      {/* Meter bar */}
      <div style={{ position: "relative", height: 6, background: "#21262d", borderRadius: 3 }}>
        <div style={{
          position: "absolute", left: 0, top: 0, height: "100%",
          width: `${Math.min(100, parseFloat(pct))}%`,
          background: overBudget
            ? "linear-gradient(90deg, #f85149, #da3633)"
            : "linear-gradient(90deg, #238636, #2ea043)",
          borderRadius: 3,
          transition: "width 0.4s ease",
        }} />
      </div>
      
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "#8b949e" }}>
        <span>PSNR: {psnrVal ? `${psnrVal} dB` : "—"}</span>
        <span>{result.reconW}×{result.reconH}px</span>
      </div>
      
      {result.packet && <HexDump data={result.packet} />}
    </div>
  );
}

// ── Main App ─────────────────────────────────────────────────────
export default function App() {
  const [sourceCanvas, setSourceCanvas] = useState(null);
  const [results, setResults] = useState(null);
  const [budget, setBudget] = useState(200);
  const [processing, setProcessing] = useState(false);
  const [sourceDataUrl, setSourceDataUrl] = useState(null);
  const sourcePreviewRef = useRef(null);
  const fileInputRef = useRef(null);
  
  const loadImage = useCallback((src) => {
    if (src instanceof HTMLCanvasElement) {
      setSourceCanvas(src);
      setSourceDataUrl(src.toDataURL());
      return;
    }
    const img = new Image();
    img.onload = () => {
      const maxDim = 256;
      let w = img.width, h = img.height;
      if (w > maxDim || h > maxDim) {
        const scale = maxDim / Math.max(w, h);
        w = Math.round(w * scale); h = Math.round(h * scale);
      }
      const c = drawImageToCanvas(img, w, h);
      setSourceCanvas(c);
      setSourceDataUrl(c.toDataURL());
    };
    img.src = src;
  }, []);
  
  const handleFileUpload = useCallback((e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => loadImage(ev.target.result);
    reader.readAsDataURL(file);
  }, [loadImage]);
  
  const runCompression = useCallback(() => {
    if (!sourceCanvas) return;
    setProcessing(true);
    setTimeout(() => {
      const res = [
        encodeMicroThumb(sourceCanvas, budget),
        encodeDithered(sourceCanvas, budget),
        encodeBlockAverage(sourceCanvas, budget),
        encodeEdgeSketch(sourceCanvas, budget),
        encodeVectorPrimitives(sourceCanvas, budget),
      ];
      setResults(res);
      setProcessing(false);
    }, 50);
  }, [sourceCanvas, budget]);
  
  useEffect(() => {
    if (sourceCanvas) runCompression();
  }, [sourceCanvas, budget]);
  
  const loadTestImage = () => loadImage(generateTestImage());

  return (
    <div style={{
      minHeight: "100vh",
      background: "#010409",
      color: "#e6edf3",
      fontFamily: "'DM Sans', -apple-system, sans-serif",
      padding: "24px 20px",
    }}>
      <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet" />
      
      {/* Header */}
      <div style={{ maxWidth: 1100, margin: "0 auto" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 4 }}>
          <div style={{
            width: 10, height: 10, borderRadius: "50%",
            background: "#2ea043",
            boxShadow: "0 0 8px #2ea04366",
            animation: "pulse 2s ease-in-out infinite",
          }} />
          <span style={{ fontFamily: "'Space Mono', monospace", fontSize: 11, color: "#58a6ff", letterSpacing: 2, textTransform: "uppercase" }}>
            LoRaVis Compression Lab
          </span>
        </div>
        <h1 style={{
          fontFamily: "'Space Mono', monospace",
          fontSize: 28,
          fontWeight: 700,
          margin: "8px 0 6px",
          background: "linear-gradient(135deg, #58a6ff, #3fb950)",
          WebkitBackgroundClip: "text",
          WebkitTextFillColor: "transparent",
        }}>
          Image → LoRaWAN Packet
        </h1>
        <p style={{ color: "#8b949e", fontSize: 14, margin: "0 0 24px", maxWidth: 600, lineHeight: 1.5 }}>
          Compare compression strategies for transmitting images over extremely
          bandwidth-constrained channels. Upload an image or use the test scene.
        </p>
        
        {/* Controls */}
        <div style={{
          display: "flex",
          gap: 16,
          alignItems: "center",
          flexWrap: "wrap",
          marginBottom: 24,
          padding: "16px 20px",
          background: "#0d1117",
          borderRadius: 10,
          border: "1px solid #21262d",
        }}>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileUpload}
            style={{ display: "none" }}
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            style={{
              padding: "8px 18px",
              background: "#21262d",
              color: "#e6edf3",
              border: "1px solid #30363d",
              borderRadius: 6,
              fontFamily: "'DM Sans', sans-serif",
              fontSize: 13,
              fontWeight: 500,
              cursor: "pointer",
            }}
          >
            Upload Image
          </button>
          <button
            onClick={loadTestImage}
            style={{
              padding: "8px 18px",
              background: "transparent",
              color: "#58a6ff",
              border: "1px solid #58a6ff44",
              borderRadius: 6,
              fontFamily: "'DM Sans', sans-serif",
              fontSize: 13,
              fontWeight: 500,
              cursor: "pointer",
            }}
          >
            Use Test Scene
          </button>
          
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginLeft: "auto" }}>
            <label style={{ fontSize: 12, color: "#8b949e", fontFamily: "'Space Mono', monospace" }}>
              BUDGET
            </label>
            <input
              type="range"
              min={50}
              max={500}
              step={10}
              value={budget}
              onChange={(e) => setBudget(Number(e.target.value))}
              style={{ width: 130, accentColor: "#58a6ff" }}
            />
            <span style={{
              fontFamily: "'Space Mono', monospace",
              fontSize: 16,
              fontWeight: 700,
              color: "#58a6ff",
              minWidth: 65,
            }}>
              {budget} B
            </span>
          </div>
        </div>
        
        {/* Source preview + results */}
        {sourceCanvas && (
          <>
            <div style={{
              display: "flex",
              alignItems: "center",
              gap: 16,
              marginBottom: 20,
              padding: "12px 16px",
              background: "#0d1117",
              borderRadius: 8,
              border: "1px solid #21262d",
            }}>
              <div style={{
                width: 80, height: 80,
                borderRadius: 6,
                overflow: "hidden",
                border: "1px solid #30363d",
                flexShrink: 0,
              }}>
                <img
                  src={sourceDataUrl}
                  alt="Source"
                  style={{ width: 80, height: 80, objectFit: "cover", imageRendering: "pixelated" }}
                />
              </div>
              <div>
                <div style={{ fontSize: 13, fontWeight: 500 }}>Original Image</div>
                <div style={{ fontSize: 11, color: "#8b949e" }}>
                  {sourceCanvas.width}×{sourceCanvas.height}px — {(sourceCanvas.width * sourceCanvas.height * 3).toLocaleString()} bytes uncompressed
                </div>
                <div style={{ fontSize: 11, color: "#8b949e" }}>
                  Compression target: <span style={{ color: "#58a6ff", fontWeight: 700 }}>{budget} bytes</span> — that's a{" "}
                  <span style={{ color: "#f0883e" }}>
                    {((sourceCanvas.width * sourceCanvas.height * 3) / budget).toFixed(0)}:1
                  </span> ratio
                </div>
              </div>
              {processing && (
                <div style={{ marginLeft: "auto", color: "#58a6ff", fontSize: 12 }}>Processing…</div>
              )}
            </div>
            
            <div style={{
              display: "flex",
              flexWrap: "wrap",
              gap: 16,
            }}>
              {results?.map((r, i) => (
                <ResultCard key={i} result={r} originalCanvas={sourceCanvas} budget={budget} />
              ))}
            </div>

            <div style={{
              marginTop: 24,
              padding: "16px 20px",
              background: "#0d1117",
              borderRadius: 8,
              border: "1px solid #21262d",
              fontSize: 12,
              color: "#8b949e",
              lineHeight: 1.7,
            }}>
              <div style={{ fontFamily: "'Space Mono', monospace", fontWeight: 700, color: "#e6edf3", marginBottom: 8, fontSize: 12 }}>
                Strategy Notes
              </div>
              <div><span style={{color:"#58a6ff"}}>Micro Thumbnail</span> — Extreme downsampling with k-means palette. Best for "is there something there?" detection.</div>
              <div><span style={{color:"#58a6ff"}}>Dithered 4-Color</span> — Floyd-Steinberg dithering preserves perceived detail with very few colors. Good for natural scenes.</div>
              <div><span style={{color:"#58a6ff"}}>Block Average</span> — Each 4×4 block averaged to one RGB555 color. Simple, predictable, hardware-friendly.</div>
              <div><span style={{color:"#58a6ff"}}>Edge Sketch</span> — Sobel edge detection → binary edge map → RLE. Best for structural/shape information.</div>
              <div><span style={{color:"#58a6ff"}}>Vector Primitives</span> — Greedy circle placement that iteratively reduces MSE. Painterly results, good for smooth scenes.</div>
            </div>
          </>
        )}
        
        {!sourceCanvas && (
          <div style={{
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            height: 300,
            border: "2px dashed #21262d",
            borderRadius: 12,
            color: "#484f58",
            fontSize: 14,
          }}>
            Upload an image or click "Use Test Scene" to begin
          </div>
        )}
      </div>
      
      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.4; }
        }
        input[type="range"] {
          height: 4px;
        }
        button:hover {
          filter: brightness(1.2);
        }
        * { box-sizing: border-box; }
      `}</style>
    </div>
  );
}
