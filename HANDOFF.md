# LoRaVis — Image Compression for LoRaWAN

## Project Summary

We're building a compression/approximation library to send images over extremely bandwidth-constrained channels like LoRaWAN (50–250 byte payloads). The core idea is to explore multiple strategies that trade pixel-fidelity for extreme compression ratios (often 500:1 to 5000:1).

## What We've Built So Far

Two interactive React prototypes exploring different compression strategies. Both are fully functional with real encode → decode round-trips, PSNR quality metrics, byte budget tracking, and pipeline visualizations.

### Prototype 1: Multi-Strategy Comparison (`loravis.jsx`)

Compares 5 classical compression approaches side-by-side:

1. **Micro Thumbnail** — Aggressive downsampling + k-means 8-color palette, 3-bit packed indices
2. **Dithered 4-Color** — Floyd-Steinberg dithering with 4-color k-means palette at 2bpp
3. **Block Average** — 4×4 blocks averaged to RGB555 (2 bytes/block)
4. **Edge Sketch** — Sobel edge detection → binary edge map → RLE encoding
5. **Vector Primitives** — Greedy circle placement iteratively minimizing MSE (painterly results)

Each shows actual byte count, budget %, PSNR, hex dump of encoded packet.

### Prototype 2: Advanced Deep-Dive (`loravis-advanced.jsx`)

Focuses on the two most promising approaches with full pipeline visualization:

**A) Hybrid Layered Encoding** — 3 stacked layers:
- Layer 1: 4×4 spatial color map with bilinear interpolation (48B)
- Layer 2: Sobel edge map, RLE compressed (variable)
- Layer 3: Quantized block residual correction (budget permitting)
- Gracefully degrades — drops layers when over budget

**B) VQ Patch Codebook** (VQ-VAE core mechanism):
- Image split into NxN patches, k-means finds K representative patches (codebook)
- Each patch replaced by nearest codebook entry index
- Interactive controls: K (4-64), patch size (2×2, 4×4, 8×8)
- Shows codebook tiles, frequency histogram, color-coded index map
- Key insight: with pre-shared codebook, only transmit indices (10-50B per frame)

## Agreed Next Steps

The user expressed interest in building these next:

1. **Python library** with actual LoRaWAN packet serialization — proper binary encoding/decoding, packet framing, maybe integration with LoRaWAN payload formatting
2. **Pre-shared codebook system** — train VQ-VAE offline on a domain (e.g., security cams, agriculture), then only transmit indices at runtime
3. **Temporal diff mode** — for periodic imaging (security cams), send full frame rarely + tiny diffs
4. **Adaptive mode selection** — analyze image and auto-pick best strategy

## Architecture Vision

```
┌─────────────┐         LoRaWAN (50-250 bytes)        ┌──────────────┐
│   ENCODER   │ ──────────────────────────────────────▶│   DECODER    │
│             │                                        │              │
│ - Downscale │   Packet: [mode][payload...]           │ - VQ-VAE dec │
│ - VQ-VAE enc│   mode 0: raw thumbnail                │ - SD/FLUX    │
│ - Captioner │   mode 1: vector primitives            │ - Procedural │
│ - Edge det  │   mode 2: text prompt (tokenized)      │ - Colorizer  │
│ - Diff calc │   mode 3: latent vector                │ - Upscaler   │
│             │   mode 4: hybrid (prompt + correction)  │              │
└─────────────┘                                        └──────────────┘
```

A mode byte at the start lets you pick the best strategy per image.

## Design Space (from brainstorm)

Approaches ranked from least to most compression:

1. **Classical compression** — downsample + tiny palette + LZ4/Huffman (200-500B)
2. **Vector/SVG encoding** — shapes as compact binary DSL (4-8B per primitive)
3. **Parametric scene description** — pre-agreed vocabulary of objects + positions (5-10B per object)
4. **Text-to-image prompt** — send compressed text, receiver runs Stable Diffusion/FLUX
5. **Latent space (VQ-VAE)** — learned encoder/decoder, quantized latent = the packet
6. **Hybrid layered** — text prompt for semantic baseline + latent correction vector
7. **Domain codebooks** — reference image index + lightweight diff
8. **Progressive refinement** — 4×4 first, then 8×8, then edges
9. **Temporal compression** — full frame rare, motion vectors + diffs rest of time

## Key Constraints

- LoRaWAN payloads: typically 50–250 bytes (max ~250 depending on data rate / region)
- Encoder may run on constrained hardware (ESP32, Raspberry Pi)
- Decoder can be more powerful (server-side)
- Latency tolerance is high (not real-time)
- Use cases: agricultural monitoring, wildlife cameras, remote sensor imagery, security cams

## Files

- `loravis.jsx` — Prototype 1 (React, multi-strategy comparison)
- `loravis-advanced.jsx` — Prototype 2 (React, hybrid + VQ-VAE deep-dive)
