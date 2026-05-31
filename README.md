# MRI-Segmentation
Automated detection &amp; Genant-scale classification of lumbar vertebral compression deformities on MRI · Mask R-CNN · PyTorch · OpenCV
# 🦴 Lumbar Vertebral Deformity Detection on MRI

> Automated end-to-end pipeline for detecting and classifying compression deformities  
> of lumbar vertebral bodies (L1–L5) on T2-weighted MRI using Mask R-CNN

---

## Overview

Compression fractures of vertebral bodies affect **~34% of women and ~27% of men** over 50,
yet up to **84% are missed** on initial radiograph review due to time pressure and inter-rater
variability. This project implements an automated diagnostic assistant that segments vertebral
bodies, measures their geometry, and classifies deformities per the standard **Genant
semiquantitative scale** — the same scale referenced in Russian Ministry of Health clinical
guidelines for osteoporosis.

The system covers the full pipeline:

1. **Instance segmentation** — Mask R-CNN isolates each vertebral body as a separate binary mask
2. **Morphometric analysis** — 6 anatomical keypoints are located per vertebra; anterior (Ah), middle (Mh), and posterior (Ph) heights are computed
3. **Genant classification** — each vertebra is assigned Grade 0–3 and deformity type (wedge / biconcave / compression)

---

## Results

| Metric | Value |
|--------|-------|
| Best validation loss (epoch 28 / 30) | **0.1984** |
| Detection confidence | **0.99 – 1.00** |
| Vertebrae detected per scan | **5 – 7** (L1–L5 ± Th12 / S1) |
| False positives @ conf. threshold 0.5 | **0** |
| Inference time (CPU, Apple M4) | **389 ± 1.6 ms** |

---

## Dataset

| Parameter | Value |
|-----------|-------|
| Patients | 49 |
| Scanner | Siemens Magnetom Verio Tim, **3T**, T2-TSE |
| Annotated vertebral bodies | **520** |
| Annotation tool | 3D Slicer |
| Train / val split | 39 / 10 patients (patient-level) |

Data collected at BARSMED Medical Centre, Kazan, Russia.  
Sequence parameters: TR 1660–5270 ms · TE 77–112 ms · slice thickness 4 mm · pixel size 0.58–0.93 mm.

---

## Architecture & Key Design Decisions

```
Input (JPG 512×512)
        │
        ▼
Mask R-CNN  ──  ResNet-50 + FPN  (COCO pretrained, fine-tuned)
        │
        ▼
Post-processing
  • Morphological open → close
  • Largest connected component  ◄─── original modification #3
        │
        ▼
Orientation  ──  cv2.minAreaRect  ◄──── original modification #1 (replaces PCA)
        │
        ▼
Keypoint detection
  • Gaussian-smoothed column histograms  ◄── original modification #2
  • 6 anatomical points (Ah, Mh, Ph per half)
        │
        ▼
Genant classification  ──  Grade 0 / 1 / 2 / 3  +  deformity type
        │
        ▼
Annotated output report (JPG + text)
```

**Three original contributions** vs. the reference method (Al-Haidri et al., 2025):

| # | Original approach | This work |
|---|---|---|
| 1 | PCA for mask orientation | `cv2.minAreaRect` — more stable on small/cropped masks |
| 2 | Raw histogram extrema search | Gaussian smoothing (σ = 3% array length) — eliminates pixel-noise artifacts |
| 3 | Standard morphological post-processing | + largest connected component filtering — removes neighbour fragments |

---

## Tech Stack

```
Python       3.13
PyTorch      2.11
torchvision  0.26   (Mask R-CNN implementation)
OpenCV             (image processing, minAreaRect)
NumPy / SciPy      (morphometry, Gaussian filtering)
3D Slicer          (dataset annotation, .seg.nrrd format)
```

---

## Project Structure

```
├── pipeline_seg.py        # Dataset preparation & NRRD → mask conversion
├── train_maskrcnn.py      # Model training loop
├── deformity_analysis.py  # Post-processing, keypoint detection, Genant classification
└── main.py                # Inference pipeline & report generation
```

---

## Training Details

| Parameter | Value |
|-----------|-------|
| Architecture | Mask R-CNN, ResNet-50 + FPN |
| Pre-training | COCO (transfer learning) |
| Input size | 512 × 512 |
| Batch size | 2 |
| Optimiser | SGD (momentum 0.9, weight decay 5e-4) |
| LR schedule | StepLR (step = 20, γ = 0.5), initial LR = 5e-3 |
| Loss | L = L_cls + L_box + L_mask |
| Epochs | 30 (early stopping, patience = 10) |
| Best checkpoint | Epoch 28, val loss = **0.1984** |
| Hardware | Apple M4 CPU (no GPU) |

---

## References

- Al-Haidri W. et al. *Automated diagnosis of lumbar vertebral body deformities on MRI using deep learning* — Biomedical Signal Processing and Control, 2025
- Genant H.K. et al. *Vertebral fracture assessment using a semiquantitative technique* — JBMR, 1993
- He K. et al. *Mask R-CNN* — ICCV, 2017

---

*Coursework project, Kazan Federal University, Institute of Physics, 2026.*  
*Supervised by Prof. Ilyasov K.A. (D.Sc.)*
