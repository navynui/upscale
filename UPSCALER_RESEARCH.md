# =============================================================================
# FILE: UPSCALER_RESEARCH.md
# DESCRIPTION: Candidate Models for Local Upscaling on Tesla P100 (16GB VRAM)
# ----------------------------------------------------------------------------
# This research was conducted to identify highly efficient image upscaling candidates that can run concurrently with an active Llama.cpp server on a Tesla P100 GPU.
# The primary constraint is minimizing VRAM footprint while maximizing quality/performance.
# =============================================================================

## 🎯 Critical Constraint Summary
*   **Hardware:** NVIDIA Tesla P100 (Pascal architecture, 16GB VRAM).
*   **Challenge:** Shared memory contention with an active Llama.cpp server. Upscalers must be lightweight and efficient.
*   **Focus:** Local execution and minimal resource overhead.

## ⭐ Top Candidate Models for P100 Inference (Local Run)

Based on the constraints, pure high-end models are out. We need efficiency. The best candidates are highly optimized diffusion or ESRGAN variants that support low-precision loading.

### 1. Real-ESRGAN / SRGAN Variants
*   **Name:** Real-ESRGAN (and its derivatives like GFPGAN for face recovery).
*   **Description:** These models are dedicated Super-Resolution networks, often built on deep convolutional neural network (CNN) architectures rather than the massive Transformer/Diffusion pipelines. They are inherently lighter and faster to run locally, making them excellent candidates for running alongside an LLM server without major VRAM conflict.
*   **Local Suitability:** Extremely high. They typically use fewer parameters than a large diffusion model, allowing them to occupy minimal memory footprint compared to models like Stable Diffusion XL or complex transformers.

### 2. SwinIR (Image Restoration)
*   **Name:** SwinIR / various Transformer-based image restoration models.
*   **Description:** These are more advanced network architectures that use a hybrid of CNNs and Self-Attention mechanisms, making them highly effective for tasks like de-blurring or super-resolution while maintaining semantic context better than classic GANs.
*   **Local Suitability:** High potential. While they are transformers (like the LLM), their specialized nature for image *restoration/upscaling* allows them to be fine-tuned and quantized specifically for image tasks, potentially achieving good efficiency when run in 4-bit or lower precision mode.

### 3. Lightweight Diffusion Models (e.g., Tiny Stable Diffusion variants)
*   **Name:** Smaller-scale diffusion models trained purely on upscaling/super-resolution datasets (or highly pruned versions of base SD).
*   **Description:** Instead of running a full generation pipeline, these specialized micro-models are focused only on the upsampling step. These can sometimes provide better perceptual quality than CNNs while still being small enough to fit alongside other processes.
*   **Local Suitability:** Medium to High. The main difficulty here is finding a pre-trained version that has been aggressively optimized (quantized/pruned) for P100, but the concept offers the highest potential image quality.

## 💡 Deployment Strategy (Action Plan)
Since we are prioritizing local execution on the P100:
1.  **Focus:** Start with **Real-ESRGAN**. It is the most straightforward and lowest risk candidate for fitting alongside the LLM server.
2.  **Verification:** We must use `nvidia-smi` to monitor VRAM usage during a test run of each model. The goal is to ensure that combined VRAM usage (LLaMA.cpp + Upscaler) remains below 16GB with adequate headroom.

# =============================================================================
# =============================================================================