# Keras 3 & TensorFlow Computation Graph Architecture
**Comprehensive Architectural Guide & Execution Foundations**

---

## 📌 Executive Architecture & Visual Flowchart

```
                 TF.DATA HIGH-THROUGHPUT PIPELINE
    Disk (Files) ──► Interleave ──► Prefetch ──► Map (Augment) ──► Batch ──► GPU Memory
                           │
                 Runs in parallel C++ threads!
                 GPU never starves for data!
```

---

## 🧭 Deep Theoretical Foundations

### 1. TensorFlow API Tiers
- **Sequential API:** Linear stack of layers for simple feedforward pipelines.
- **Functional API:** Directed Acyclic Graph (DAG) supporting multi-input, multi-output, and residual skip connections.
- **Model Subclassing:** Imperative dynamic execution via `call(inputs, training=False)` allowing dynamic loops and custom attention.

### 2. High-Performance `tf.data` Pipeline
To achieve 100% GPU saturation, `tf.data` uses:
- `.cache()`: Keeps decoded records in RAM.
- `.map(num_parallel_calls=tf.data.AUTOTUNE)`: Parallel CPU preprocessing.
- `.prefetch(buffer_size=tf.data.AUTOTUNE)`: Pre-stages next training batch in GPU VRAM during backward pass execution.
