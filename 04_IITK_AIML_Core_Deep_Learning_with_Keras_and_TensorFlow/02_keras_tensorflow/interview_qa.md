# Interview Q&A — Keras & TensorFlow 2 Architecture

> **30 High-Yield Questions & Model Answers** for AI/ML and Data Science Technical Interviews.

### Q1. Compare Keras Sequential API, Functional API, and Model Subclassing.

**Answer:** Sequential API: linear stack of layers with single input and single output; simple but cannot handle shared layers, multi-inputs, or residual skips. Functional API: treats layers as callable functions on tensors ('y = Dense()(x)'); supports arbitrary DAGs, multiple inputs/outputs, and skip connections. Model Subclassing: inherit from 'tf.keras.Model' and implement '__init__' and 'call()'; provides complete dynamic imperative control in Python, but loses serialization and static graph inspection.

### Q2. Explain Eager Execution vs Graph Execution ('@tf.function') in TensorFlow 2.

**Answer:** Eager execution evaluates operations imperatively and immediately in Python, enabling intuitive debugging and line-by-line inspection. Graph execution ('@tf.function') parses Python code into an optimized, language-independent computational C++ graph (AutoGraph), performing constant folding, subexpression elimination, and parallel kernel execution across GPUs.

### Q3. How does 'tf.data.Dataset' optimize data pipeline throughput?

**Answer:** It constructs asynchronous, multi-threaded ETL pipelines. Key optimizations: '.prefetch(tf.data.AUTOTUNE)' overlaps GPU training with CPU data loading; '.interleave()' parallelizes disk reads across files; '.map(num_parallel_calls=AUTOTUNE)' parallelizes batch preprocessing; '.cache()' caches processed data in memory.

### Q4. Explain the Adam optimizer and why it combines AdaGrad and RMSprop.

**Answer:** Adam (Adaptive Moment Estimation) maintains exponentially decaying averages of past gradients (first moment m_t, momentum) and past squared gradients (second moment v_t, RMSprop adaptive learning rate): m_t = β1 m_{t-1} + (1 - β1) g_t; v_t = β2 v_{t-1} + (1 - β2) g_t². It applies bias corrections m̂_t and v̂_t to account for initialization at zero, updating weights via W = W - η * m̂_t / (sqrt(v̂_t) + ε).

### Q5. What is AdamW and why is it preferred over Adam with L2 regularization?

**Answer:** In Adam, standard L2 regularization adds λW to the gradient, which gets scaled by the adaptive second moment sqrt(v_t), effectively weakening the regularization penalty on weights with large historical gradients. AdamW decouples weight decay from the gradient update, subtracting λW directly from the weight after the adaptive step, restoring true weight decay generalization.

### Q6. Explain the role of Keras Callbacks and name four essential built-in callbacks.

**Answer:** Callbacks inject custom logic at specific stages of training (epoch start/end, batch start/end). Key callbacks: (1) 'EarlyStopping' halts training when validation loss stops improving, preventing overfitting. (2) 'ModelCheckpoint' saves model weights periodically when validation metric hits a new record. (3) 'ReduceLROnPlateau' halves learning rate when plateauing. (4) 'TensorBoard' logs metrics and histograms for live visualization.

### Q7. How does 'tf.GradientTape' compute custom gradients in custom training loops?

**Answer:** 'with tf.GradientTape() as tape:' records forward operations executed on watched tensors in memory. Calling 'grads = tape.gradient(loss, model.trainable_variables)' uses reverse-mode automatic differentiation to compute gradients, which are then applied to weights using 'optimizer.apply_gradients(zip(grads, vars))'.

### Q8. What is the difference between 'model.compile()' parameters: optimizer, loss, and metrics?

**Answer:** Optimizer specifies the gradient update algorithm (e.g. Adam). Loss is the scalar objective function minimized during backpropagation. Metrics (e.g. accuracy, AUC) are tracked and displayed for human evaluation but do NOT directly contribute to backpropagation gradients.

### Q9. How do you save and serialize Keras models (.keras vs SavedModel format)?

**Answer:** The native '.keras' format is a zip archive containing model architecture (JSON), weights (HDF5/binary), and training/optimizer configuration, standard in Keras 3. 'SavedModel' is TensorFlow's native directory format containing computational graphs and variables, standard for serving via TensorFlow Serving and TFLite conversion.

### Q10. Explain Mixed Precision Training ('tf.keras.mixed_precision').

**Answer:** Mixed precision executes forward and backward tensor operations in 16-bit floating point (FP16 or BF16) while storing master weights and loss scaling in 32-bit (FP32). This doubles GPU tensor core throughput, halves VRAM consumption, and prevents gradient underflow via dynamic loss scaling.

### Q11. What is Distributed Training in TensorFlow and how does 'tf.distribute.MirroredStrategy' work?

**Answer:** MirroredStrategy performs synchronous data parallelism across multiple GPUs on a single machine: it replicates the model on each GPU, splits each mini-batch across GPUs, executes forward and backward passes independently, and aggregates gradients across GPUs using hardware AllReduce (NVLink/NCCL) before updating master weights synchronously.

### Q12. What is MultiWorkerMirroredStrategy and parameter server strategy?

**Answer:** MultiWorkerMirroredStrategy extends synchronous data-parallel mirrored training across multiple separate physical machines connected via network. Parameter Server strategy delegates asynchronous parameter storage and updates to dedicated CPU parameter servers while worker machines compute gradients.

### Q13. How do custom Keras Layers and custom Models differ in implementation?

**Answer:** Inherit from 'tf.keras.layers.Layer' to create reusable computation blocks (implement '__init__', 'build(input_shape)' to define weights, and 'call(inputs)'). Inherit from 'tf.keras.Model' when the component represents the complete network container requiring training loop orchestration ('fit()', 'evaluate()', 'save()').

### Q14. What is Custom Loss Function in Keras and how do you write Huber Loss?

**Answer:** A custom loss can be a Python function 'def custom_loss(y_true, y_pred): return tf.reduce_mean(...)' or subclass 'tf.keras.losses.Loss'. Huber loss transitions smoothly from quadratic MSE for small errors (|e| <= δ) to linear MAE for large errors (|e| > δ), providing robustness against extreme outliers.

### Q15. Explain the difference between Huber loss and Mean Squared Error.

**Answer:** MSE penalizes errors quadratically ((y - ŷ)²), which forces models to bend excessively to accommodate rare extreme outliers. Huber loss penalizes large errors linearly (δ * |e| - 0.5 * δ²), bounding outlier influence while retaining smooth MSE differentiability near zero.

### Q16. What are Keras Functional API shared layers and multi-input architectures?

**Answer:** The Functional API allows a single instantiated layer to be called on multiple distinct tensors, sharing its exact weights across different pathways (e.g. Siamese networks). Multi-input models accept disparate data types (e.g. tabular features and image inputs) and concatenate intermediate embeddings before prediction.

### Q17. How does 'tf.RaggedTensor' handle variable-length sequential inputs?

**Answer:** Standard tensors require rectangular dimensions, requiring padding with zeros for variable-length text or time-series. 'tf.RaggedTensor' represents nested arrays with non-uniform slice lengths without zero-padding overhead, saving memory and compute in NLP pipelines.

### Q18. What is the purpose of 'tf.stop_gradient()'?

**Answer:** 'tf.stop_gradient(t)' treats tensor t as a constant during backpropagation, preventing error gradients from flowing backward past that point in the computational graph (essential in target networks for DQN Reinforcement Learning and self-supervised SimSiam).

### Q19. Explain Learning Rate Schedulers in Keras ('tf.keras.callbacks.LearningRateScheduler').

**Answer:** It updates the learning rate at the start of each epoch according to a deterministic schedule function 'lr = schedule(epoch, lr)'. It differs from ReduceLROnPlateau because it updates based on epoch count rather than monitoring validation loss.

### Q20. How do you extract intermediate layer activations for feature visualization or Grad-CAM?

**Answer:** Build a multi-output functional model: 'extractor = tf.keras.Model(inputs=base_model.input, outputs=[base_model.get_layer('conv_target').output, base_model.output])'. Passing an image yields both the final classification and the intermediate spatial feature map tensor.

### Q21. What is Transfer Learning and Fine-Tuning in Keras?

**Answer:** Transfer learning initializes a network with weights pre-trained on a massive dataset (e.g. ImageNet), freezes all convolutional base layers ('layer.trainable = False'), and trains a newly added classification head. Fine-tuning subsequently unfreezes top convolutional layers and trains them with a very small learning rate (e.g. 1e-5).

### Q22. Why must you re-compile a Keras model after toggling 'layer.trainable = True'?

**Answer:** 'layer.trainable' changes which weight tensors are included in 'model.trainable_variables'. Calling 'model.compile()' is mandatory to rebuild the underlying computational graph and update the optimizer's parameter list.

### Q23. What is the difference between 'training=True' and 'training=False' in 'model(inputs, training=...)'?

**Answer:** Certain layers behave differently during training vs inference. During training ('training=True'), Dropout randomly zeroes neuron activations, and Batch Normalization computes batch statistics. During inference ('training=False'), Dropout is disabled, and Batch Normalization uses frozen running mean and variance.

### Q24. How does TensorFlow Profiler identify GPU bottlenecks?

**Answer:** TensorFlow Profiler captures execution traces on CPUs and GPUs, identifying whether the GPU is underutilized due to CPU data loading bottlenecks ('InputPipelineAnalyzer'), host-device memory copying latency, or sub-optimal kernel memory bandwidth.

### Q25. What is TensorFlow Lite (TFLite) and Quantization?

**Answer:** TFLite converts trained TensorFlow models into lightweight, compact flatbuffer formats (.tflite) for edge and mobile devices. Quantization converts 32-bit floating point weights (FP32) to 8-bit integers (INT8), shrinking model size by 75% and accelerating inference on mobile NPUs with negligible accuracy loss.

### Q26. What is TensorFlow Serving and how does it achieve high-throughput microservices?

**Answer:** TensorFlow Serving is a high-performance C++ server designed for production ML environments. It exposes gRPC and REST endpoints, loads new model versions dynamically without downtime, supports batching requests across concurrent clients, and executes directly on GPUs.

### Q27. How do you implement custom training metrics in Keras?

**Answer:** Subclass 'tf.keras.metrics.Metric' and implement: '__init__' (allocate state variables via add_weight), 'update_state(y_true, y_pred)' (accumulate metric values over batches), 'result()' (compute final metric), and 'reset_state()' (clear accumulators at epoch boundaries).

### Q28. What is Keras Tuner and what search strategies does it support?

**Answer:** Keras Tuner automates hyperparameter optimization for deep networks. It supports: Random Search, Grid Search, Hyperband (uses early stopping on poor trials to evaluate more configurations), and Bayesian Optimization (Gaussian processes).

### Q29. Explain the difference between 'tf.constant' and 'tf.Variable'.

**Answer:** 'tf.constant' creates an immutable tensor whose values cannot be altered. 'tf.Variable' creates a mutable tensor backed by a persistent memory buffer whose values are modified in-place via '.assign()' and '.assign_add()', which is how trainable weights are stored.

### Q30. What is Keras 3 and how does it achieve multi-backend support?

**Answer:** Keras 3 is a complete ground-up rewrite that decouples the Keras API from TensorFlow, allowing models to run interchangeably on top of TensorFlow, PyTorch, or JAX backends with identical Python code, enabling dynamic framework switching.
