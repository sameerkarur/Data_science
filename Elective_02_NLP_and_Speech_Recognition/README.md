# Elective 02: Natural Language Processing & Speech Recognition
**E&ICT Academy, IIT Kanpur — Specialization Syllabus & Engineering Guide**

---

## 📌 Domain Overview

Natural Language Processing (NLP) and Speech Recognition encompass computational linguistics, acoustic modeling, and modern deep autoregressive language modeling. This elective covers the transition from traditional sequence models to modern Self-Attention Transformers, Audio Spectrogram Encoders, and Large Multimodal Audio Models.

---

## 🧭 Specialization Architecture & Curriculum Roadmap

### 1. Vector Semantics & Sequence-to-Sequence Modeling
- **Distributed Representations:** Word2Vec (Skip-gram with Negative Sampling, CBOW), GloVe log-bilinear matrix factorization, and Subword Tokenization (BPE, WordPiece, SentencePiece).
- **Recurrent Topologies:** Gated Recurrent Units (GRU) and Long Short-Term Memory (LSTM) gating mechanisms resolving vanishing gradients in sequential temporal data.
- **Attention Pioneers:** Bahdanau additive attention and Luong multiplicative attention bridging encoder-decoder information bottlenecks.

### 2. The Transformer Architecture & Foundation Models
- **Scaled Dot-Product & Multi-Head Attention:** Mathematical formulation mapping Queries ($Q$), Keys ($K$), and Values ($V$) into parallel subspace projections:
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
- **Positional Encoding Systems:** Sinusoidal absolute positions, learnable 1D embeddings, and Rotary Position Embeddings (RoPE) for context extrapolation.
- **Architectural Paradigms:**
  - *Encoder-Only (BERT, RoBERTa):* Bidirectional contextual representations optimized via Masked Language Modeling (MLM).
  - *Decoder-Only (GPT series, LLaMA, Mistral):* Causal autoregressive language models scaling with next-token prediction.
  - *Encoder-Decoder (T5, BART):* Cross-attention sequence transduction for summarization and translation.

### 3. Speech Recognition (ASR) & Acoustic Modeling
- **Audio Signal Representation:** Short-Time Fourier Transform (STFT), Mel-Scale filterbanks, and Log-Mel Spectrogram extraction.
- **Connectionist Temporal Classification (CTC):** Alignment-free loss computing probability of label sequences across arbitrary frame intervals:
  $$\mathcal{L}_{\text{CTC}} = -\ln P(Y | X) = -\ln \sum_{\pi \in \mathcal{B}^{-1}(Y)} P(\pi | X)$$
- **Modern Speech Transformers:** OpenAI Whisper encoder-decoder architecture performing weakly-supervised multilingual ASR, voice activity detection, and language identification.

### 4. Text-to-Speech (TTS) & Voice Synthesis
- **Mel-Spectrogram Generation:** Tacotron 2 and FastSpeech 2 non-autoregressive acoustic decoders with duration and pitch predictors.
- **Neural Vocoders:** WaveGlow, HiFi-GAN, and Diffusion-based vocoders reconstructing time-domain audio waveforms from spectrogram representations.

---

## 📐 Evaluation Metrics & Benchmarks

### Word Error Rate (WER) & Character Error Rate (CER)
$$\text{WER} = \frac{S + D + I}{N} = \frac{\text{Substitutions} + \text{Deletions} + \text{Insertions}}{\text{Total Words in Reference}}$$

### BLEU & ROUGE Linguistic Scoring
- **BLEU-$N$:** Modified $N$-gram precision with brevity penalty for automated machine translation.
- **ROUGE-L:** Longest Common Subsequence (LCS) F1-score measuring recall for document summarization.

---

## 💻 Recommended Applied Projects & Research Benchmarks

1. **Enterprise Multilingual Speech-to-Text Pipeline:** Whisper fine-tuned with LoRA on domain-specific acoustic audio streams.
2. **Abstractive Financial Document Summarizer:** Fine-tuned Longformer / LED model parsing 30+ page regulatory filings.
3. **Conversational Voice AI Agent:** End-to-end streaming speech recognition, LLM reasoning, and HiFi-GAN low-latency audio response loop.
