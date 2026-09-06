# Probability Theory & Parametric Distributions
**Comprehensive Architectural Guide & Execution Foundations**

---

## 📌 Executive Architecture & Visual Flowchart

```
                    BAYESIAN INFERENCE PIPELINE
            Prior Knowledge P(θ)  ×  Observed Likelihood P(D|θ)
    ──────────────────────────────────────────────────────────────────
                        Marginal Evidence P(D)
                                  │
                                  ▼
                    Posterior Distribution P(θ|D)
```

---

## 🧭 Deep Theoretical Foundations

### 1. Bayes' Theorem & Conditional Probability
$$P(A | B) = rac{P(B | A) \cdot P(A)}{P(B)}$$
Forms the probabilistic engine for Naive Bayes classifiers, Bayesian optimization, and Kalman filters.

### 2. Core Distribution Taxonomies
- **Binomial Distribution:** $P(X = k) = inom{n}{k} p^k (1-p)^{n-k}$ (Discrete binary success rate).
- **Poisson Distribution:** $P(X = k) = rac{\lambda^k e^{-\lambda}}{k!}$ (Arrival rate in fixed intervals).
- **Normal (Gaussian) Distribution:** $f(x) = rac{1}{\sigma \sqrt{2\pi}} e^{-rac{1}{2}\left(rac{x-\mu}{\sigma}ight)^2}$.
