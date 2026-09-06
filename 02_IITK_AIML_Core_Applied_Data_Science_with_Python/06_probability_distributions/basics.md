# Chapter 6: Probability Theory & Parametric Distributions
**Comprehensive Textbook Guide — Advanced Applied Data Science**

---

## 1. Executive Overview & Mental Models

Probability theory provides the mathematical calculus of uncertainty. In statistical modeling:
- A random variable maps physical event outcomes to real numbers: $X: \Omega \to \mathbb{R}$.
- Parametric distributions compress infinite empirical measurements into compact analytical forms defined by a few governing parameters ($\mu, \sigma, \lambda, p$).
- Bayesian inference continuously updates prior probability beliefs with newly observed evidence.

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

## 2. Core Distribution Taxonomies & Mathematical Properties

### 1. Discrete Parametric Distributions

| Distribution | Support ($k$) | Probability Mass Function (PMF) | Expected Value $\mathbb{E}[X]$ | Variance $\text{Var}(X)$ | Industrial AI Application |
|---|---|---|---|---|---|
| **Bernoulli** | $\{0, 1\}$ | $p^k (1 - p)^{1 - k}$ | $p$ | $p(1 - p)$ | Binary click-through prediction |
| **Binomial** | $\{0, \dots, n\}$ | $\binom{n}{k} p^k (1 - p)^{n - k}$ | $n p$ | $n p (1 - p)$ | Batch hardware defect counts |
| **Poisson** | $\{0, 1, 2, \dots\}$ | $\frac{\lambda^k e^{-\lambda}}{k!}$ | $\lambda$ | $\lambda$ | Website query arrival rate per sec |
| **Geometric** | $\{1, 2, \dots\}$ | $(1 - p)^{k - 1} p$ | $\frac{1}{p}$ | $\frac{1 - p}{p^2}$ | Trials until first successful API call |

### 2. Continuous Parametric Distributions

| Distribution | Support ($x$) | Probability Density Function (PDF) | Expected Value $\mathbb{E}[X]$ | Variance $\text{Var}(X)$ | Industrial AI Application |
|---|---|---|---|---|---|
| **Gaussian (Normal)** | $(-\infty, \infty)$ | $\frac{1}{\sigma \sqrt{2\pi}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$ | $\mu$ | $\sigma^2$ | Sensor noise, measurement errors |
| **Log-Normal** | $(0, \infty)$ | $\frac{1}{x \sigma \sqrt{2\pi}} \exp\left(-\frac{(\ln x - \mu)^2}{2\sigma^2}\right)$ | $\exp\left(\mu + \frac{\sigma^2}{2}\right)$ | $(\exp(\sigma^2) - 1)\mathbb{E}[X]^2$ | Financial wealth, web page dwell time |
| **Exponential** | $[0, \infty)$ | $\lambda e^{-\lambda x}$ | $\frac{1}{\lambda}$ | $\frac{1}{\lambda^2}$ | Time between server failure events |
| **Beta** | $[0, 1]$ | $\frac{x^{\alpha - 1} (1 - x)^{\beta - 1}}{\text{B}(\alpha, \beta)}$ | $\frac{\alpha}{\alpha + \beta}$ | $\frac{\alpha \beta}{(\alpha + \beta)^2 (\alpha + \beta + 1)}$ | Prior beliefs over conversion rates |

---

## 3. Deep Theoretical Foundations

### 1. The Principle of Maximum Likelihood Estimation (MLE)
Given an observed dataset $D = \{x_1, x_2, \dots, x_n\}$ assumed i.i.d. from parameterized distribution $f(x \mid \theta)$, the likelihood function is:
$$L(\theta) = \prod_{i=1}^n f(x_i \mid \theta)$$
Maximizing the log-likelihood avoids numerical underflow and converts products to sums:
$$\ell(\theta) = \ln L(\theta) = \sum_{i=1}^n \ln f(x_i \mid \theta)$$
Setting the gradient score vector to zero yields the MLE estimator:
$$\nabla_\theta \ell(\theta) = 0 \implies \hat{\theta}_{\text{MLE}}$$

### 2. Conjugate Priors & Analytical Bayesian Updating
In Bayesian statistics, if the posterior distribution $P(\theta \mid D)$ belongs to the same probability distribution family as the prior $P(\theta)$, the prior is termed **conjugate** to the likelihood.
- **Beta-Binomial Conjugacy:**
  - Prior: $\theta \sim \text{Beta}(\alpha, \beta)$
  - Likelihood: $k$ successes in $n$ trials $\sim \text{Binomial}(n, \theta)$
  - Analytical Posterior: $\theta \mid D \sim \text{Beta}(\alpha + k, \beta + (n - k))$
  This allows instant analytical real-time updates in Multi-Armed Bandits (Thompson Sampling) without expensive Markov Chain Monte Carlo (MCMC) simulations.

---

## 4. Production Implementation: Bayesian Conjugate Updating & Thompson Sampling

```python
import numpy as np

class BetaBinomialBandit:
    """Thompson Sampling multi-armed bandit using exact Beta-Binomial conjugacy."""
    def __init__(self, n_arms: int):
        self.n_arms = n_arms
        # Uninformative Uniform Prior: Beta(1, 1)
        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)

    def select_arm(self) -> int:
        """Samples from posterior distributions to balance exploration and exploitation."""
        samples = np.random.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, chosen_arm: int, reward: int) -> None:
        """Instantaneous O(1) Bayesian conjugate parameter update."""
        if reward == 1:
            self.alpha[chosen_arm] += 1
        else:
            self.beta[chosen_arm] += 1

    def expected_conversion_rates(self) -> np.ndarray:
        """Returns the posterior mean expectation for each arm."""
        return self.alpha / (self.alpha + self.beta)
```
