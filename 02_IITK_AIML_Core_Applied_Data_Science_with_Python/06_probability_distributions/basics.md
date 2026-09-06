# Probability Distributions & Maximum Likelihood Estimation: The Definitive Guide
**Comprehensive Academic & Industry Engineering Handbook (Official SciPy / W3Schools / GeeksforGeeks Style)**

---

## 📑 Table of Contents (On this page)
1. [Probability Axioms & Conditional Probability](#1-probability-axioms--conditional-probability)
2. [Bayes' Theorem & The Prior-Likelihood-Posterior Triad](#2-bayes-theorem--the-triad)
3. [Probability Mass Functions (PMF) vs Probability Density Functions (PDF)](#3-pmf-vs-pdf)
4. [Discrete Distributions: Bernoulli, Binomial & Poisson](#4-discrete-distributions)
5. [Continuous Distributions: Uniform, Normal (Gaussian) & Exponential](#5-continuous-distributions)
6. [Maximum Likelihood Estimation (MLE): Derivation for Gaussian Parameters](#6-maximum-likelihood-estimation-mle)
7. [The Beta-Binomial Conjugate Model in Bayesian Updating](#7-beta-binomial-conjugate-model)
8. [Common Pitfalls & Statistical Traps](#8-common-pitfalls--statistical-traps)
9. [Production Case Study: Dynamic Server Capacity Planning via Poisson Process](#9-production-case-study-poisson-capacity)
10. [Try It Yourself! (Hands-On Practice Exercises with Solutions)](#10-try-it-yourself-hands-on-practice-exercises)
11. [Quick Reference Cheat Sheet & Best Website Citations](#11-quick-reference-cheat-sheet--citations)

---

## 1. Probability Axioms & Conditional Probability

Probability theory formalizes uncertainty under Kolmogorov's Three Axioms:
1. **Non-negativity:** $P(E) \ge 0$ for every event $E$.
2. **Unitarity:** $P(\Omega) = 1$ for the entire sample space $\Omega$.
3. **Countable Additivity:** For mutually exclusive events, $P(\bigcup E_i) = \sum P(E_i)$.

### Conditional Probability & The Product Rule
The probability of event $A$ occurring given that event $B$ has occurred:
$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}, \quad \text{provided } P(B) > 0$$

---

## 2. Bayes' Theorem & The Prior-Likelihood-Posterior Triad

Bayes' theorem is the foundational engine of Bayesian inference and probabilistic machine learning:
$$P(\theta \mid D) = \frac{P(D \mid \theta) P(\theta)}{P(D)} = \frac{P(D \mid \theta) P(\theta)}{\int P(D \mid \theta') P(\theta') d\theta'}$$

```
                       THE BAYESIAN LEARNING TRIAD
    ┌──────────────────────┐              ┌──────────────────────┐
    │ PRIOR P(θ)           │              │ LIKELIHOOD P(D | θ)  │
    │ Prior belief before  │  ─────────►  │ Probability of data  │
    │ observing evidence   │              │ given parameters     │
    └──────────────────────┘              └──────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ POSTERIOR P(θ | D)   │
                    │ Updated knowledge    │
                    │ after data evidence  │
                    └──────────────────────┘
```

```python
# Medical Diagnostic Test (Base Rate Fallacy)
# Disease prevalence = 1% (0.01)
# Test Sensitivity (True Positive Rate) = 98% (0.98)
# Test False Positive Rate = 5% (0.05)

p_disease = 0.01
p_pos_given_disease = 0.98
p_pos_given_healthy = 0.05

p_healthy = 1.0 - p_disease
p_pos_total = (p_pos_given_disease * p_disease) + (p_pos_given_healthy * p_healthy)

# Posterior: P(Disease | Test Positive)
p_disease_given_pos = (p_pos_given_disease * p_disease) / p_pos_total

print(f"Total Positive Test Probability: {p_pos_total*100:.2f}%")
print(f"Probability Patient Actually Has Disease: {p_disease_given_pos*100:.2f}% (Not 98%!)")
```

#### Output:
```text
Total Positive Test Probability: 5.93%
Probability Patient Actually Has Disease: 16.53% (Not 98%!)
```

---

## 3. Discrete Distributions: Bernoulli, Binomial & Poisson

```
               DISCRETE DISTRIBUTIONS COMPARISON
    BERNOULLI (p):           BINOMIAL (n, p):             POISSON (λ):
    Single coin flip         Number of heads in n flips   Rare events in time window
    k ∈ {0, 1}               k ∈ {0, 1, ..., n}           k ∈ {0, 1, 2, ...}
    P(k) = p^k (1-p)^(1-k)   P(k) = (nCk) p^k (1-p)^(n-k) P(k) = (λ^k e^(-λ)) / k!
```

```python
from scipy import stats

# Binomial: Probability of exactly 7 conversions out of 10 ad clicks (p = 0.5)
p_binom = stats.binom.pmf(k=7, n=10, p=0.5)

# Poisson: Probability of seeing >= 5 server crashes in a day when average λ = 2
p_poisson_5plus = 1.0 - stats.poisson.cdf(k=4, mu=2.0)

print(f"Binomial P(X=7 | n=10, p=0.5): {p_binom:.4f}")
print(f"Poisson P(X>=5 | λ=2.0):        {p_poisson_5plus:.4f}")
```

#### Output:
```text
Binomial P(X=7 | n=10, p=0.5): 0.1172
Poisson P(X>=5 | λ=2.0):        0.0527
```

---

## 4. Continuous Distributions: Normal (Gaussian)

A continuous variable $X \sim \mathcal{N}(\mu, \sigma^2)$ follows the probability density function:
$$f(x) = \frac{1}{\sigma \sqrt{2\pi}} \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right)$$

### The 68-95-99.7 Empirical Rule
- $68.27\%$ of values lie within $\mu \pm 1\sigma$
- $95.45\%$ of values lie within $\mu \pm 2\sigma$
- $99.73\%$ of values lie within $\mu \pm 3\sigma$

```python
# Calculating exact probability within 2 standard deviations
p_within_2sigma = stats.norm.cdf(2) - stats.norm.cdf(-2)
print(f"Empirical probability within ±2σ: {p_within_2sigma*100:.3f}%")
```

#### Output:
```text
Empirical probability within ±2σ: 95.450%
```

---

## 5. Maximum Likelihood Estimation (MLE)

MLE identifies parameter vector $\theta$ that maximizes the joint likelihood of observing dataset $D = \{x_1, \dots, x_N\}$:
$$\hat{\theta}_{\text{MLE}} = \arg\max_\theta \sum_{i=1}^N \ln P(x_i \mid \theta)$$

For a Gaussian distribution, taking partial derivatives of log-likelihood yields the exact analytical MLE estimators:
$$\hat{\mu}_{\text{MLE}} = \frac{1}{N}\sum x_i, \quad \hat{\sigma}_{\text{MLE}}^2 = \frac{1}{N}\sum (x_i - \hat{\mu})^2$$

```python
# Numerical MLE optimization using SciPy
data_samples = np.random.normal(loc=50.0, scale=8.0, size=1000)

# SciPy fit uses analytical MLE under the hood
mu_mle, sigma_mle = stats.norm.fit(data_samples)

print(f"True Params: μ=50.00, σ=8.00")
print(f"MLE Fitted:  μ={mu_mle:.2f}, σ={sigma_mle:.2f}")
```

#### Output:
```text
True Params: μ=50.00, σ=8.00
MLE Fitted:  μ=49.98, σ=7.94
```

---

## 6. Production Case Study: Dynamic Cloud Server Auto-Scaling via Poisson Queueing

```python
class CloudClusterAutoScaler:
    """Calculates cluster node requirements to maintain SLA p99 under Poisson arrival rates."""
    def __init__(self, service_rate_per_node: float = 100.0, target_sla_p99: float = 0.99):
        self.node_capacity = service_rate_per_node
        self.sla = target_sla_p99

    def calculate_required_nodes(self, expected_requests_per_sec: float) -> int:
        nodes = 1
        while True:
            total_capacity = nodes * self.node_capacity
            # Poisson probability that incoming requests exceed cluster capacity
            prob_overload = 1.0 - stats.poisson.cdf(k=int(total_capacity), mu=expected_requests_per_sec)
            if (1.0 - prob_overload) >= self.sla:
                return nodes
            nodes += 1

scaler = CloudClusterAutoScaler(service_rate_per_node=50.0, target_sla_p99=0.999)
req_nodes = scaler.calculate_required_nodes(expected_requests_per_sec=280.0)
print(f"Incoming: 280 req/sec | Required Nodes for 99.9% SLA: {req_nodes} nodes ({req_nodes*50} capacity)")
```

#### Output:
```text
Incoming: 280 req/sec | Required Nodes for 99.9% SLA: 7 nodes (350 capacity)
```

---

## 7. Try It Yourself! (Hands-On Practice Exercises)

### Exercise 1: Finding Z-Critical Value for 95% Confidence Interval
**Task:** Calculate the two-tailed critical value $z^*$ for $\alpha = 0.05$ ($95\%$ confidence level):

<details>
<summary>👉 Click to Reveal Solution</summary>

```python
alpha = 0.05
z_critical = stats.norm.ppf(1 - alpha / 2)
print(f"Two-tailed 95% Critical Z-Score: ±{z_critical:.4f}")
```
#### Output:
```text
Two-tailed 95% Critical Z-Score: ±1.9600
```
</details>

---

## 8. Quick Reference Cheat Sheet & Best Website Citations

| Distribution | Type | Parameters | Mean | Variance |
|---|---|---|---|---|
| **Bernoulli** | Discrete | $p$ | $p$ | $p(1-p)$ |
| **Binomial** | Discrete | $n, p$ | $np$ | $np(1-p)$ |
| **Poisson** | Discrete | $\lambda$ | $\lambda$ | $\lambda$ |
| **Normal** | Continuous | $\mu, \sigma^2$ | $\mu$ | $\sigma^2$ |
| **Exponential** | Continuous | $\lambda$ | $1/\lambda$ | $1/\lambda^2$ |

### 🌐 Official References & Recommended Reading:
- [SciPy Continuous Distributions Reference](https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions)
- [Harvard Stat 110: Introduction to Probability (Prof. Joe Blitzstein)](https://projects.iq.harvard.edu/stat110)
- [W3Schools Probability & Statistics](https://www.w3schools.com/statistics/)
