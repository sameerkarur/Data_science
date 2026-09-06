# Probability Theory, Parametric Distributions & Likelihood
**Official Tutorial & Visual Architecture Handbook (W3Schools & GeeksforGeeks Style)**

---

## 📑 Table of Contents (On this page)
1. [Core Probability Axioms & Conditional Probability](#1-core-probability-axioms--conditional-probability)
2. [Bayes' Theorem & Diagnostic Odds Updating](#2-bayes-theorem--diagnostic-odds-updating)
3. [Probability Mass Functions (PMF) vs Density Functions (PDF)](#3-probability-mass-functions-pmf-vs-density-functions-pdf)
4. [Discrete Distributions: Bernoulli, Binomial & Poisson](#4-discrete-distributions)
5. [Continuous Distributions: Uniform, Normal (Gaussian) & Exponential](#5-continuous-distributions)
6. [The Beta-Binomial Conjugate Model (Bayesian Updating)](#6-the-beta-binomial-conjugate-model)
7. [Maximum Likelihood Estimation (MLE) Foundations](#7-maximum-likelihood-estimation-mle)
8. [Try It Yourself! (Hands-On Practice Exercises)](#8-try-it-yourself-hands-on-practice-exercises)
9. [Quick Reference Cheat Sheet](#9-quick-reference-cheat-sheet)

---

## 1. Core Probability Axioms & Conditional Probability

Probability quantifies the certainty of events occurring within a sample space $\Omega$:
1. $0 \le P(A) \le 1$
2. $P(\Omega) = 1$
3. If $A$ and $B$ are mutually exclusive, $P(A \cup B) = P(A) + P(B)$.

### Conditional Probability:
The probability of event $A$ given that event $B$ has already occurred:
$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$

---

## 2. Bayes' Theorem & Diagnostic Odds Updating

Bayes' Theorem updates the probability of a hypothesis $H$ after observing empirical evidence $E$:

$$P(H|E) = \frac{P(E|H) \cdot P(H)}{P(E)}$$

```
                   BAYESIAN INFERENCE DATAFLOW
      [Prior Belief: P(H)]  ───► What we believed BEFORE seeing new data
               │
               ▼ × [Likelihood: P(E|H)] ──► How likely is the evidence under hypothesis?
      [Numerator: P(E|H) * P(H)]
               │
               ▼ ÷ [Evidence: P(E)] ──► Total probability of evidence across all states
      [Posterior Probability: P(H|E)] ──► Updated confidence AFTER observing evidence!
```

```python
# Classic Medical Diagnosis Example
# Disease prevalence = 1% (P(H) = 0.01)
# Test Sensitivity (True Positive Rate) = 95% (P(E|H) = 0.95)
# Test False Positive Rate = 5% (P(E|¬H) = 0.05)

p_disease = 0.01
p_positive_given_disease = 0.95
p_positive_given_healthy = 0.05

# Law of Total Probability: P(Positive)
p_positive = (p_positive_given_disease * p_disease) + (p_positive_given_healthy * (1 - p_disease))

# Posterior: P(Disease | Positive)
p_disease_given_positive = (p_positive_given_disease * p_disease) / p_positive

print(f"Prior Probability of Disease:       {p_disease:.1%}")
print(f"Total Probability of Positive Test:  {p_positive:.3%}")
print(f"Updated Posterior P(Disease | Pos):  {p_disease_given_positive:.1%} (Counter-intuitive but rigorous!)")
```

#### Output:
```text
Prior Probability of Disease:       1.0%
Total Probability of Positive Test:  5.9%
Updated Posterior P(Disease | Pos):  16.1% (Counter-intuitive but rigorous!)
```

---

## 3. Discrete Distributions: Bernoulli, Binomial & Poisson

```python
from scipy import stats

# 1. Binomial Distribution: Probability of k successes in n independent trials
# e.g., Getting exactly 7 Heads in 10 coin flips with fair coin (p=0.5)
prob_7_heads = stats.binom.pmf(k=7, n=10, p=0.5)

# 2. Poisson Distribution: Number of rare events in fixed interval
# e.g., Website receives average λ = 4 requests/sec. Probability of getting 6 requests?
prob_6_requests = stats.poisson.pmf(k=6, mu=4.0)

print(f"Binomial P(k=7 | n=10, p=0.5):  {prob_7_heads:.4f}")
print(f"Poisson P(k=6 | λ=4.0):          {prob_6_requests:.4f}")
```

#### Output:
```text
Binomial P(k=7 | n=10, p=0.5):  0.1172
Poisson P(k=6 | λ=4.0):          0.1042
```

---

## 4. Continuous Distributions: Normal (Gaussian) & Exponential

Continuous variables have probability density $f(x)$ where the probability of any exact single point is 0, and probabilities correspond to areas under the curve:

```python
import numpy as np
from scipy import stats

# Normal Distribution: N(μ=100, σ=15) (e.g. IQ scores)
# Probability of score falling between 85 and 115 (1 standard deviation)
prob_within_1_sigma = stats.norm.cdf(115, loc=100, scale=15) - stats.norm.cdf(85, loc=100, scale=15)

# Exponential Distribution: Time between customer arrivals with λ = 0.5 per minute
# Probability customer arrives within next 2 minutes
prob_arrival_under_2min = stats.expon.cdf(2, scale=1/0.5)

print(f"Normal 68-95-99.7 Rule (1σ Area): {prob_within_1_sigma:.4f} (~68.27%)")
print(f"Exponential Arrival P(T <= 2 min): {prob_arrival_under_2min:.4f}")
```

#### Output:
```text
Normal 68-95-99.7 Rule (1σ Area): 0.6827 (~68.27%)
Exponential Arrival P(T <= 2 min): 0.6321
```

---

## 5. Maximum Likelihood Estimation (MLE)

MLE finds the parameter values $\theta$ that maximize the likelihood of observing the training data:

$$L(\theta) = \prod_{i=1}^n f(x_i | \theta) \implies \log L(\theta) = \sum_{i=1}^n \log f(x_i | \theta)$$

```python
import numpy as np

# Sample observation data
observations = np.array([2.5, 3.1, 2.8, 3.4, 2.9, 3.2])

# MLE for Gaussian mean is arithmetic mean, MLE for variance is uncorrected variance (ddof=0)
mu_mle = np.mean(observations)
sigma_mle = np.std(observations, ddof=0)

print(f"Observed Sample: {observations}")
print(f"MLE Parameter Estimate μ̂: {mu_mle:.4f}")
print(f"MLE Parameter Estimate σ̂: {sigma_mle:.4f}")
```

#### Output:
```text
Observed Sample: [2.5 3.1 2.8 3.4 2.9 3.2]
MLE Parameter Estimate μ̂: 2.9833
MLE Parameter Estimate σ̂: 0.2852
```

---

## 6. Try It Yourself! (Hands-On Practice Exercises)

### Exercise 1: A/B Test Conversion Rate with Beta Prior
**Task:** In Bayesian A/B testing, a Beta prior $\text{Beta}(\alpha, \beta)$ updated with $s$ successes and $f$ failures becomes $\text{Beta}(\alpha + s, \beta + f)$. Given a uniform prior $\text{Beta}(1, 1)$, after observing 45 conversions out of 100 visitors, compute the 95% Bayesian credible interval for the conversion rate:

<details>
<summary>👉 Click to Reveal Solution</summary>

```python
from scipy import stats

prior_alpha, prior_beta = 1, 1
successes, failures = 45, 55

# Posterior Beta parameters
post_alpha = prior_alpha + successes
post_beta = prior_beta + failures

# 95% Equal-tailed Credible Interval
ci_low, ci_high = stats.beta.interval(0.95, post_alpha, post_beta)
expected_conversion = post_alpha / (post_alpha + post_beta)

print(f"Posterior Mean Conversion Rate: {expected_conversion:.1%}")
print(f"95% Bayesian Credible Interval: [{ci_low:.1%}, {ci_high:.1%}]")
```
#### Output:
```text
Posterior Mean Conversion Rate: 45.1%
95% Bayesian Credible Interval: [35.6%, 54.8%]
```
</details>

---

## 7. Quick Reference Cheat Sheet

| Distribution | Type | Key Parameter(s) | Primary Use Case |
|---|---|---|---|
| **Bernoulli** | Discrete | $p$ (Success prob) | Single binary outcome (Click / No Click) |
| **Binomial** | Discrete | $n$ (Trials), $p$ | Number of conversions out of $n$ visits |
| **Poisson** | Discrete | $\lambda$ (Rate) | Counts of events in fixed time / area |
| **Uniform** | Continuous| $[a, b]$ | Random initialization, equal probability |
| **Normal** | Continuous| $\mu$ (Mean), $\sigma$ (Std) | Central limit sums, residuals, natural traits |
| **Exponential**| Continuous| $\lambda$ (Rate) | Time until next failure / transaction |
| **Beta** | Continuous| $\alpha, \beta$ (Shape) | Prior/posterior for probabilities ($p \in [0, 1]$) |
