# Interview Q&A — Probability Distributions & Bayes

> **30 High-Yield Questions & Model Answers** for AI/ML and Data Science Technical Interviews.

### Q1. What is the difference between Probability Mass Function (PMF), Probability Density Function (PDF), and Cumulative Distribution Function (CDF)?

**Answer:** PMF P(X = x) assigns discrete probabilities to specific integer outcomes such that sum(P(x)) = 1. PDF f(x) describes continuous probability density where the probability of any single exact point is 0, and probabilities are obtained by integrating over intervals: P(a <= X <= b) = int_a^b f(x) dx. CDF F(x) = P(X <= x) gives the cumulative probability up to value x for both discrete and continuous distributions.

### Q2. State Bayes' Theorem mathematically and define prior, likelihood, marginal, and posterior.

**Answer:** P(A|B) = [P(B|A) * P(A)] / P(B). P(A) is the Prior probability of hypothesis A before observing evidence. P(B|A) is the Likelihood of observing evidence B given hypothesis A. P(B) is the Marginal probability (evidence). P(A|B) is the Posterior probability of hypothesis A updated after observing evidence B.

### Q3. Explain the difference between Mutually Exclusive events and Independent events.

**Answer:** Mutually exclusive events cannot occur simultaneously: P(A and B) = 0, so P(A or B) = P(A) + P(B). Independent events have no influence on each other's occurrence: P(A|B) = P(A), so P(A and B) = P(A) * P(B). If two events with non-zero probability are mutually exclusive, they CANNOT be independent.

### Q4. What is a Bernoulli Trial and how does it relate to the Binomial Distribution?

**Answer:** A Bernoulli trial is a single random experiment with exactly two mutually exclusive outcomes: Success (probability p) and Failure (probability 1 - p). A Binomial distribution Bin(n, p) models the number of successes k observed in n independent and identical Bernoulli trials: P(k) = C(n, k) * p^k * (1 - p)^(n - k).

### Q5. When should you model counts using a Poisson Distribution instead of a Binomial Distribution?

**Answer:** Use a Poisson distribution Po(λ) when modeling the number of rare events occurring in a fixed interval of time or space with a known constant average rate λ, where events occur independently (e.g. server requests per second, website crashes per month). The Poisson is the mathematical limit of the Binomial as n -> inf and p -> 0 with n*p = λ.

### Q6. Explain the memoryless property of the Exponential Distribution.

**Answer:** The Exponential distribution models the waiting time between Poisson events. It is memoryless: P(X > s + t | X > s) = P(X > t). The probability that an event occurs in the next t minutes does not depend on how much time s has already elapsed without the event occurring.

### Q7. What is the Uniform Distribution and what are its mean and variance?

**Answer:** A continuous Uniform distribution U(a, b) assigns constant probability density f(x) = 1 / (b - a) across the interval [a, b]. Mean = (a + b) / 2. Variance = (b - a)² / 12.

### Q8. What is Maximum Likelihood Estimation (MLE)?

**Answer:** MLE is a method of estimating parameters θ of a probability distribution by finding parameter values that maximize the likelihood function L(θ; X) = prod(f(x_i; θ)), typically achieved by maximizing the log-likelihood sum(log f(x_i; θ)) via calculus or gradient descent.

### Q9. How does Maximum A Posteriori (MAP) estimation differ from MLE?

**Answer:** MLE finds parameters that maximize likelihood alone: θ_MLE = argmax P(X|θ). MAP incorporates prior beliefs via Bayes' rule: θ_MAP = argmax P(X|θ) P(θ). When a Gaussian prior is placed on weights, MAP is mathematically equivalent to L2 regularization (Ridge); a Laplace prior corresponds to L1 (Lasso).

### Q10. Explain the Law of Total Probability.

**Answer:** If {B1, B2, ..., Bn} forms a partition of the sample space, then for any event A: P(A) = sum_{i=1}^n P(A|B_i) * P(B_i). It allows computing marginal probabilities by conditioning across all mutually exclusive scenarios.

### Q11. What is Joint Probability vs Marginal Probability vs Conditional Probability?

**Answer:** Joint probability P(A and B) is the likelihood that both events occur simultaneously. Marginal probability P(A) is the probability of an event irrespective of the outcome of other variables. Conditional probability P(A|B) = P(A and B) / P(B) is the probability of A occurring given that B is known to have occurred.

### Q12. What is a Random Variable?

**Answer:** A random variable is a mathematical function that maps outcomes of a random process in a sample space to real numbers. It can be discrete (countable set of values) or continuous (uncountable real continuum).

### Q13. Explain Expectation and Variance of a random variable.

**Answer:** Expected value E[X] is the probability-weighted average: sum(x * P(x)) for discrete, int(x * f(x) dx) for continuous. Variance Var(X) = E[(X - E[X])²] = E[X²] - (E[X])² measures dispersion around the expectation.

### Q14. What is the Geometric Distribution and what is its expected value?

**Answer:** The Geometric distribution models the number of Bernoulli trials X required to achieve the first success: P(X = k) = (1 - p)^(k - 1) * p. Its expected value is E[X] = 1 / p.

### Q15. What is the Hypergeometric Distribution and how does it differ from the Binomial?

**Answer:** The Binomial distribution models sampling WITH replacement (independent trials, constant p). The Hypergeometric distribution models sampling WITHOUT replacement from a finite population of size N containing K successes, where probabilities change on every draw.

### Q16. Explain the Beta Distribution and why it is commonly used as a conjugate prior for Binomial likelihood.

**Answer:** The Beta distribution Beta(α, β) is defined over the interval [0, 1], making it ideal for modeling probabilities and conversion rates. When combined with a Binomial likelihood of k successes and n-k failures, the posterior distribution is analytically Beta(α + k, β + n - k), enabling closed-form Bayesian updating.

### Q17. What is the Gamma Distribution?

**Answer:** The Gamma distribution is a two-parameter family of continuous distributions that generalizes the Exponential distribution to model waiting times until α Poisson events occur.

### Q18. What is the Chi-Square Distribution and how is it related to the Normal Distribution?

**Answer:** A Chi-Square distribution with k degrees of freedom is the distribution of the sum of squares of k independent standard normal random variables: Q = sum_{i=1}^k Z_i². It is the basis for goodness-of-fit and contingency table tests.

### Q19. Explain Student's t-Distribution and why it has heavier tails than the Normal Distribution.

**Answer:** The t-distribution arises when estimating the mean of a normally distributed population when sample size is small and population standard deviation σ is unknown. Because sample variance s² introduces additional estimation uncertainty, the t-distribution has fatter tails than the Gaussian, approaching normal as df -> inf.

### Q20. What is the F-Distribution and where is it used in data science?

**Answer:** The F-distribution is the ratio of two independent Chi-Square distributed variables divided by their respective degrees of freedom: F = (U1 / d1) / (U2 / d2). It is the test statistic in ANOVA and tests for equality of variances between two populations.

### Q21. What is Covariance of independent random variables?

**Answer:** If two random variables X and Y are independent, their covariance Cov(X, Y) = E[XY] - E[X]E[Y] = 0. However, the converse is not always true: zero covariance only indicates absence of linear relationship; non-linear dependencies may still exist.

### Q22. Explain Markov's Inequality.

**Answer:** For any non-negative random variable X and constant a > 0: P(X >= a) <= E[X] / a. It provides an upper bound on the probability that a non-negative variable exceeds a threshold based strictly on its mean.

### Q23. What is a Probability Generating Function (PGF) and Moment Generating Function (MGF)?

**Answer:** The MGF M_X(t) = E[e^(tX)] uniquely determines the probability distribution. Taking the nth derivative evaluated at t = 0 yields the nth non-central moment: M_X^(n)(0) = E[X^n].

### Q24. Explain the Monte Carlo method in probabilistic simulations.

**Answer:** Monte Carlo methods use repeated pseudo-random sampling from probability distributions to approximate numerical results that are analytically intractable (e.g. multi-dimensional integration, project burndown risk simulation, option pricing).

### Q25. What is Inverse Transform Sampling?

**Answer:** Inverse transform sampling generates random numbers from any distribution with an invertible CDF F(x): sample u ~ Uniform(0, 1), then x = F^(-1)(u) has the desired distribution.

### Q26. What is a Multinomial Distribution?

**Answer:** The Multinomial distribution generalizes the Binomial distribution to outcomes with k > 2 mutually exclusive categories across n independent trials (e.g. rolling a 6-sided die n times).

### Q27. Explain the Dirichlet Distribution and its application in Latent Dirichlet Allocation (LDA).

**Answer:** The Dirichlet distribution is a multivariate generalization of the Beta distribution over probability simplices (vectors summing to 1). In topic modeling (LDA), it serves as the conjugate prior for multinomial document-topic and topic-word distributions.

### Q28. What is the difference between Frequentist and Bayesian probability interpretations?

**Answer:** Frequentists view probability as the long-run relative frequency of repeatable events under identical conditions; parameters are fixed, unknown constants. Bayesians view probability as a subjective degree of belief quantified using probability theory, updating prior distributions using evidence into posterior distributions.

### Q29. Explain the Monty Hall Problem and its counter-intuitive probabilistic solution.

**Answer:** In a 3-door game with 1 car and 2 goats, you pick Door 1 (P = 1/3). Host opens Door 3 showing a goat. Switching to Door 2 doubles your win probability to 2/3 because the host's action is constrained: the 2/3 probability that the car was behind an unchosen door collapses entirely onto the remaining unopened door.

### Q30. What is the Birthday Paradox and what is the underlying probabilistic formula?

**Answer:** The probability that in a group of n people, at least two share a birthday exceeds 50% with just n = 23 people. It is computed via the complement: P(at least 1 match) = 1 - P(no matches) = 1 - prod_{i=0}^{n-1} (365 - i) / 365.
