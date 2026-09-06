# Interview Q&A — Introduction to Data Science & CRISP-DM

> **30 High-Yield Questions & Model Answers** for AI/ML and Data Science Technical Interviews.

### Q1. Explain the complete Data Science lifecycle (CRISP-DM phases).

**Answer:** CRISP-DM consists of 6 iterative phases: (1) Business Understanding (defining KPIs, problem formulation), (2) Data Understanding (collection, EDA, identifying data quality issues), (3) Data Preparation (cleaning, feature engineering, imputation, normalization), (4) Modeling (selecting algorithms, training, tuning), (5) Evaluation (assessing against business metrics and statistical benchmarks), and (6) Deployment (monitoring data drift, serving predictions via API or batch pipelines).

### Q2. Differentiate between Descriptive, Diagnostic, Predictive, and Prescriptive analytics.

**Answer:** Descriptive: 'What happened?' (e.g. historical sales dashboards). Diagnostic: 'Why did it happen?' (e.g. root cause analysis, correlation breakdowns). Predictive: 'What is likely to happen?' (e.g. churn forecasting, demand prediction models). Prescriptive: 'What action should we take?' (e.g. price optimization algorithms, automated inventory reordering).

### Q3. What is the difference between structured, semi-structured, and unstructured data?

**Answer:** Structured data has a strict, predefined tabular schema (e.g. SQL relational databases, CSV files). Semi-structured data possesses organizational markers without a rigid tabular schema (e.g. JSON documents, XML, NoSQL documents). Unstructured data lacks any conceptual schema (e.g. raw text, audio files, video streams, medical imaging scans).

### Q4. Explain the difference between a population and a sample in data analysis.

**Answer:** A population is the entire universe of entities or observations of interest (e.g. all 100 million retail transactions in a year). A sample is a representative subset drawn from that population. Statistical inference uses sample statistics (mean x̄, variance s²) to estimate unobservable true population parameters (μ, σ²).

### Q5. What are confounding variables and how do they lead to spurious correlations?

**Answer:** A confounding variable is an external variable that correlates with both the independent variable and the dependent variable, creating a misleading statistical association. Classic example: ice cream sales correlate with drowning deaths; the confounding variable is summer temperature. Solved via randomized controlled trials (RCTs), stratification, or multivariate regression control.

### Q6. Explain Simpson's Paradox with a practical data science example.

**Answer:** Simpson's Paradox occurs when a statistical trend appears in several distinct subgroups of data but reverses or disappears when the groups are combined. For example, Treatment A may have higher cure rates than Treatment B in both mild and severe patient cohorts, but Treatment B looks better overall because it was disproportionately administered to patients with mild illness.

### Q7. What is Data Leakage in a machine learning pipeline?

**Answer:** Data leakage occurs when information from the target variable or future test data is inadvertently introduced into the training process. Examples include: scaling the entire dataset before train/test splitting, including features calculated after the target event occurs, or leaking duplicate user records across train and test splits.

### Q8. How do observational studies differ from randomized experiments (A/B testing)?

**Answer:** In observational studies, researchers record data without intervening, which exposes findings to confounding bias and selection effects. In randomized experiments (A/B testing), subjects are randomly assigned to treatment and control groups, balancing observed and unobserved confounders and establishing true causal relationships.

### Q9. What is the difference between quantitative and qualitative data types?

**Answer:** Quantitative data represents numerical measurements and counts: discrete (integer counts like website visits) and continuous (infinitely divisible measurements like height or latency). Qualitative data represents categorical labels: nominal (unordered categories like gender or state) and ordinal (ranked categories like customer rating: Low, Medium, High).

### Q10. Explain the concept of exploratory data analysis (EDA) and its primary goals.

**Answer:** EDA is an investigative approach to examine datasets before formal modeling. Primary goals: uncover underlying structure, identify data hygiene anomalies (missing values, typos, outliers), test fundamental statistical assumptions (normality, collinearity), and generate feature engineering hypotheses.

### Q11. What are key dimensions of Data Quality in enterprise analytics?

**Answer:** Key dimensions include: Accuracy (correct real-world representation), Completeness (absence of missing fields), Consistency (uniform values across systems), Timeliness (data freshness and latency), Validity (adherence to business schemas and regex formats), and Uniqueness (absence of duplicated records).

### Q12. What is the difference between Batch processing and Stream processing?

**Answer:** Batch processing collects and processes high volumes of data in scheduled periodic chunks (e.g. nightly ETL running via Apache Spark). Stream processing continuously ingests, transforms, and analyzes individual events in real-time with sub-second latency (e.g. Apache Kafka, Flink).

### Q13. Explain the difference between ETL (Extract-Transform-Load) and ELT (Extract-Load-Transform).

**Answer:** ETL transforms data on a specialized staging server before loading it into the data warehouse, common with legacy systems and sensitive PII masking. ELT loads raw data directly into high-scale modern cloud data lakes/warehouses (Snowflake, BigQuery) and performs distributed SQL transformations inside the warehouse.

### Q14. What is survivor bias in data collection?

**Answer:** Survivor bias occurs when an analysis focuses only on entities that passed a selection process while overlooking those that failed. Classic example: analyzing armor on returning WWII aircraft (the undamaged areas were where shot-down planes were hit) or studying only successful startup founders.

### Q15. How do you design a robust data dictionary for a business dataset?

**Answer:** A data dictionary documents: Column Name, Data Type, Nullability, Description/Business Meaning, Unit of Measure, Acceptable Value Range/Enumeration, Primary/Foreign Key relationships, and Upstream Data Source/Owner.

### Q16. What is the difference between internal validity and external validity?

**Answer:** Internal validity assesses whether the experimental design eliminates alternative explanations and confounders within the test sample. External validity assesses whether experimental findings generalize to broader populations, external environments, and future timeframes.

### Q17. What is cross-sectional data vs time-series data vs panel data?

**Answer:** Cross-sectional data captures multiple subjects at a single point in time (e.g. customer survey in Dec). Time-series data captures a single entity tracked over sequential intervals (e.g. daily Apple stock price). Panel (longitudinal) data captures multiple entities tracked repeatedly across multiple time periods.

### Q18. What is the role of reproducible research in data science?

**Answer:** Reproducibility ensures independent practitioners can execute the exact code and pipeline on the raw data to achieve identical analytical results. Achieved via virtual environments (Docker, venv), seed fixing (np.random.seed), deterministic ETL scripts, and version-controlled data/models (DVC, Git).

### Q19. Explain the GIGO (Garbage In, Garbage Out) principle in AI/ML.

**Answer:** GIGO dictates that no matter how complex, deep, or state-of-the-art an algorithm is, feeding it noisy, biased, uncleaned, or mislabeled training data inevitably produces invalid, unreliable, and biased predictions in production.

### Q20. What is selection bias and how can it skew data analysis?

**Answer:** Selection bias occurs when the sample data is systematically non-representative of the target population due to flawed sampling procedures (e.g. voluntary online polls over-representing extreme opinions). It skews effect sizes and undermines model generalization.

### Q21. How does correlation differ from causation?

**Answer:** Correlation measures the statistical strength and direction of linear association between two variables. Causation implies that changing variable X directly produces a change in variable Y. Correlation can exist without causation due to coincidental trends, reverse causation, or confounding third variables.

### Q22. What is an outlier and what are common strategies to handle it?

**Answer:** An outlier is an observation that deviates substantially from the overall pattern of data. Strategies: (1) verify if it's a data entry error, (2) winsorize/cap values at 1st and 99th percentiles, (3) log-transform skewed features, (4) use robust estimators (median, Huber loss), or (5) isolate via algorithms like Isolation Forest.

### Q23. What is the purpose of hypothesis testing in business analytics?

**Answer:** Hypothesis testing provides a mathematical framework to determine whether an observed business metric change (e.g. +5% conversion on a redesigned checkout) is statistically significant or merely an artifact of random sampling noise.

### Q24. Explain the concept of statistical power.

**Answer:** Statistical power (1 - β) is the probability that a statistical test will correctly reject a false null hypothesis (i.e. detecting a true effect when one exists). High power depends on larger sample sizes, larger true effect sizes, and lower background variance.

### Q25. What is a KPI (Key Performance Indicator) and what makes a good KPI?

**Answer:** A KPI is a quantifiable metric reflecting the core business objectives of an enterprise. A strong KPI is: Aligned with strategic goals, Actionable, Measurable without ambiguity, Interpretable by stakeholders, and resistant to gaming (Goodhart's Law).

### Q26. Explain Goodhart's Law in metrics and machine learning.

**Answer:** 'When a measure becomes a target, it ceases to be a good measure.' When employees or models are incentivized strictly to optimize a proxy metric, they find degenerate shortcuts that hit the metric while violating underlying business intentions.

### Q27. What is data drift vs concept drift?

**Answer:** Data drift (covariate shift) occurs when the input feature distribution P(X) changes over time while the relationship to the target P(Y|X) remains constant. Concept drift occurs when the true relationship between features and target P(Y|X) changes (e.g. consumer spending patterns post-pandemic).

### Q28. What is an executive summary and how do you present technical findings to non-technical leaders?

**Answer:** An executive summary highlights the core business context, the key finding/quantified financial impact, and actionable recommendations. Avoid technical jargon, emphasize ROI/risk reduction, and support conclusions with intuitive visualizations.

### Q29. What is privacy-preserving data analysis and what does PII mean?

**Answer:** PII (Personally Identifiable Information) includes data that identifies an individual (SSN, email, phone, IP). Privacy-preserving techniques include: anonymization, pseudonymization, k-anonymity, differential privacy, and hashing.

### Q30. How do you evaluate whether a business problem should be solved with ML vs simple heuristics?

**Answer:** Use rule-based heuristics if the logic is deterministic, rules are easily maintained, data is sparse, or strict auditability is required. Use Machine Learning when patterns are too complex for human rules, relationships are non-linear, high-dimensional data is available, and predictions scale across millions of instances.
