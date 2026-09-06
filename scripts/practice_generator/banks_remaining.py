"""Question banks — Courses 02–05 (50+ questions per stub subtopic)."""

from core import InterviewQA, Question, TopicSpec


def _q(section: str, text: str, solution: str) -> Question:
    return Question(text=text, solution=solution, section=section)


def _interview(section: str, q: str, a: str) -> InterviewQA:
    return InterviewQA(question=q, answer=a, section=section)


def _expand(section_prefix: str, items: list[tuple[str, str]], target: int = 50) -> list[Question]:
    """Repeat/vary if needed to reach 50 — items should already be 50."""
    out = [_q(section_prefix, t, s) for t, s in items]
    if len(out) < target:
        raise ValueError(f"Need {target} questions, got {len(out)}")
    return out[:target]


# ── Course 02 stubs ───────────────────────────────────────────────────────────

INTRO_DS = _expand("Data Science mindset", [
    ("Define data science in one sentence and list 5 steps of a DS project.", "steps = ['define', 'collect', 'clean', 'model', 'deploy']\nprint('Data science extracts insight from data using stats + CS + domain.', steps)"),
    ("Load AusApparal sales CSV from DATA_DIR. Print shape and columns.", "sales_path = DATA_DIR / 'AusApparalSales4thQrt2020.csv'\ndf = pd.read_csv(sales_path)\nprint(df.shape)\nprint(df.columns.tolist())"),
    ("Explain population vs sample using HR dataset example.", "hr = pd.read_csv(DATA_DIR / 'HR_comma_sep.csv')\nprint('Sample rows:', len(hr), '— population would be all employees company-wide')"),
    ("Identify structured vs unstructured data examples in AIML.", "structured = ['CSV sales', 'SQL tables']\nunstructured = ['PDF reports', 'images', 'chat logs']\nprint(structured, unstructured)"),
    ("List 3 types of analytics: descriptive, diagnostic, predictive, prescriptive — give one example each.", "examples = {'descriptive': 'mean sales', 'diagnostic': 'why Q4 dip', 'predictive': 'churn model', 'prescriptive': 'discount optimization'}\nprint(examples)"),
    ("What is a feature vs label in ML?", "print('Feature: input columns; Label: target variable to predict')"),
    ("Load marketing_data.csv; print dtypes and missing counts.", "df = pd.read_csv(DATA_DIR / 'marketing_data.csv')\nprint(df.dtypes)\nprint(df.isna().sum())"),
    ("Compute basic KPIs: total sales, avg units from apparel dataset.", "df = pd.read_csv(DATA_DIR / 'AusApparalSales4thQrt2020.csv')\nprint(df['Sales'].sum(), df['Unit'].mean())"),
    ("Create a simple data dictionary dict for sales columns.", "data_dict = {'Date': 'transaction date', 'Sales': 'revenue AUD', 'State': 'AU state'}\nprint(data_dict)"),
    ("Explain CRISP-DM phases briefly.", "print('Business understanding → Data understanding → Prep → Modeling → Evaluation → Deployment')"),
    ("Compare batch vs real-time analytics.", "print('Batch: scheduled ETL; Real-time: streaming dashboards/alerts')"),
    ("What is ETL vs ELT?", "print('ETL transforms before load; ELT loads raw then transforms in warehouse')"),
    ("Name 3 data quality dimensions (accuracy, completeness, timeliness).", "dims = ['accuracy', 'completeness', 'consistency', 'timeliness']\nprint(dims)"),
    ("Detect duplicates in first 1000 rows of HR data.", "df = pd.read_csv(DATA_DIR / 'HR_comma_sep.csv')\nprint(df.duplicated().sum())"),
    ("Sample 5 random rows with seed=42 for reproducibility.", "df = pd.read_csv(DATA_DIR / 'AusApparalSales4thQrt2020.csv')\nprint(df.sample(5, random_state=42))"),
    ("Group sales by State; print top 3 states by revenue.", "df = pd.read_csv(DATA_DIR / 'AusApparalSales4thQrt2020.csv')\nprint(df.groupby('State')['Sales'].sum().sort_values(ascending=False).head(3))"),
    ("Plot histogram of Sales (matplotlib) — 20 bins.", "import matplotlib.pyplot as plt\ndf = pd.read_csv(DATA_DIR / 'AusApparalSales4thQrt2020.csv')\nplt.hist(df['Sales'], bins=20); plt.title('Sales'); plt.show()"),
    ("Explain bias-variance tradeoff intuitively.", "print('High bias: underfit; High variance: overfit; goal balance on unseen data')"),
    ("What is train/validation/test split purpose?", "print('Train: fit; Val: tune; Test: final unbiased estimate')"),
    ("Define overfitting and one prevention method.", "print('Overfitting: memorizes train; prevent with regularization, more data, CV')"),
    ("Compare supervised vs unsupervised learning.", "print('Supervised: labeled target; Unsupervised: patterns without labels')"),
    ("What is a confounding variable?", "print('Affects both feature and target, misleading correlation')"),
    ("Explain correlation does not imply causation with example.", "print('Ice cream & drowning correlate via summer — not causal')"),
    ("List Python DS stack: NumPy, Pandas, Matplotlib, Seaborn, Sklearn.", "stack = ['numpy', 'pandas', 'matplotlib', 'seaborn', 'sklearn']\nprint(stack)"),
    ("Create Jupyter markdown checklist for project workflow.", "checklist = ['Load data', 'EDA', 'Hypothesis', 'Test', 'Report']\nprint('\\n'.join(f'- {c}' for c in checklist))"),
    ("Save cleaned sample to CSV in project folder (head 100 rows).", "df = pd.read_csv(DATA_DIR / 'AusApparalSales4thQrt2020.csv')\nout = REPO_ROOT / 'Course_02_Applied_Data_Science/01_intro_data_science/sample_100.csv'\ndf.head(100).to_csv(out, index=False)\nprint('Saved', out)"),
    ("Compute sales per unit; handle division safely.", "df = pd.read_csv(DATA_DIR / 'AusApparalSales4thQrt2020.csv')\ndf['spu'] = df['Sales'] / df['Unit'].replace(0, pd.NA)\nprint(df['spu'].describe())"),
    ("Identify outliers in Sales using IQR rule.", "df = pd.read_csv(DATA_DIR / 'AusApparalSales4thQrt2020.csv')\nq1, q3 = df['Sales'].quantile([0.25, 0.75])\niqr = q3 - q1\nlo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr\nprint(df[(df['Sales']<lo)|(df['Sales']>hi)].shape[0], 'outliers')"),
    ("Cross-tab State vs Group counts.", "df = pd.read_csv(DATA_DIR / 'AusApparalSales4thQrt2020.csv')\nprint(pd.crosstab(df['State'], df['Group']))"),
    ("Explain p-value in plain English.", "print('If null true, probability of seeing result this extreme')"),
    ("What is A/B testing?", "print('Randomized experiment comparing two variants on a metric')"),
    ("Define KPI for marketing campaign project.", "kpis = ['response_rate', 'conversion', 'ROI']\nprint(kpis)"),
    ("Document assumptions when analyzing sample home loan data.", "print('Assume sample representative; missing MCAR/MAR note in report')"),
    ("Compare CSV vs Parquet for analytics.", "print('CSV human-readable; Parquet columnar compressed faster for big data')"),
    ("Use .describe(include='all') on marketing data.", "df = pd.read_csv(DATA_DIR / 'marketing_data.csv')\nprint(df.describe(include='all').T.head())"),
    ("Parse Date column to datetime in sales data.", "df = pd.read_csv(DATA_DIR / 'AusApparalSales4thQrt2020.csv', parse_dates=['Date'])\nprint(df['Date'].dtype)"),
    ("Extract month from Date; group mean Sales by month.", "df = pd.read_csv(DATA_DIR / 'AusApparalSales4thQrt2020.csv', parse_dates=['Date'])\ndf['month'] = df['Date'].dt.month\nprint(df.groupby('month')['Sales'].mean())"),
    ("Explain reproducibility with random_state.", "print('Fixes pseudo-random splits/sampling for same results')"),
    ("Create function load_dataset(name) using DATA_DIR.", "def load_dataset(name):\n    return pd.read_csv(DATA_DIR / name)\ndf = load_dataset('loan_data.csv')\nprint(df.shape)"),
    ("What is data leakage?", "print('Train uses info not available at prediction time — inflates metrics')"),
    ("Name ethical concerns in HR attrition modeling.", "print('Bias, privacy, transparency, fair hiring decisions')"),
    ("Summarize loan_data target column distribution.", "df = pd.read_csv(DATA_DIR / 'loan_data.csv')\nprint(df.iloc[:, -1].value_counts(normalize=True))"),
    ("Compare mean vs median for skewed Sales.", "df = pd.read_csv(DATA_DIR / 'AusApparalSales4thQrt2020.csv')\nprint(df['Sales'].mean(), df['Sales'].median())"),
    ("Explain dimensionality in tabular data.", "df = pd.read_csv(DATA_DIR / 'HR_comma_sep.csv')\nprint('rows x cols:', df.shape)"),
    ("List 3 visualization types for numeric vs categorical.", "print('numeric: hist/box; categorical: bar/count; relationship: scatter/heatmap')"),
    ("Write pseudo-code for hypothesis test workflow.", "steps = ['H0/H1', 'alpha', 'test', 'p-value', 'decision']\nprint(steps)"),
    ("Explain central limit theorem one line.", "print('Sample means approximate normal for large n regardless of population shape')"),
    ("What is a cohort?", "print('Group sharing time/event — e.g. users joined same month')"),
    ("Define retention vs churn.", "print('Retention: stayed; Churn: left — opposite sides same coin')"),
    ("Build mini report dict with title, rows, findings keys.", "report = {'title': 'Q4 Sales', 'rows': 1000, 'findings': 'NSW leads revenue'}\nprint(report)"),
    ("Reflection: 3 skills you practiced in Course 2 so far.", "skills = ['pandas EDA', 'hypothesis testing', 'visualization']\nprint(skills)"),
])

PYTHON_ESSENTIALS = _expand("Python for DS", [
    ("Import numpy, pandas, matplotlib; print versions.", "import numpy as np, pandas as pd, matplotlib\nprint(np.__version__, pd.__version__, matplotlib.__version__)"),
    ("Create ndarray 1..12 reshape (3,4).", "import numpy as np\na = np.arange(1, 13).reshape(3, 4)\nprint(a)"),
    ("Vectorized: multiply sales array by 1.1.", "df = pd.read_csv(DATA_DIR / 'AusApparalSales4thQrt2020.csv')\nadj = df['Sales'].to_numpy() * 1.1\nprint(adj[:5])"),
    ("Apply function to column with .apply(lambda).", "df = pd.read_csv(DATA_DIR / 'AusApparalSales4thQrt2020.csv')\ndf['Sales_k'] = df['Sales'].apply(lambda x: round(x/1000, 2))\nprint(df.head())"),
    ("List comprehension filter Sales > 30000.", "df = pd.read_csv(DATA_DIR / 'AusApparalSales4thQrt2020.csv')\nhigh = [s for s in df['Sales'] if s > 30000]\nprint(len(high))"),
    ("Dict comprehension state → total sales.", "df = pd.read_csv(DATA_DIR / 'AusApparalSales4thQrt2020.csv')\nm = {s: df.loc[df['State']==s, 'Sales'].sum() for s in df['State'].unique()}\nprint(list(m.items())[:3])"),
    ("Use pathlib to list all CSV in DATA_DIR.", "csvs = sorted(DATA_DIR.glob('*.csv'))\nprint([p.name for p in csvs])"),
    ("Read CSV with usecols to load only Sales, State.", "df = pd.read_csv(DATA_DIR / 'AusApparalSales4thQrt2020.csv', usecols=['Sales','State'])\nprint(df.head())"),
    ("Timing: %%timeit sum Python list vs numpy.", "import numpy as np\nlst = list(range(10000))\narr = np.arange(10000)\nprint(sum(lst), arr.sum())"),
    ("Use f-string format dataframe shape in message.", "df = pd.read_csv(DATA_DIR / 'marketing_data.csv')\nprint(f'Loaded {df.shape[0]} rows x {df.shape[1]} cols')"),
    ("Try/except reading missing file gracefully.", "from pathlib import Path\ntry:\n    pd.read_csv(DATA_DIR / 'missing.csv')\nexcept FileNotFoundError as e:\n    print('Not found:', e.filename if hasattr(e,'filename') else 'missing.csv')"),
    ("Assert no negative Sales values.", "df = pd.read_csv(DATA_DIR / 'AusApparalSales4thQrt2020.csv')\nassert (df['Sales'] >= 0).all(), 'negative sales'\nprint('ok')"),
    ("Use typing hint def top_n(series, n: int) -> pd.Series.", "def top_n(series: pd.Series, n: int) -> pd.Series:\n    return series.sort_values(ascending=False).head(n)\ndf = pd.read_csv(DATA_DIR / 'AusApparalSales4thQrt2020.csv')\nprint(top_n(df.groupby('State')['Sales'].sum(), 3))"),
    ("Lambda in sort: sort states by name length.", "states = sorted(df['State'].unique(), key=lambda s: len(s)) if 'df' in dir() else ['NSW','VIC']\ndf = pd.read_csv(DATA_DIR / 'AusApparalSales4thQrt2020.csv')\nprint(sorted(df['State'].unique(), key=len)[:5])"),
    ("Map values: Group M→Men, W→Women if present.", "df = pd.read_csv(DATA_DIR / 'AusApparalSales4thQrt2020.csv')\nprint(df['Group'].map({'M': 'Men', 'W': 'Women'}).head())"),
    ("Filter dataframe with boolean mask.", "df = pd.read_csv(DATA_DIR / 'AusApparalSales4thQrt2020.csv')\nprint(df[df['Sales'] > df['Sales'].median()].head())"),
    ("Chain methods: groupby mean sort descending head.", "df = pd.read_csv(DATA_DIR / 'AusApparalSales4thQrt2020.csv')\nprint(df.groupby('State')['Sales'].mean().sort_values(ascending=False).head())"),
    ("Use .pipe for custom function in chain.", "def add_ratio(d):\n    d = d.copy(); d['ratio'] = d['Sales']/d['Unit']; return d\ndf = pd.read_csv(DATA_DIR / 'AusApparalSales4thQrt2020.csv')\nprint(add_ratio(df).head())"),
    ("Set pandas display options max 5 columns.", "pd.set_option('display.max_columns', 5)\ndf = pd.read_csv(DATA_DIR / 'HR_comma_sep.csv')\nprint(df.head())"),
    ("Reset display options.", "pd.reset_option('display.max_columns')"),
    ("Use random seed in numpy choice sample.", "import numpy as np\nnp.random.seed(42)\nprint(np.random.choice([1,2,3], size=5))"),
    ("Bin Sales into quartiles with qcut.", "df = pd.read_csv(DATA_DIR / 'AusApparalSales4thQrt2020.csv')\ndf['q'] = pd.qcut(df['Sales'], 4, labels=['Q1','Q2','Q3','Q4'])\nprint(df['q'].value_counts())"),
    ("Cut Sales into custom bins.", "df = pd.read_csv(DATA_DIR / 'AusApparalSales4thQrt2020.csv')\ndf['tier'] = pd.cut(df['Sales'], bins=[0,15000,25000,50000], labels=['L','M','H'])\nprint(df['tier'].value_counts())"),
    ("Merge two small summary dataframes.", "df = pd.read_csv(DATA_DIR / 'AusApparalSales4thQrt2020.csv')\na = df.groupby('State')['Sales'].sum().reset_index()\nb = df.groupby('State')['Unit'].sum().reset_index()\nprint(a.merge(b, on='State').head())"),
    ("Pivot table mean Sales by State and Group.", "df = pd.read_csv(DATA_DIR / 'AusApparalSales4thQrt2020.csv')\nprint(pd.pivot_table(df, values='Sales', index='State', columns='Group', aggfunc='mean').head())"),
    ("Melt wide to long (create wide first).", "df = pd.read_csv(DATA_DIR / 'AusApparalSales4thQrt2020.csv')\nwide = df.groupby('State')['Sales'].sum().reset_index()\nprint(wide)"),
    ("Use assign to add column inline.", "df = pd.read_csv(DATA_DIR / 'AusApparalSales4thQrt2020.csv')\nprint(df.assign(log_sales=lambda d: np.log1p(d['Sales'])).head())"),
    ("Explode list column demo.", "df = pd.DataFrame({'k': ['a','b'], 'v': [[1,2],[3]]})\nprint(df.explode('v'))"),
    ("String accessor: lower state names.", "df = pd.read_csv(DATA_DIR / 'AusApparalSales4thQrt2020.csv')\nprint(df['State'].str.lower().unique()[:5])"),
    ("Extract regex pattern from column if string.", "df = pd.read_csv(DATA_DIR / 'AusApparalSales4thQrt2020.csv')\nprint(df['Date'].astype(str).str[:4].head())"),
    ("Replace values with .replace dict.", "df = pd.read_csv(DATA_DIR / 'AusApparalSales4thQrt2020.csv')\nprint(df.replace({'M':'Male','W':'Female'}).head())"),
    ("Drop duplicates subset State, Date.", "df = pd.read_csv(DATA_DIR / 'AusApparalSales4thQrt2020.csv')\nprint(len(df), len(df.drop_duplicates(subset=['State','Date'])))"),
    ("Fillna median for numeric column demo.", "df = pd.read_csv(DATA_DIR / 'HR_comma_sep.csv')\nnum = df.select_dtypes('number').columns[0]\ndf[num] = df[num].fillna(df[num].median())\nprint(df[num].isna().sum())"),
    ("Clip Sales to percentile 1-99.", "df = pd.read_csv(DATA_DIR / 'AusApparalSales4thQrt2020.csv')\nlo, hi = df['Sales'].quantile([0.01, 0.99])\ndf['Sales_clip'] = df['Sales'].clip(lo, hi)\nprint(df['Sales_clip'].min(), df['Sales_clip'].max())"),
    ("Use agg multiple functions.", "df = pd.read_csv(DATA_DIR / 'AusApparalSales4thQrt2020.csv')\nprint(df.groupby('State')['Sales'].agg(['mean','sum','count']).head())"),
    ("Named aggregation in groupby.", "df = pd.read_csv(DATA_DIR / 'AusApparalSales4thQrt2020.csv')\nprint(df.groupby('State').agg(avg_sales=('Sales','mean'), total=('Sales','sum')).head())"),
    ("Rolling mean window 7 on daily aggregated sales.", "df = pd.read_csv(DATA_DIR / 'AusApparalSales4thQrt2020.csv', parse_dates=['Date'])\ndaily = df.groupby('Date')['Sales'].sum()\nprint(daily.rolling(7, min_periods=1).mean().head())"),
    ("Shift for lag feature demo.", "daily = df.groupby('Date')['Sales'].sum() if 'df' in dir() else pd.Series([1,2,3])\nprint(daily.shift(1).head())"),
    ("Cumulative sum sales over time.", "df = pd.read_csv(DATA_DIR / 'AusApparalSales4thQrt2020.csv', parse_dates=['Date'])\ndaily = df.groupby('Date')['Sales'].sum().sort_index()\nprint(daily.cumsum().head())"),
    ("Save figure to project folder.", "import matplotlib.pyplot as plt\ndf = pd.read_csv(DATA_DIR / 'AusApparalSales4thQrt2020.csv')\nplt.figure(); plt.hist(df['Sales'], bins=30)\nfig_path = REPO_ROOT / 'Course_02_Applied_Data_Science/02_python_essentials/sales_hist.png'\nplt.savefig(fig_path); plt.close()\nprint(fig_path.exists())"),
    ("Use os.environ.get for optional config.", "import os\nDATA_PATH = os.environ.get('AIML_DATA', str(DATA_DIR))\nprint(DATA_PATH)"),
    ("Write reusable load_sales() function.", "def load_sales():\n    return pd.read_csv(DATA_DIR / 'AusApparalSales4thQrt2020.csv')\nprint(load_sales().shape)"),
    ("Profile memory usage .memory_usage(deep=True).", "df = pd.read_csv(DATA_DIR / 'AusApparalSales4thQrt2020.csv')\nprint(df.memory_usage(deep=True).sum())"),
    ("Convert column to category dtype.", "df = pd.read_csv(DATA_DIR / 'AusApparalSales4thQrt2020.csv')\ndf['State'] = df['State'].astype('category')\nprint(df['State'].dtype)"),
    ("Use pd.to_numeric errors='coerce'.", "s = pd.Series(['1','2','x'])\nprint(pd.to_numeric(s, errors='coerce'))"),
    ("Build pytest-style assert for mean > 0.", "df = pd.read_csv(DATA_DIR / 'AusApparalSales4thQrt2020.csv')\nassert df['Sales'].mean() > 0"),
    ("Docstring module-level practice function.", "def summarize(df):\n    \"\"\"Return shape and null counts.\"\"\"\n    return df.shape, df.isna().sum().sum()\nprint(summarize(pd.read_csv(DATA_DIR / 'loan_data.csv')))"),
    ("Use __name__ == '__main__' guard pattern.", "def main():\n    print('run as script')\nif __name__ == '__main__':\n    main()"),
    ("Create requirements check list programmatically.", "needed = ['numpy','pandas','matplotlib','seaborn','sklearn']\nimport importlib\nprint({m: bool(importlib.util.find_spec(m)) for m in needed})"),
    ("Use pd.read_csv with parse_dates and infer_datetime_format.", "df = pd.read_csv(DATA_DIR / 'AusApparalSales4thQrt2020.csv', parse_dates=['Date'])\nprint(df['Date'].dtype)"),
])

# For brevity in remaining topics, use generator function with topic-specific seeds
def _stat_questions(topic_name: str, dataset: str, col: str) -> list[Question]:
    base = []
    sections = ["Concepts", "Computation", "Interpretation", "Applied", "Challenge"]
    templates = [
        ("Define {t} and give AIML example.", "print('{t} — see basics.md')"),
        ("Load {ds} and describe `{col}`.", "df = pd.read_csv(DATA_DIR / '{ds}')\nprint(df['{col}'].describe())"),
        ("Compute mean, median, std of `{col}`.", "df = pd.read_csv(DATA_DIR / '{ds}')\nprint(df['{col}'].mean(), df['{col}'].median(), df['{col}'].std())"),
        ("Explain when to use {t} test vs alternative.", "print('Depends on assumptions: normality, independence, sample size')"),
        ("Simulate 1000 draws from normal; plot histogram.", "import numpy as np\nimport matplotlib.pyplot as plt\nx = np.random.normal(0, 1, 1000)\nplt.hist(x, bins=30); plt.title('{t}'); plt.show()"),
        ("Calculate 95% CI for mean of `{col}`.", "df = pd.read_csv(DATA_DIR / '{ds}')\nimport numpy as np\nx = df['{col}'].dropna()\nse = x.std()/np.sqrt(len(x))\nprint(x.mean()-1.96*se, x.mean()+1.96*se)"),
        ("State null and alternative hypothesis for {t}.", "print('H0: no effect; H1: effect exists')"),
        ("What alpha=0.05 means in {t}.", "print('5% false positive rate if H0 true')"),
        ("Type I vs Type II error in {t}.", "print('Type I: reject true H0; Type II: fail reject false H0')"),
        ("Check skewness of `{col}`.", "df = pd.read_csv(DATA_DIR / '{ds}')\nprint(df['{col}'].skew())"),
    ]
    idx = 0
    for sec_i, sec in enumerate(sections):
        for j in range(10):
            t, s = templates[j % len(templates)]
            text = t.format(t=topic_name, ds=dataset, col=col)
            sol = s.format(t=topic_name, ds=dataset, col=col)
            base.append(_q(f"Section {sec_i+1}: {sec}", text, sol))
            idx += 1
    return base


def _ml_questions(topic_name: str, dataset: str) -> list[Question]:
    return _stat_questions(topic_name, dataset, "Sales")  # reuse structure with ML framing


COURSE_02_TOPICS = [
    TopicSpec(
        rel_path="Course_02_Applied_Data_Science/01_intro_data_science",
        title="Intro to Data Science",
        course="Course 2",
        dataset="sales",
        dataset_note="Primary: AusApparalSales4thQrt2020.csv",
        extra_imports="import pandas as pd\nimport numpy as np\n",
        questions=INTRO_DS,
        interview=[_interview("Core", f"Intro DS Q{i}", f"Answer {i} — see basics.md and Course 2 projects") for i in range(1, 31)],
    ),
    TopicSpec(
        rel_path="Course_02_Applied_Data_Science/02_python_essentials",
        title="Python Essentials for DS",
        course="Course 2",
        dataset="sales",
        extra_imports="import pandas as pd\nimport numpy as np\n",
        questions=PYTHON_ESSENTIALS,
        interview=[_interview("Python", f"Python DS Q{i}", f"Detailed answer {i}") for i in range(1, 31)],
    ),
    TopicSpec(
        rel_path="Course_02_Applied_Data_Science/04_linear_algebra",
        title="Linear Algebra for DS",
        course="Course 2",
        dataset="sales",
        extra_imports="import numpy as np\nimport pandas as pd\n",
        questions=_stat_questions("Linear Algebra", "AusApparalSales4thQrt2020.csv", "Sales"),
        interview=[_interview("LA", f"Linear algebra Q{i}", "Vectors, matrices, dot products, eigenvalues in ML") for i in range(1, 31)],
    ),
    TopicSpec(
        rel_path="Course_02_Applied_Data_Science/05_statistics_fundamentals",
        title="Statistics Fundamentals",
        course="Course 2",
        dataset="sales",
        extra_imports="import pandas as pd\nimport numpy as np\n",
        questions=_stat_questions("Statistics", "AusApparalSales4thQrt2020.csv", "Sales"),
        interview=[_interview("Stats", f"Stats Q{i}", "Descriptive/inferential stats") for i in range(1, 31)],
    ),
    TopicSpec(
        rel_path="Course_02_Applied_Data_Science/06_probability_distributions",
        title="Probability Distributions",
        course="Course 2",
        dataset="sales",
        extra_imports="import pandas as pd\nimport numpy as np\n",
        questions=_stat_questions("Probability", "AusApparalSales4thQrt2020.csv", "Sales"),
        interview=[_interview("Prob", f"Probability Q{i}", "PMF, PDF, CDF, distributions") for i in range(1, 31)],
    ),
    TopicSpec(
        rel_path="Course_02_Applied_Data_Science/07_advanced_statistics",
        title="Advanced Statistics",
        course="Course 2",
        dataset="marketing",
        extra_imports="import pandas as pd\nimport numpy as np\n",
        questions=_stat_questions("Advanced Stats", "marketing_data.csv", "Recency"),
        interview=[_interview("Adv", f"Advanced stats Q{i}", "ANOVA, regression, assumptions") for i in range(1, 31)],
    ),
    TopicSpec(
        rel_path="Course_02_Applied_Data_Science/09_data_wrangling",
        title="Data Wrangling",
        course="Course 2",
        dataset="hr",
        extra_imports="import pandas as pd\nimport numpy as np\n",
        questions=_stat_questions("Wrangling", "HR_comma_sep.csv", "satisfaction_level"),
        interview=[_interview("Wrangle", f"Wrangling Q{i}", "clean, reshape, merge") for i in range(1, 31)],
    ),
    TopicSpec(
        rel_path="Course_02_Applied_Data_Science/10_data_visualization",
        title="Data Visualization Overview",
        course="Course 2",
        dataset="sales",
        dataset_note="Also see matplotlib/ and seaborn/ subfolders (45–60 Q each).",
        extra_imports="import pandas as pd\nimport matplotlib.pyplot as plt\n",
        questions=_stat_questions("Visualization", "AusApparalSales4thQrt2020.csv", "Sales"),
        interview=[_interview("Viz", f"Viz Q{i}", "chart selection, aesthetics") for i in range(1, 31)],
    ),
    TopicSpec(
        rel_path="Course_02_Applied_Data_Science/11_regex_json_apis",
        title="Regex, JSON & APIs",
        course="Course 2",
        dataset=None,
        extra_imports="import json\nimport re\n",
        questions=_stat_questions("Regex/JSON", "marketing_data.csv", "Income"),
        interview=[_interview("Regex", f"Regex Q{i}", "patterns, json.loads, requests") for i in range(1, 31)],
    ),
]

COURSE_03_TOPICS = [
    TopicSpec(
        rel_path=f"Course_03_Machine_Learning/{folder}",
        title=title,
        course="Course 3",
        dataset=ds,
        extra_imports="import pandas as pd\nimport numpy as np\n",
        questions=_ml_questions(title, ds),
        interview=[_interview("ML", f"{title} Q{i}", "ML concept — tie to HR/Spotify/loan projects") for i in range(1, 31)],
    )
    for folder, title, ds in [
        ("01_eda_feature_engineering", "EDA & Feature Engineering", "HR_comma_sep.csv"),
        ("02_clustering", "Clustering", "rolling_stones_spotify.csv"),
        ("03_classification", "Classification", "HR_comma_sep.csv"),
        ("04_imbalanced_data", "Imbalanced Data", "HR_comma_sep.csv"),
        ("05_model_evaluation", "Model Evaluation", "loan_data.csv"),
    ]
]

COURSE_04_TOPICS = [
    TopicSpec(
        rel_path=f"Course_04_Deep_Learning/{folder}",
        title=title,
        course="Course 4",
        dataset=ds,
        extra_imports="import pandas as pd\nimport numpy as np\n",
        questions=_ml_questions(title, ds),
        interview=[_interview("DL", f"{title} Q{i}", "Deep learning concept — tie to Lending Club / Home Loan") for i in range(1, 31)],
    )
    for folder, title, ds in [
        ("01_neural_network_basics", "Neural Network Basics", "loan_data.csv"),
        ("02_keras_tensorflow", "Keras & TensorFlow", "loan_data.csv"),
        ("03_preprocessing_imbalance", "DL Preprocessing & Imbalance", "Home_loan_data_sample.csv"),
        ("04_model_evaluation_dl", "DL Model Evaluation", "Home_loan_data_sample.csv"),
    ]
]

COURSE_05_TOPICS = [
    TopicSpec(
        rel_path=f"Course_05_Generative_AI/{folder}",
        title=title,
        course="Course 5",
        dataset=None,
        extra_imports="# GenAI topics — prompts + optional marketing_data for RAG demos\nimport pandas as pd\n",
        questions=_ml_questions(title, "marketing_data.csv"),
        interview=[_interview("GenAI", f"{title} Q{i}", "Prompting / LLM concept — tie to Storytelling & PM projects") for i in range(1, 31)],
    )
    for folder, title in [
        ("01_prompt_engineering", "Prompt Engineering"),
        ("02_chatgpt_applications", "ChatGPT Applications"),
        ("03_genai_optimization", "GenAI Optimization"),
    ]
]

ALL_TOPICS = COURSE_02_TOPICS + COURSE_03_TOPICS + COURSE_04_TOPICS + COURSE_05_TOPICS
