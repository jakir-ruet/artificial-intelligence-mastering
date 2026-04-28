## More About Me – [Take a Look!](http://www.mjakaria.me)

### Install

```bash
python -m venv venv
source venv/bin/activate
```

```bash
pip install notebook
jupyter notebook # if not work then
jupyter notebook --allow-root
```

### Pandas

Pandas is a powerful Python library used for data analysis and manipulation. It’s especially popular in data science, machine learning, and analytics workflows. Pandas is like combination of `Excel + SQL + Python`.

```python
import pandas as pd

data = {'name': ['A', 'B'], 'age': [20, 25]}
df = pd.DataFrame(data)
print(df)
```

**Key Features:**

- Provides efficient data structures:
  - Series (1D)
  - DataFrame (2D table)
- Helps in:
  - Data cleaning (handling missing values, formatting)
  - Data transformation
  - Aggregation (sum, mean, group by)
  - Basic visualization (often with Matplotlib)
- Works well with large datasets

👉 In short: Pandas = Excel-like data handling in Python

### Pandas Data Structures

| Feature              | Series                                      | DataFrame                                        |
| -------------------- | ------------------------------------------- | ------------------------------------------------ |
| **Dimension**        | 1D (one-dimensional)                        | 2D (rows & columns)                              |
| **Structure**        | Single column                               | Table (multiple columns)                         |
| **Data Type**        | Usually homogeneous (can be mixed)          | Heterogeneous (different types per column)       |
| **Labels**           | Index only                                  | Index (rows) + Columns                           |
| **Size Mutability**  | Limited (can grow/shrink but less flexible) | Fully mutable (add/remove rows & columns easily) |
| **Value Mutability** | Yes                                         | Yes                                              |
| **Example Use**      | Single feature / column                     | Full dataset / table                             |
| **Creation Example** | `pd.Series([1,2,3])`                        | `pd.DataFrame({...})`                            |
| **Analogy**          | Excel column                                | Excel sheet                                      |

> A Series is like a single column in Excel.
> A DataFrame is like a table Excel.

### Jupyter

Jupyter is a web-based interactive computing platform used to write and run code.

**Key Features:**

- Allows you to:
  - Write Python code in small blocks (cells)
  - Run code step-by-step
  - See output instantly
- Supports:
  - Code
  - Text (Markdown)
  - Visualizations
- Widely used in:
  - Data Science
  - Machine Learning
  - Research & prototyping

👉 In short: Jupyter = Interactive coding environment in browser

### Jupyter Notebook?

A Jupyter Notebook is a file with extension `.ipynb`

**Characteristics:**

- Runs on a Jupyter Notebook server
- Contains:
  - Code cells (Python)
  - Markdown cells (text, notes, documentation)
  - Output (tables, charts, results)
- Can be:
  - Edited
  - Executed
  - Shared easily

👉 Example uses:

- Data analysis
- Machine learning experiments
- Tutorials and reports

### Comparison

| Feature        | Pandas              | Jupyter                 |
| -------------- | ------------------- | ----------------------- |
| Type           | Python Library      | Tool / Environment      |
| Purpose        | Data manipulation   | Run and explore code    |
| Output         | DataFrames, results | Interactive notebooks   |
| Usage Together | Yes (very common)   | Used to run Pandas code |

### Data Cleaning

Data cleaning is the process of:

- Correcting incorrect data
- Handling missing values
- Removing duplicates
- Fixing corrupt or inconsistent data

> Simple idea: Clean data = reliable analysis

### Why is Data Cleaning Important?

- `Garbage In (GI) = Garbage Out (GO)`. If your input data is bad → your output (analysis/model) will also be bad.

#### Benefits:

- Better data analysis accuracy
- Improved decision making
- Increased productivity & revenue
- More efficient business processes

#### Data Cleaning Process - `Data Importing, Merging & Exploring`

- Load data from multiple sources
- Combine datasets
- Understand structure

```python
import pandas as pd

df = pd.read_csv("data.csv")
df.head()
df.info()
```

### Data Filtering

- Remove irrelevant or invalid data
- Apply conditions

```python
df = df[df['age'] > 0]
```

### Data Cleaning & Transformation

- Handle missing values
- Remove duplicates
- Fix formats
- Feature engineering

### Missing data means:

- Empty values
- NaN (Not a Number)

> Problem: Many ML algorithms cannot handle missing data

### Causes of Missing Data

- Human error
- Sensor failure
- Data corruption
- Software bugs

### Correlation Analysis

Correlation analysis is a statistical technique used to measure:

- Strength of relationship
- Direction of relationship

> Between `two` or `more` variables

#### Correlation Coefficient

A numerical value that represents correlation.

> Range:
> - +1 → Strong positive
> - 0 → No relationship
> - -1 → Strong negative

#### Types of Correlation

| Feature                 | Pearson Correlation  | Spearman Correlation              |
| ----------------------- | -------------------- | --------------------------------- |
| Data Type               | Continuous (numeric) | Ordinal / Ranked data             |
| Relationship Type       | Linear relationship  | Monotonic (can be non-linear)     |
| Sensitivity to Outliers | Sensitive            | Robust                            |
| Measurement Basis       | Actual values        | Rank (order) of values            |
| Use Case                | Height vs Weight     | Rankings, Ratings                 |
| Example                 | Salary vs Experience | Student rank vs performance level |

#### Dealing with Categorical Features

##### Label Encoding

Concept: Assign numbers to categories

```bash
USA → 0
India → 1
Japan → 2
```

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['country'] = le.fit_transform(df['country'])
```

##### One-Hot Encoding (Recommended)

Concept: Create binary columns

| Country_USA | Country_India | Country_Japan |
| ----------- | ------------- | ------------- |
| 1           | 0             | 0             |

```python
df = pd.get_dummies(df, columns=['country'])
```

#### Label vs One-Hot

| Feature     | Label Encoding | One-Hot Encoding |
| ----------- | -------------- | ---------------- |
| Output      | Single column  | Multiple columns |
| Order issue | Yes            | No               |
| Use case    | Ordinal data   | Nominal data     |

### Plotting Using Pandas

Pandas has built-in plotting (powered by Matplotlib) so you can visualize data quickly.

#### Plot Types in Pandas

| Plot Type | Use Case            | Example          | Best Fit / When to Use                         |
| --------- | ------------------- | ---------------- | ---------------------------------------------- |
| Line      | Trends over time    | Sales growth     | Time series data (daily, monthly, yearly)      |
| Bar       | Category comparison | Dept vs salary   | Comparing discrete categories                  |
| Histogram | Distribution        | Age distribution | Understanding data spread & frequency          |
| Box       | Outliers & spread   | Salary analysis  | Detecting outliers & data variability          |
| Scatter   | Relationship        | Age vs salary    | Checking correlation between two variables     |
| Area      | Cumulative trends   | Multi-year sales | Showing cumulative totals over time            |
| Pie       | Proportion          | Dept percentage  | Showing percentage share (few categories only) |

## With Regards, `Jakir`

[![LinkedIn][linkedin-shield-jakir]][linkedin-url-jakir]
[![Facebook-Page][facebook-shield-jakir]][facebook-url-jakir]
[![Youtube][youtube-shield-jakir]][youtube-url-jakir]

### Wishing you a wonderful day! Keep in touch

<!-- Personal profile -->

[linkedin-shield-jakir]: https://img.shields.io/badge/linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white
[linkedin-url-jakir]: https://www.linkedin.com/in/jakir-ruet/
[facebook-shield-jakir]: https://img.shields.io/badge/Facebook-%231877F2.svg?style=for-the-badge&logo=Facebook&logoColor=white
[facebook-url-jakir]: https://www.facebook.com/jakir.ruet/
[youtube-shield-jakir]: https://img.shields.io/badge/YouTube-%23FF0000.svg?style=for-the-badge&logo=YouTube&logoColor=white
[youtube-url-jakir]: https://www.youtube.com/@mjakaria-ruet/featured
