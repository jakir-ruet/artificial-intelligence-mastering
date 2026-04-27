## More About Me – [Take a Look!](http://www.mjakaria.me)

### Numpy (Numerical Python)

It's a core Python library used for:

- Fast numerical computations
- Working with arrays and matrices
- Performing mathematical operations efficiently

> In a word, NumPy = High-performance alternative to Python lists for numerical data.

### Install

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy
pip install --upgrade pip # if needed
```

### Summary

NumPy is the foundation of data science & machine learning in Python, If you master NumPy;

- Data handling becomes easy
- ML becomes understandable
- Performance improves drastically
- Machine Learning pipelines
- Data preprocessing
- Image processing
- Scientific simulations
- Financial analysis

### NumPy vs Python List

| Feature          | Python List | NumPy     |
| ---------------- | ----------- | --------- |
| Speed            | Slow        | Fast      |
| Memory           | High        | Efficient |
| Math Ops         | Manual      | Built-in  |
| Multidimensional | Hard        | Easy      |

### Main Array (15, 3)

```python
[[11.  8. 12.]
 [ 9. 11. 14.]
 [ 7. 11. 12.]
 [ 9.  8. 12.]
 [12.  7. 10.]
 [ 9.  6. 12.]
 [10.  6.  9.]
 [ 5. 14. 10.]
 [13.  5. 14.]
 [ 7. 11.  8.]
 [13.  7.  9.]
 [ 7. 11.  9.]
 [13. 11.  6.]
 [ 8. 13.  6.]
 [14. 13. 14.]]
```

### Selected Sample (Top 5 Rows)

```python
[[11.  8. 12.]
 [ 9. 11. 14.]
 [ 7. 11. 12.]
 [ 9.  8. 12.]
 [12.  7. 10.]]
```

### Statistical Summary

*Mean Equation: $\mu = \frac{1}{n} \sum_{i=1}^{n} x_i$*

> x_i = 11, 9, 7, 9, ... 14

*Standard Deviation: $\sigma = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (x_i - \mu)^2}$*

Or

*Standard Deviation: $s = \sqrt{\frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2}$*

| Metric                            | x1    | x2    | x3     |
| :-------------------------------- | :---- | :---- | :----- |
| **Mean ($\mu$)**                  | 9.800 | 9.467 | 10.467 |
| **Standard Deviation ($\sigma$)** | 2.663 | 2.802 | 2.579  |
| **Min**                           | 5.0   | 5.0   | 6.0    |
| **Max**                           | 14.0  | 14.0  | 14.0   |

*Normalized Sample (Z-Scores): $z = \frac{x - \mu}{\sigma}$*

### Data Normalization Report (Z-Scores)

#### Column 1 Analysis

Parameters: $\mu = 9.80$ | $\sigma = 2.6633$

| Metric              | Row 1  |  Row 2  |  Row 3  |  Row 4  | Row 5  |
| :------------------ | :----: | :-----: | :-----: | :-----: | :----: |
| **Raw Value ($x$)** | 11.00  |  9.00   |  7.00   |  9.00   | 12.00  |
| **Z-Score ($z$)**   | 0.4506 | -0.3004 | -1.0513 | -0.3004 | 0.8260 |

#### Column 2 Analysis

Parameters: $\mu = 9.4667$ | $\sigma = 2.8016$

| Metric              | Row 1 | Row 2  | Row 3  |  Row 4  |  Row 5  |
| :------------------ | :---: | :----: | :----: | :-----: | :-----: |
| **Raw Value ($x$)** | 8.00  | 11.00  | 11.00  |  8.00   |  7.00   |
| **Z-Score ($z$)**   |   0   | 0.5150 | 0.5150 | -0.5664 | -0.9269 |

#### Column 3 Analysis

Parameters: $\mu = 10.4667$ | $\sigma = 2.5785$

| Metric              | Row 1  | Row 2  | Row 3  | Row 4  |  Row 5  |
| :------------------ | :----: | :----: | :----: | :----: | :-----: |
| **Raw Value ($x$)** | 12.00  | 14.00  | 12.00  | 12.00  |  10.00  |
| **Z-Score ($z$)**   | 0.6453 | 1.4309 | 0.6453 | 0.6453 | -0.1402 |

### Cumulative Distribution Function (CDF) Analysis

The CDF value ($\Phi$) represents the probability that a random variable from the population is less than or equal to our specific value. In practical terms, this tells us the **percentile rank** of each data point.

#### Key Sample Lookups

| Data Point       | Z-Score ($z$) | CDF Calculation $\Phi(z)$ | Percentile (%) | Interpretation                           |
| :--------------- | :-----------: | :-----------------------: | :------------: | :--------------------------------------- |
| **Col 1, Row 1** |    0.4506     |          0.6738           |   **67.38%**   | Above average; higher than ~67% of data. |
| **Col 1, Row 3** |    -1.0513    |          0.1465           |   **14.65%**   | Significantly below average.             |
| **Col 3, Row 2** |    1.4309     |          0.9238           |   **92.38%**   | High outlier; in the top 8% of the data. |

### Full Sample Percentile Table

| Row   | Column 1 (%) | Column 2 (%) | Column 3 (%) |
| :---- | :----------: | :----------: | :----------: |
| **1** |    67.38%    |    50.00%    |    74.06%    |
| **2** |    38.20%    |    69.67%    |    92.38%    |
| **3** |    14.65%    |    69.67%    |    74.06%    |
| **4** |    38.20%    |    28.56%    |    74.06%    |
| **5** |    79.56%    |    17.70%    |    44.42%    |

#### 💡 Technical Note

These values are calculated using the **Standard Normal Cumulative Distribution Function**:
$P(X \le x) = \Phi(z) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{z} e^{-\frac{t^2}{2}} dt$

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
