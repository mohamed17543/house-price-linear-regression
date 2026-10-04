# House Price Prediction using Linear Regression

## Project Overview
A beginner-friendly Machine Learning project that predicts house sale prices using Linear Regression. Built with the Ames Housing dataset and scikit-learn. Every important step is explained in the notebook.

## Dataset
- **File:** `data/train.csv`
- **Rows:** 2,051 houses
- **Columns:** 81 (80 features + 1 target)
- **Target:** `SalePrice`
- **Source:** Ames Housing dataset (Iowa, USA)

## Problem Statement
Given features describing a house (size, quality, location, year built, etc.), predict its sale price in dollars.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## Project Workflow
```
Load Dataset → Inspect → EDA → Preprocessing → Train Model → Evaluate → Analyze
```

## Exploratory Data Analysis
- SalePrice is right-skewed (skewness = 1.56)
- `Overall_Qual` is the strongest predictor (r = 0.80)
- `Gr_Liv_Area` (r = 0.70) — larger living area = higher price
- Neighborhood has a large effect on price
- 26 columns have missing values (most mean "feature doesn't exist")

## Data Preprocessing
- Removed identifier columns (`Id`, `PID`)
- Converted `MS_SubClass` from integer to category
- Split data into train/test **before** any preprocessing
- Numerical: `SimpleImputer(strategy="median")`
- Categorical: `SimpleImputer(most_frequent)` + `OneHotEncoder`
- Everything wrapped in a `ColumnTransformer` + `Pipeline`

## Leakage Prevention
- Train/test split happens **before** fitting any transformation
- All imputation statistics and encodings are learned from training data only
- Test set remains completely unseen until final evaluation
- `handle_unknown="ignore"` protects against unseen categories at predict time

## Machine Learning Model
- **Model:** `LinearRegression` (scikit-learn)
- **Pipeline:** Preprocessor → LinearRegression
- No absolute paths — project works on any computer

## Evaluation Metrics

| Metric | Value | Meaning |
|---|---|---|
| MAE | $14,816 | Average prediction error |
| RMSE | $21,132 | Penalizes large errors more |
| R² | 0.9248 | Model explains 92.5% of price variance |

## Results
The model achieves **R² = 0.9248**, meaning it explains 92.5% of the variance in house prices. On average, predictions are off by about **$14,816** (MAE).

## Visualizations

| Chart | Description |
|---|---|
| `01_saleprice_distribution.png` | SalePrice histogram + log transform |
| `02_missing_values.png` | Missing values by column |
| `03_correlation_heatmap.png` | Correlation between top features |
| `04_feature_vs_saleprice.png` | Key features vs SalePrice |
| `05_categorical_vs_saleprice.png` | Neighborhood and zoning effects |
| `06_outlier_boxplots.png` | Outlier detection |
| `07_actual_vs_predicted.png` | Actual vs Predicted prices |
| `08_residuals.png` | Residual plot |
| `09_residual_distribution.png` | Residual distribution |
| `10_coefficients.png` | Top 20 model coefficients |

## Limitations
- Linear Regression assumes linear relationships
- Right-skewed target — a log transform could improve results
- Large errors on very expensive houses
- Coefficient sizes depend on feature units — direct comparison is misleading
- Correlation ≠ Causation

## Future Improvements
- Apply `log1p` transform to `SalePrice`
- Try `Ridge` or `Lasso` regression
- Engineer new features (e.g., total area = basement + floors)
- Try tree-based models (Random Forest, Gradient Boosting)

## How to Run
```bash
# 1. Clone the repo
git clone https://github.com/mohamed17543/house-price-linear-regression.git
cd house-price-linear-regression

# 2. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install requirements
pip install -r requirements.txt

# 4. Start Jupyter
jupyter notebook notebooks/house_price_linear_regression.ipynb
```
Run the notebook from the `notebooks/` folder — the dataset path is `../data/train.csv`.

## Requirements
```
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

## Author
Built by a Computer Science student as a beginner portfolio project in Machine Learning.
