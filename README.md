# 🚗 Car Selling Price Prediction
### Linear Regression with Gradient Descent — from scratch

> Machine Learning Project 1 | 2025–2026 Spring

---

## 📌 About

Predicts used car selling prices using a **linear regression model built entirely from scratch** — no scikit-learn, no ML libraries. Only `numpy` and `pandas`.

The model is trained with **gradient descent** and evaluated across multiple learning rates.

---

## 📊 Dataset

| | Train | Test |
|---|---|---|
| Rows | 3,471 | 869 |
| Features | 11 | 11 |
| Target | Selling Price (INR) | Selling Price (INR) |

**Features used:** `year`, `km_driven`, `fuel`, `seller_type`, `transmission`, `owner`

---

## ⚙️ Data Preprocessing

| Column | Method | Reason |
|---|---|---|
| `name` | Dropped | Text label, no numeric value |
| `year`, `km_driven` | MinMax Scaling | Large value ranges |
| `fuel`, `seller_type` | One-Hot Encoding | No natural order |
| `transmission` | Binary (0/1) | Only 2 values |
| `owner` | Ordinal (0–4) | Natural order exists |
| `selling_price` | MinMax Scaling | Stabilizes gradient descent |

---

## 🧮 Model

**Hypothesis:**
```
h(x) = Xθ = θ₀x₀ + θ₁x₁ + ... + θₙxₙ
```

**Cost Function:**
```
J(θ) = 1/(2m) × Σ(h(x) - y)²
```

**Gradient Descent Update:**
```
θ := θ - α · ∇J(θ)     where     ∇J(θ) = 1/m · Xᵀ(Xθ - y)
```

**Convergence criterion:** `|J(t-1) - J(t)| < 1e-6`

---

## 📈 Results

| Learning Rate (α) | Iterations | Final Cost | Test MSE |
|:-:|:-:|:-:|:-:|
| 0.001 | 628 | 0.002499 | 0.004896 |
| 0.01 | 319 | 0.001819 | 0.003789 |
| 0.05 | 207 | 0.001511 | 0.003353 |
| **0.1** | **152** | **0.001443** | **0.003246** |

### Best Model (α = 0.1)

| Metric | Scaled | Real INR |
|---|---|---|
| MSE | 0.003246 | ~1.2M |
| RMSE | 0.057 | ~463,000 |
| MAE | 0.029 | ~232,000 |
| R² | — | 0.3415 |

---

## 🔍 Key Findings

- **Transmission** is the strongest price predictor (θ = +0.086) — automatic cars cost significantly more
- **Year** has moderate positive impact (θ = +0.059) — newer cars are more expensive
- **km_driven** has negative impact (θ = -0.0085) — higher mileage reduces price
- **Fuel type** has minor effects — Diesel carries a small premium

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat)

---

## 🚀 Run

```bash
git clone https://github.com/yourusername/car-price-prediction
cd car-price-prediction
pip install numpy pandas matplotlib
python linear_regression_gd.py
```

---

## 📁 Files

```
├── linear_regression_gd.py   # Main model
├── trainDATA.csv             # Training data
├── testDATA.csv              # Test data
└── README.md
```
