# Homework 3 - Survival Analysis

Course: DS 223 | Marketing Analytics
Student: Milena Sargsyan

## Project Overview

Survival analysis on a telecom customer churn dataset using Accelerated Failure Time (AFT) models and Customer Lifetime Value (CLV) estimation.

## Project Structure

```
Survival_Analysis/
│
├── data/
│   └── telco.csv
│
├── img/
│   └── generated plots and visualizations
│
├── survival_analysis.ipynb
├── requirements.txt
└── README.md
```

## Objective

The goal of this project is to analyze customer churn behavior using survival analysis techniques and estimate Customer Lifetime Value (CLV) for telecom subscribers.

## Files

- `survival_analysis.ipynb` — complete analysis, modeling, visualizations, and report
- `data/` — telecom churn dataset
- `img/` — saved plots and figures
- `requirements.txt` — required Python libraries

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook survival_analysis.ipynb
```

## Summary of Findings

Several AFT survival models were trained and compared. The Weibull AFT model was selected as the final model based on interpretability and model performance. The analysis identified significant factors affecting churn risk and estimated customer lifetime value across different customer segments. High-value and high-risk customer groups were also identified to support retention strategy decisions.
