# Customer Churn Prediction

Predict which bank customers are about to leave — using Random Forest, XGBoost, and LightGBM — with a rule-based system for business teams.

---

## Dataset

The dataset has **10,000 rows** and **14 columns**:

| Column | Description |
|--------|-------------|
| `CreditScore` | Customer's credit score (350–850) |
| `Geography` | Country: France, Spain, or Germany |
| `Gender` | Male / Female |
| `Age` | Customer age |
| `Tenure` | Years as a customer (0–10) |
| `Balance` | Account balance |
| `NumOfProducts` | Number of bank products held |
| `HasCrCard` | Owns a credit card (0/1) |
| `IsActiveMember` | Active in the last period (0/1) |
| `EstimatedSalary` | Annual salary estimate |
| `Exited` | **Target** — did they churn? (0=No, 1=Yes) |

---

## Tasks Covered

- Load & explore the dataset
- Clean data (handle nulls, drop ID columns)
- Encode categorical features (Label + One-Hot)
- Feature engineering (AgeGroup, BalancePerProduct, IsHighRisk)
- Train 3 ML models (RF, XGBoost, LightGBM)
- Compare using Accuracy, ROC-AUC, F1 Score
- Visualize ROC curves and confusion matrices
- Build a rule-based churn detection system
- Explain which model performs best and why

---

## Models Used

### 1. Random Forest
Builds many decision trees independently and takes a majority vote. Robust, reliable, and hard to overfit. Great starting point.

### 2. XGBoost
Gradient boosting — each new tree learns from the mistakes of the previous one. Usually very accurate on structured/tabular data.

### 3. LightGBM
Microsoft's faster gradient boosting framework. Grows trees leaf-wise instead of level-wise, making it more efficient without sacrificing accuracy.

### 4. Rule-Based System 
- Germany customers → higher risk
- Older + inactive → high risk
- 3+ products → suspicious churn pattern
- Low credit + inactive → flag for retention

---

## Key Insights from EDA

- **Germany** has the highest churn rate (~27%) vs France/Spain (~16%)
- **Customers 45+** are significantly more likely to churn
- **Inactive members** have nearly 2× the churn rate of active ones
- **3–4 products** → surprisingly high churn (possibly over-sold?)
- **Credit score** shows weaker individual correlation, but matters in combination

---

## Conclusion

### Model Comparison Summary

| Model | Accuracy | ROC-AUC | F1 Score |
|-------|----------|---------|----------|
| **Random Forest** | 84.20% | 0.8609 | 0.6291 |
| XGBoost | 83.85% | 0.8598 | 0.6325 |
| LightGBM | 81.30% | 0.8578 | 0.6088 |
| Rule-Based | ~78% | N/A | ~0.55 |


---

### Recommendation: Random Forest

**Random Forest performs best** for the following reasons:

1. **Highest Accuracy** — 84.20%, best overall correctness across the test set
2. **Highest ROC-AUC** — 0.8609, best at distinguishing churners from non-churners
3. **Strong F1 Score** — 0.6291, good balance of precision and recall for churners
4. **Handles imbalance** — class_weight='balanced' helps catch more churners

**XGBoost is a close second** with a slightly better F1 (0.6325) and may be preferred when catching churners matters more than overall accuracy.

### Rule-Based System 

- Fully transparent — every decision is explainable to management
- Easy to update — just change the thresholds
- No training required — works immediately on new customer profiles
- Lower recall — misses subtler churn patterns

**Best approach:** Use LightGBM for prediction, and the rule-based system for generating human-readable explanations ("Why did we flag this customer?").

---

