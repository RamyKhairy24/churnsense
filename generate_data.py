"""
generate_data.py
Generates an enhanced simulated churn dataset with behavioral features,
trains multiple models, picks the best, saves model.pkl + model_meta.json.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).parent
np.random.seed(42)
N = 3000

# ── Categorical features ───────────────────────────────────────────────────────
genders        = np.random.choice(['Male', 'Female'], N)
countries      = np.random.choice(['Canada', 'Egypt', 'Germany', 'India', 'UK', 'USA'], N)
sub_types      = np.random.choice(['Basic', 'Free', 'Premium'], N, p=[0.35, 0.35, 0.30])
contract_types = np.random.choice(['Monthly', 'Annual'], N, p=[0.60, 0.40])

# ── Numeric features ───────────────────────────────────────────────────────────
tenure = np.random.randint(1, 73, N)

monthly_charge = np.round(np.where(
    sub_types == 'Free',    np.random.uniform(0,  15,  N),
    np.where(sub_types == 'Basic', np.random.uniform(20, 50, N),
                                   np.random.uniform(50, 100, N))
), 2)

support_tickets   = np.random.poisson(1.5, N).clip(0, 8).astype(int)
days_since_login  = np.round(np.random.exponential(22, N)).clip(0, 365).astype(int)
payment_failures  = np.random.poisson(0.4, N).clip(0, 5).astype(int)

# ── Churn logit  (intercept tuned so churn rate ≈ 50%) ────────────────────────
logit = (
    -0.040 * tenure
    +  0.022 * monthly_charge
    +  0.42  * support_tickets
    +  0.012 * days_since_login
    +  1.70  * (contract_types == 'Monthly').astype(float)
    +  0.70  * payment_failures
    +  0.40  * (sub_types == 'Free').astype(float)
    -  0.30  * (sub_types == 'Premium').astype(float)
    +  np.random.normal(0, 0.7, N)
    -  1.50   # intercept
)

prob  = 1 / (1 + np.exp(-logit))
churn = (np.random.uniform(0, 1, N) < prob).astype(int)

df = pd.DataFrame({
    'Gender':             genders,
    'Country':            countries,
    'SubscriptionType':   sub_types,
    'ContractType':       contract_types,
    'TenureMonths':       tenure,
    'MonthlyCharge':      monthly_charge,
    'SupportTickets':     support_tickets,
    'DaysSinceLastLogin': days_since_login,
    'PaymentFailures':    payment_failures,
    'Churn':              churn,
})

print(f"Dataset       : {df.shape}")
print(f"Churn rate    : {df['Churn'].mean():.1%}")

# Save enhanced CSV
csv_path = BASE / 'Simulated Churn Data' / 'Simulated Customer Data.csv'
df.to_csv(csv_path, index=False)
print(f"CSV saved     : {csv_path}")

# ── Preprocessing ──────────────────────────────────────────────────────────────
CATEGORICAL = ['Gender', 'Country', 'SubscriptionType', 'ContractType']
NUMERIC     = ['TenureMonths', 'MonthlyCharge', 'SupportTickets',
               'DaysSinceLastLogin', 'PaymentFailures']
FEATURES    = CATEGORICAL + NUMERIC
TARGET      = 'Churn'

X = df[FEATURES]
y = df[TARGET]

preprocessor = ColumnTransformer([
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL),
    ('scl', StandardScaler(), NUMERIC),
])

# ── Model comparison ───────────────────────────────────────────────────────────
candidates = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    'Gradient Boosting':   GradientBoostingClassifier(
                               n_estimators=200, learning_rate=0.08,
                               max_depth=4, random_state=42),
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

print(f"\n{'Model':<25}  {'CV AUC (mean±std)':<22}  Test AUC")
print('-' * 62)

for name, clf in candidates.items():
    pipe = Pipeline([('pre', preprocessor), ('clf', clf)])
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='roc_auc')
    pipe.fit(X_train, y_train)
    test_auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
    results[name] = {'pipe': pipe, 'cv': cv_scores, 'test_auc': test_auc}
    print(f"{name:<25}  {cv_scores.mean():.4f} ± {cv_scores.std():.4f}       {test_auc:.4f}")

best_name = max(results, key=lambda n: results[n]['test_auc'])
best_pipe  = results[best_name]['pipe']
print(f"\n✓ Best model  : {best_name}  (AUC = {results[best_name]['test_auc']:.4f})")
print()
print(classification_report(y_test, best_pipe.predict(X_test), target_names=['No Churn', 'Churn']))

# ── Retrain on full data ───────────────────────────────────────────────────────
final_pipe = Pipeline([('pre', preprocessor), ('clf', candidates[best_name])])
final_pipe.fit(X, y)

with open(BASE / 'model.pkl', 'wb') as f:
    pickle.dump(final_pipe, f)

# ── Save metadata ──────────────────────────────────────────────────────────────
meta = {
    'model_name':           best_name,
    'features':             FEATURES,
    'categorical_features': CATEGORICAL,
    'numeric_features':     NUMERIC,
    'categories': {col: sorted(df[col].unique().tolist()) for col in CATEGORICAL},
    'numeric_ranges': {
        'TenureMonths':       {'min': 1,   'max': 72,  'default': 24, 'step': 1, 'unit': 'months', 'label': 'Tenure'},
        'MonthlyCharge':      {'min': 0,   'max': 100, 'default': 49, 'step': 1, 'unit': '$',      'label': 'Monthly Charge'},
        'SupportTickets':     {'min': 0,   'max': 8,   'default': 1,  'step': 1, 'unit': '',       'label': 'Support Tickets'},
        'DaysSinceLastLogin': {'min': 0,   'max': 365, 'default': 14, 'step': 1, 'unit': 'days',   'label': 'Days Since Login'},
        'PaymentFailures':    {'min': 0,   'max': 5,   'default': 0,  'step': 1, 'unit': '',       'label': 'Payment Failures'},
    },
}

with open(BASE / 'model_meta.json', 'w') as f:
    json.dump(meta, f, indent=2)

print(f"✓ model.pkl saved")
print(f"✓ model_meta.json saved")
