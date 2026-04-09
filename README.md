# ChurnSense — AI Customer Churn Predictor

A mobile-first progressive web app that predicts customer churn risk using a machine learning pipeline trained on simulated behavioural data.

**Live model: Logistic Regression · AUC 0.79 · 9 features · 3,000 training samples**

---

## Features

- **Instant churn prediction** — fill in a customer profile and get a probability score in real time
- **9-feature model** — categorical and numeric behavioural signals
- **Animated SVG ring gauge** — visual churn probability with colour-coded verdict
- **Risk Signal Analysis** — ranked bar chart of the 4 strongest churn drivers for each prediction
- **Prediction history drawer** — last 10 predictions stored in `localStorage`, with time-ago display
- **Form state restore** — last session's inputs are silently restored on page load
- **Loading skeleton** — pulsing shimmer card while waiting for the API
- **Mobile-first PWA** — works on all screen sizes, supports iPhone Dynamic Island (`viewport-fit=cover`)
- **Dark / light theme** — user preference persisted in `localStorage`
- **Installable** — full PWA manifest + service worker; "Add to Home Screen" prompt on Android
- **Production-hardened backend** — `SECRET_KEY` from env, `PORT`/`FLASK_DEBUG` env vars, input validation, `/health` endpoint

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.12, Flask 3.1.3 |
| ML | scikit-learn 1.6.1 (Pipeline, ColumnTransformer, LogisticRegression) |
| Data | pandas 2.2.3, numpy 2.2.3 |
| Frontend | Vanilla HTML/CSS/JS (no framework) |
| Deployment | Vercel (`@vercel/python`) |
| Production server | Gunicorn 23.0.0 |

---

## Input Features

| Feature | Type | Range / Options |
|---|---|---|
| Gender | Categorical | Female, Male |
| Country | Categorical | Canada, Egypt, Germany, India, UK, USA |
| SubscriptionType | Categorical | Basic, Free, Premium |
| ContractType | Categorical | Monthly, Annual |
| TenureMonths | Numeric | 1 – 72 months |
| MonthlyCharge | Numeric | $0 – $100 |
| SupportTickets | Numeric | 0 – 8 |
| DaysSinceLastLogin | Numeric | 0 – 365 days |
| PaymentFailures | Numeric | 0 – 5 |

---

## Project Structure

```
churnsense/
├── app.py                        # Flask backend — /predict, /health, /
├── generate_data.py              # Dataset generation + model training script
├── model.pkl                     # Trained sklearn Pipeline (serialised)
├── model_meta.json               # Feature metadata (categories, ranges, model name)
├── requirements.txt              # Python dependencies
├── vercel.json                   # Vercel deployment config
├── .gitignore
├── templates/
│   └── index.html                # Single-file PWA frontend
├── static/
│   ├── manifest.json             # PWA manifest
│   ├── sw.js                     # Service worker
│   ├── icon-192.png
│   └── icon-512.png
└── Simulated Churn Data/
    └── Simulated Customer Data.csv   # 3,000-row training dataset
```

---

## API

### `GET /health`
Returns model status.
```json
{ "status": "ok", "model": "Logistic Regression" }
```

### `POST /predict`
Accepts a JSON body with all 9 features.

**Request**
```json
{
  "Gender": "Male",
  "Country": "India",
  "SubscriptionType": "Premium",
  "ContractType": "Monthly",
  "TenureMonths": 6,
  "MonthlyCharge": 75,
  "SupportTickets": 3,
  "DaysSinceLastLogin": 45,
  "PaymentFailures": 2
}
```

**Response**
```json
{ "churn": 1, "probability": 97.9 }
```

`churn` is `1` (will churn) or `0` (will stay). `probability` is the churn likelihood as a percentage (0–100).

---

## Running Locally

```bash
# 1. Clone
git clone https://github.com/RamyKhairy24/churnsense.git
cd churnsense

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Regenerate the dataset and retrain the model
python generate_data.py

# 5. Start the dev server
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `SECRET_KEY` | random (auto-generated) | Flask session secret |
| `PORT` | `5000` | HTTP port |
| `FLASK_DEBUG` | `false` | Enable debug mode (`true` / `false`) |

---

## Deployment on Vercel

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/RamyKhairy24/churnsense)

The repository includes a `vercel.json` that routes all traffic through `app.py` using the `@vercel/python` builder. No extra configuration is needed — just connect the repo in the Vercel dashboard and deploy.

Set the `SECRET_KEY` environment variable in the Vercel project settings for production.

---

## Model Training

`generate_data.py` builds the full ML pipeline:

1. Generates 3,000 synthetic customer records with a logistic churn model
2. Compares three classifiers via 5-fold stratified CV: Logistic Regression, Random Forest, Gradient Boosting
3. Retrains the best model on the full dataset
4. Saves `model.pkl` and `model_meta.json`

| Model | CV AUC | Test AUC |
|---|---|---|
| **Logistic Regression** | 0.7730 ± 0.017 | **0.7905** |
| Gradient Boosting | 0.7437 ± 0.018 | 0.7588 |
| Random Forest | 0.7417 ± 0.019 | 0.7426 |

---

## License

MIT
