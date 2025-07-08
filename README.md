# ML_CICD

## Structured ML CI/CD Implementation Plan
### Phase 1: Foundation (Data & Feature Pipeline)
Use Case Definition & Faker Dataset Creation ← (We start here)

Feature Ingestion to Feature Store (like Feast)

Initial Training Pipeline (Logging with MLflow or Prometheus + Grafana)

### Phase 2: DevOps Integration
Create CI pipeline (GitHub Actions for testing, linting)

Create CT pipeline (automated testing of data, model metrics, feature schema)

Create CD pipeline (Docker + model deployment + service startup)

### Phase 3: Experimentation & Monitoring
Experimentation (MLflow + different model versions)

Model Version Upgrade

Data Drift Detection (e.g., Evidently, WhyLabs, or custom Prometheus metrics)



## Faker :

Data is faked to imitate real world usecase --

A user applies for a loan and based on features like income, age, employment, etc., we predict whether they will default.
* `user_id`: Unique user identifier
* `age`: Integer \[18, 65]
* `income`: Monthly income in USD
* `employment_type`: \['salaried', 'self-employed', 'unemployed']
* `loan_amount`: Amount requested
* `loan_term_months`: Duration of the loan
* `credit_score`: Simulated credit score \[300, 850]
* `default`: Binary target (0 = No default, 1 = Default)       

