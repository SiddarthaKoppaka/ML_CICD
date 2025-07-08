import pandas as pd
import random
from faker import Faker

fake = Faker()

def generate_fake_loan_data(num_rows=1000, seed=42):
    random.seed(seed)
    Faker.seed(seed)
    
    data = []
    for i in range(num_rows):
        age = random.randint(18, 65)
        income = round(random.uniform(2000, 15000), 2)
        employment_type = random.choice(['salaried', 'self-employed', 'unemployed'])
        loan_amount = round(random.uniform(1000, 50000), 2)
        loan_term = random.choice([12, 24, 36, 48, 60])
        credit_score = random.randint(300, 850)
        
        # Simple rule-based target generation
        default = 1 if (credit_score < 600 or income < 3000) and employment_type != 'salaried' else 0
        
        data.append({
            "user_id": fake.uuid4(),
            "age": age,
            "income": income,
            "employment_type": employment_type,
            "loan_amount": loan_amount,
            "loan_term_months": loan_term,
            "credit_score": credit_score,
            "default": default
        })
    
    return pd.DataFrame(data)

# Example usage
df = generate_fake_loan_data(1000)
df.to_csv("data/loan_data.csv", index=False)