# training_pipeline/utils.py

def calculate_income_to_loan_ratio(income, loan):
    if loan == 0:
        return 0
    return round(income / loan, 2)
