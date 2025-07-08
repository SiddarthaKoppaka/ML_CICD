# tests/test_utils.py

from training_pipeline.utils import calculate_income_to_loan_ratio

def test_income_to_loan_ratio_normal():
    assert calculate_income_to_loan_ratio(5000, 1000) == 5.0

def test_income_to_loan_ratio_zero_loan():
    assert calculate_income_to_loan_ratio(5000, 0) == 0
