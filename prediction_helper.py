import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Path to the saved model and its components
MODEL_PATH = 'main_engine/model_data.joblib'

# Load the model and its components
model_data = joblib.load(MODEL_PATH)
model = model_data['model']
scaler = model_data['scaler']
features = model_data['features']
cols_to_scale = model_data['cols_to_scale']


def prepare_input(age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
                  delinquency_ratio, credit_utilization_ratio, num_open_accounts, residence_type,
                  loan_purpose, loan_type):
    # Create a dictionary with input values and dummy values for missing features
    input_data = {
        'age': age,
        'loan_tenure_months': loan_tenure_months,
        'number_of_open_accounts': num_open_accounts,
        'credit_utilization_ratio': credit_utilization_ratio,
        'loan_to_income': loan_amount / income if income > 0 else 0,
        'delinquency_ratio': delinquency_ratio,
        'avg_dpd_per_delinquency': avg_dpd_per_delinquency,
        'residence_type_Owned': 1 if residence_type == 'Owned' else 0,
        'residence_type_Rented': 1 if residence_type == 'Rented' else 0,
        'loan_purpose_Education': 1 if loan_purpose == 'Education' else 0,
        'loan_purpose_Home': 1 if loan_purpose == 'Home' else 0,
        'loan_purpose_Personal': 1 if loan_purpose == 'Personal' else 0,
        'loan_type_Unsecured': 1 if loan_type == 'Unsecured' else 0,
        # Dummy values for missing columns expected by the model
        'number_of_dependants': 1,  
        'years_at_current_address': 1,  
        'zipcode': 1,  
        'sanction_amount': 1,  
        'processing_fee': 1,  
        'gst': 1,  
        'net_disbursement': 1,  
        'principal_outstanding': 1,  
        'bank_balance_at_application': 1,  
        'number_of_closed_accounts': 1,  
        'enquiry_count': 1  
    }

    # Create a DataFrame from the input data
    df = pd.DataFrame([input_data])

    # Ensure all required columns for scaling are present
    for col in cols_to_scale:
        if col not in df.columns:
            df[col] = 0  

    # Apply Min-Max Scaling to numerical columns
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    # Ensure DataFrame contains only features expected by the model
    df = df[features]

    return df


def predict(age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
            delinquency_ratio, credit_utilization_ratio, num_open_accounts,
            residence_type, loan_purpose, loan_type):
    # Prepare input data
    input_df = prepare_input(age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
                             delinquency_ratio, credit_utilization_ratio, num_open_accounts, residence_type,
                             loan_purpose, loan_type)

    probability, credit_score, rating = calculate_credit_score(input_df)

    return probability, credit_score, rating


def calculate_credit_score(input_df, base_score=300, scale_length=600):
    x = np.dot(input_df.values, model.coef_.T) + model.intercept_

    # Apply the logistic function to calculate default probability
    default_probability = np.clip(1 / (1 + np.exp(-x)), 0.01, 0.99)  # Ensuring probabilities stay within a meaningful range
    non_default_probability = 1 - default_probability

    # Adjusted credit score scaling
    credit_score = base_score + (non_default_probability.flatten() ** 0.5) * scale_length  # Using square root transformation

    # Debugging prints to verify values
    print(f"Raw Model Output (x): {x}")
    print(f"Default Probability: {default_probability.flatten()[0]:.4f}")
    print(f"Credit Score: {int(credit_score[0])}")

    # Improved rating categories for better score distribution
    def get_rating(score):
        if 300 <= score < 450:
            return 'Poor'
        elif 450 <= score < 600:
            return 'Average'
        elif 600 <= score < 750:
            return 'Good'
        elif 750 <= score <= 900:
            return 'Excellent'
        return 'Undefined'

    rating = get_rating(credit_score[0])

    return default_probability.flatten()[0], int(credit_score[0]), rating
