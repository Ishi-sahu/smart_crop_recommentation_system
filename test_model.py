# =========================================================
# SMART CROP RECOMMENDATION SYSTEM
# =========================================================

import pandas as pd
import numpy as np
import joblib

# =========================================================
# LOAD MODEL FILES
# =========================================================

model = joblib.load("smart_crop_model.pkl")

crop_encoder = joblib.load("crop_encoder.pkl")

label_encoders = joblib.load("label_encoders.pkl")

print("System Loaded Successfully!")

# =========================================================
# LOAD MSP + PRICE DATASET
# =========================================================

price_df = pd.read_csv(
    "crop_prices_profitability.csv"
)

# =========================================================
# USER INPUT
# =========================================================

user_data = {

    'n': 90,
    'p': 42,
    'k': 43,

    'temperature': 22.5,

    'humidity': 80,

    'ph': 6.5,

    'rainfall': 202,

    'soil_type': 'Alluvial',

    'season': 'Kharif',

    'state': 'Andhra Pradesh',

    'irrigation': 'Yes'
}

# =========================================================
# DISPLAY INPUT
# =========================================================

print("\n======================================")
print("USER INPUT ANALYSIS")
print("======================================")

for key, value in user_data.items():

    print(f"{key} : {value}")

# =========================================================
# CONVERT TO DATAFRAME
# =========================================================

input_df = pd.DataFrame([user_data])

# =========================================================
# ENCODE CATEGORICAL FEATURES
# =========================================================

categorical_cols = [

    'soil_type',
    'season',
    'state',
    'irrigation'

]

for col in categorical_cols:

    input_df[col] = label_encoders[col].transform(

        input_df[col]

    )

# =========================================================
# GET PREDICTION PROBABILITIES
# =========================================================

probabilities = model.predict_proba(
    input_df
)[0]

# =========================================================
# TOP 3 CROP PREDICTIONS
# =========================================================

top3_indices = probabilities.argsort()[-3:][::-1]

top3_probs = probabilities[top3_indices]

top3_crops = crop_encoder.inverse_transform(
    top3_indices
)

# =========================================================
# ECONOMIC ANALYSIS
# =========================================================

results = []

best_crop = None

highest_profit = -999999

for i in range(3):

    crop_name = top3_crops[i]

    confidence = top3_probs[i] * 100

    # Fetch crop economic data
    crop_row = price_df[

        price_df['crop'].str.lower()

        == crop_name.lower()

    ]

    if not crop_row.empty:

        MSP = crop_row.iloc[0]['MSP']

        market_price = crop_row.iloc[0][
            'market_price'
        ]

        cultivation_cost = crop_row.iloc[0][
            'cost_of_cultivation'
        ]

        estimated_profit = (
            market_price - cultivation_cost
        )

        # Risk Analysis
        if estimated_profit > 3000:

            risk = "LOW"

        elif estimated_profit > 1500:

            risk = "MEDIUM"

        else:

            risk = "HIGH"

        results.append({

            'crop': crop_name,

            'confidence': confidence,

            'MSP': MSP,

            'market_price': market_price,

            'profit': estimated_profit,

            'risk': risk

        })

        # Best profitable crop
        if estimated_profit > highest_profit:

            highest_profit = estimated_profit

            best_crop = crop_name

# =========================================================
# DISPLAY TOP CROPS
# =========================================================

print("\n======================================")
print("TOP 3 CROP RECOMMENDATIONS")
print("======================================")

for i, item in enumerate(results):

    print(f"""

{i+1}. Crop Name        : {item['crop']}

   Confidence        : {item['confidence']:.2f}%

   MSP               : ₹{item['MSP']}

   Market Price      : ₹{item['market_price']}

   Estimated Profit  : ₹{item['profit']}

   Risk Level        : {item['risk']}

""")

# =========================================================
# FINAL RECOMMENDATION
# =========================================================

print("\n======================================")
print("FINAL BEST CROP")
print("======================================")

print(f"""

Recommended Crop : {best_crop}

Highest Estimated Profit : ₹{highest_profit}

Reason:
- Suitable for soil nutrients
- Good weather compatibility
- High market value
- Better profitability

""")