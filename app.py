from flask import Flask, render_template, request

import pandas as pd
import joblib

# =====================================================
# LOAD FILES
# =====================================================

model = joblib.load("smart_crop_model.pkl")

crop_encoder = joblib.load("crop_encoder.pkl")

label_encoders = joblib.load("label_encoders.pkl")

price_df = pd.read_csv(
    "crop_prices_profitability.csv"
)

# =====================================================
# CREATE FLASK APP
# =====================================================

app = Flask(__name__)

# =====================================================
# HOME PAGE
# =====================================================

@app.route('/')

def home():

    return render_template("index.html")

# =====================================================
# PREDICTION ROUTE
# =====================================================

@app.route('/predict', methods=['POST'])

def predict():

    # ============================================
    # GET USER INPUT
    # ============================================

    user_data = {

        'n': float(request.form['n']),

        'p': float(request.form['p']),

        'k': float(request.form['k']),

        'temperature': float(
            request.form['temperature']
        ),

        'humidity': float(
            request.form['humidity']
        ),

        'ph': float(request.form['ph']),

        'rainfall': float(
            request.form['rainfall']
        ),

        'soil_type': request.form['soil_type'],

        'season': request.form['season'],

        'state': request.form['state'],

        'irrigation': request.form['irrigation']
    }

    # ============================================
    # DATAFRAME
    # ============================================

    input_df = pd.DataFrame([user_data])

    # ============================================
    # ENCODE CATEGORICAL FEATURES
    # ============================================

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

    # ============================================
    # PREDICTION
    # ============================================

    probabilities = model.predict_proba(
        input_df
    )[0]

    top3_indices = probabilities.argsort()[-3:][::-1]

    top3_probs = probabilities[top3_indices]

    top3_crops = crop_encoder.inverse_transform(
        top3_indices
    )

    # ============================================
    # FINAL RESULTS
    # ============================================

    results = []

    best_crop = None

    highest_profit = -999999

    for i in range(3):

        crop_name = top3_crops[i]

        confidence = round(
            top3_probs[i] * 100,
            2
        )

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

            profit = (
                market_price
                - cultivation_cost
            )

            results.append({

                'crop': crop_name,

                'confidence': confidence,

                'MSP': MSP,

                'market_price': market_price,

                'profit': profit

            })

            if profit > highest_profit:

                highest_profit = profit

                best_crop = crop_name

    # ============================================
    # RETURN RESULT PAGE
    # ============================================

    return render_template(

        "index.html",

        prediction=best_crop,

        results=results,

        highest_profit=highest_profit

    )

# =====================================================
# RUN APP
# =====================================================

if __name__ == "__main__":

    app.run(debug=True)