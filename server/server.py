from flask import Flask, request, jsonify
import util
import pandas as pd
import joblib
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)

# Load your saved scaler and model
scaler = joblib.load('artifacts/scaler.pkl')  # Load the saved scaler
best_rf_model = joblib.load('artifacts/final_best_rf_model_with_top_12_features.pkl')  # Load the saved model

# Load the country mean target mapping (previously saved as 'country_mean_target_mapping.csv')
country_mean_target_mapping = pd.read_csv('artifacts/country_mean_target_mapping.csv', index_col=0)


@app.route('/get_countries', methods=['GET'])
def get_country_names():
    response = jsonify({
        'countries': util.get_countries()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')  # Allow CORS
    return response


@app.route('/predict_life_expectancy', methods=['POST'])
def predict_life_expectancy():
    user_input = request.json  # Get the JSON request data from the client

    try:
        # Call the prediction function
        predicted_life_expectancy = preprocess_and_predict(user_input)

        response = jsonify({
            'predicted_life_expectancy': predicted_life_expectancy
        })
        response.headers.add('Access-Control-Allow-Origin', '*')  # Allow CORS
        return response

    except ValueError as e:
        return jsonify({'error': str(e)}), 400


def preprocess_and_predict(user_input):
    """
    Preprocesses raw user input and predicts life expectancy.

    Parameters:
    user_input (dict): Dictionary containing raw input values (e.g., 'Country', 'Year', etc.)

    Returns:
    float: Predicted life expectancy
    """
    # Map 'Country' to 'Country_Mean_Target'
    country = user_input.get('Country')
    if country in country_mean_target_mapping.index:
        country_mean_target = country_mean_target_mapping.loc[country].values[0]
    else:
        raise ValueError(f"Country '{country}' not found in the mapping!")

    # Convert input values from strings to the appropriate types
    year = int(user_input.get('Year'))
    adult_mortality = float(user_input.get('Adult Mortality'))
    income_composition = float(user_input.get('Income composition of resources'))
    hiv_aids = float(user_input.get('HIV/AIDS'))
    schooling = float(user_input.get('Schooling'))
    bmi = float(user_input.get('BMI'))
    measles = float(user_input.get('Measles '))
    thinness_1_19 = float(user_input.get('thinness 1-19 years'))
    total_expenditure = float(user_input.get('Total expenditure'))
    thinness_5_9 = float(user_input.get('thinness 5-9 years'))
    gdp = float(user_input.get('GDP'))
    population = float(user_input.get('Population'))

    # Compute 'GDP_per_capita'
    gdp_per_capita = gdp / (population + 1)  # Avoid division by zero

    # Preprocess the input
    input_data = {
        'Country_Mean_Target': country_mean_target,  # Mapped value from the country
        'Year': year,
        'Adult Mortality': adult_mortality,
        'Income composition of resources': income_composition,
        'HIV/AIDS': hiv_aids,
        'Schooling': schooling,
        'BMI': bmi,
        'Measles ': measles,
        'thinness 1-19 years': thinness_1_19,
        'Total expenditure': total_expenditure,
        'thinness 5-9 years': thinness_5_9,
        'GDP_per_capita': gdp_per_capita  # Computed value
    }

    # Convert the input data into a DataFrame
    input_df = pd.DataFrame([input_data])

    # Select only the top 12 features
    features = ['Country_Mean_Target', 'Year', 'Adult Mortality', 'Income composition of resources',
                'HIV/AIDS', 'Schooling', 'BMI', 'Measles ', 'thinness 1-19 years',
                'Total expenditure', 'thinness 5-9 years', 'GDP_per_capita']

    # Standardize/normalize the input data using the same scaler used during training
    input_scaled = scaler.transform(input_df[features])

    # Make predictions using the preprocessed input data
    predicted_life_expectancy = best_rf_model.predict(input_scaled)

    return predicted_life_expectancy[0]



if __name__ == "__main__":
    print("Starting Python Flask Server For Life expectancy prediction")
    app.run(port=5002)
