from flask import Flask, render_template
import pandas as pd
import plotly.express as px
import plotly.io as pio
import pickle

app = Flask(__name__)

# Load the models
with open('model_linear_regression.pkl', 'rb') as file:
    model_lr = pickle.load(file)

with open('model_random_forest.pkl', 'rb') as file:
    model_rf = pickle.load(file)

@app.route('/')
def index():
    # Load data from CSV file
    df = pd.read_csv('Bangalore_2_dataset.csv')

    # Ensure the DataFrame contains all required features
    features = ['location', 'bath', 'bhk', 'total_sqft_float']
    
    if all(feature in df.columns for feature in features):
        # Use the models to make predictions
        df['pred_linear'] = model_lr.predict(df[features])
        df['pred_random_forest'] = model_rf.predict(df[features])

        # Generate visualizations
        fig_linear = px.scatter(df, x='price_sqft', y='pred_linear', title='Linear Regression Predictions')
        linear_regression_plot = pio.to_html(fig_linear, full_html=False)

        fig_rf = px.scatter(df, x='price_sqft', y='pred_random_forest', title='Random Forest Predictions')
        random_forest_plot = pio.to_html(fig_rf, full_html=False)

        

        return render_template('index.html', 
                               linear_regression_plot=linear_regression_plot, 
                               random_forest_plot=random_forest_plot)
    else:
        return "The required features are not present in the CSV file."

if __name__ == '__main__':
    app.run(debug=True)
