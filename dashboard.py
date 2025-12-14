"""
Interactive Dashboard for Amazon Dynamic Pricing Recommendation Predictor
Data Product for Capstone Project
"""

import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import joblib
import os

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Amazon Recommendation Predictor"

# Load model and preprocessing objects
try:
    model = joblib.load('models/best_model.pkl')
    label_encoders = joblib.load('models/label_encoders.pkl')
    target_encoder = joblib.load('models/target_encoder.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    
    # Check if scaler exists (for Neural Network)
    scaler = None
    if os.path.exists('models/scaler.pkl'):
        scaler = joblib.load('models/scaler.pkl')
    
    model_loaded = True
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please run the Jupyter notebook first to generate model files.")
    model_loaded = False

# Define app layout
app.layout = html.Div([
    html.Div([
        html.H1("Amazon Dynamic Pricing - Recommendation Predictor", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '30px'}),
        html.P("Predict customer recommendation likelihood based on demographics, behavior, and pricing perceptions",
               style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '18px', 'marginBottom': '40px'})
    ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '30px'}),
    
    html.Div([
        html.Div([
            html.H3("Customer Demographics", style={'color': '#34495e', 'marginBottom': '20px'}),
            
            html.Label("Age", style={'fontWeight': 'bold', 'marginTop': '10px'}),
            dcc.Slider(
                id='age-slider',
                min=18,
                max=80,
                value=35,
                marks={i: str(i) for i in range(20, 81, 10)},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            
            html.Label("Gender", style={'fontWeight': 'bold', 'marginTop': '20px'}),
            dcc.Dropdown(
                id='gender-dropdown',
                options=[
                    {'label': 'Male', 'value': 'Male'},
                    {'label': 'Female', 'value': 'Female'},
                    {'label': 'Other', 'value': 'Other'}
                ],
                value='Male',
                style={'marginBottom': '10px'}
            ),
            
            html.Label("Location", style={'fontWeight': 'bold', 'marginTop': '20px'}),
            dcc.Dropdown(
                id='location-dropdown',
                options=[
                    {'label': 'North America', 'value': 'North America'},
                    {'label': 'Europe', 'value': 'Europe'},
                    {'label': 'Asia', 'value': 'Asia'},
                    {'label': 'Australia', 'value': 'Australia'},
                    {'label': 'South America', 'value': 'South America'}
                ],
                value='North America',
                style={'marginBottom': '10px'}
            ),
            
            html.Label("Annual Income ($)", style={'fontWeight': 'bold', 'marginTop': '20px'}),
            dcc.Slider(
                id='income-slider',
                min=20000,
                max=200000,
                value=100000,
                marks={i: f'${i//1000}k' for i in range(20000, 201000, 40000)},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 
                  'padding': '20px', 'backgroundColor': '#ffffff', 'borderRadius': '10px', 
                  'marginRight': '2%', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
        
        html.Div([
            html.H3("Shopping Behavior", style={'color': '#34495e', 'marginBottom': '20px'}),
            
            html.Label("Browsing Time per Week (Hours)", style={'fontWeight': 'bold', 'marginTop': '10px'}),
            dcc.Slider(
                id='browsing-slider',
                min=0,
                max=30,
                value=15,
                marks={i: str(i) for i in range(0, 31, 5)},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            
            html.Label("Purchase Frequency per Month", style={'fontWeight': 'bold', 'marginTop': '20px'}),
            dcc.Slider(
                id='purchase-slider',
                min=0,
                max=20,
                value=8,
                marks={i: str(i) for i in range(0, 21, 5)},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            
            html.H3("Pricing Perceptions", style={'color': '#34495e', 'marginTop': '30px', 'marginBottom': '20px'}),
            
            html.Label("Impact of Dynamic Pricing on Purchase", style={'fontWeight': 'bold', 'marginTop': '10px'}),
            dcc.Dropdown(
                id='impact-dropdown',
                options=[
                    {'label': 'No Impact', 'value': 'No Impact'},
                    {'label': 'Low Impact', 'value': 'Low Impact'},
                    {'label': 'Moderate Impact', 'value': 'Moderate Impact'},
                    {'label': 'High Impact', 'value': 'High Impact'}
                ],
                value='Moderate Impact',
                style={'marginBottom': '10px'}
            ),
            
            html.Label("Perception of Amazon Revenue Growth", style={'fontWeight': 'bold', 'marginTop': '20px'}),
            dcc.Dropdown(
                id='revenue-dropdown',
                options=[
                    {'label': 'Negative Impact', 'value': 'Negative Impact'},
                    {'label': 'No Growth', 'value': 'No Growth'},
                    {'label': 'Moderate Growth', 'value': 'Moderate Growth'},
                    {'label': 'Significant Growth', 'value': 'Significant Growth'}
                ],
                value='Moderate Growth',
                style={'marginBottom': '10px'}
            ),
            
            html.Label("Perception of Competition", style={'fontWeight': 'bold', 'marginTop': '20px'}),
            dcc.Dropdown(
                id='competition-dropdown',
                options=[
                    {'label': 'Decreased Competition', 'value': 'Decreased Competition'},
                    {'label': 'No Change', 'value': 'No Change'},
                    {'label': 'Increased Competition', 'value': 'Increased Competition'}
                ],
                value='No Change',
                style={'marginBottom': '10px'}
            ),
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top',
                  'padding': '20px', 'backgroundColor': '#ffffff', 'borderRadius': '10px',
                  'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
    ], style={'marginBottom': '30px'}),
    
    html.Div([
        html.Button('Predict Recommendation Likelihood', 
                   id='predict-button',
                   n_clicks=0,
                   style={'backgroundColor': '#3498db', 'color': 'white', 'fontSize': '18px',
                         'padding': '15px 30px', 'border': 'none', 'borderRadius': '5px',
                         'cursor': 'pointer', 'fontWeight': 'bold', 'width': '100%'})
    ], style={'textAlign': 'center', 'marginBottom': '30px'}),
    
    html.Div(id='prediction-output', style={'marginTop': '30px'}),
    
], style={'maxWidth': '1200px', 'margin': '0 auto', 'padding': '20px', 'fontFamily': 'Arial, sans-serif'})


def predict_recommendation(customer_data, model, encoders, target_encoder, scaler=None, feature_names=None):
    """Predict recommendation likelihood for a new customer."""
    # Create DataFrame from input
    df_input = pd.DataFrame([customer_data])
    
    # Encode categorical variables
    for col in encoders.keys():
        if col in df_input.columns:
            try:
                df_input[col] = encoders[col].transform(df_input[col].astype(str))
            except ValueError:
                df_input[col] = encoders[col].transform([encoders[col].classes_[0]])[0]
    
    # Reorder columns to match training data
    if feature_names:
        df_input = df_input.reindex(columns=feature_names, fill_value=0)
    
    # Scale if needed
    if scaler:
        df_input = scaler.transform(df_input)
    
    # Make prediction
    prediction = model.predict(df_input)[0]
    prediction_proba = model.predict_proba(df_input)[0]
    
    # Decode prediction
    predicted_class = target_encoder.inverse_transform([prediction])[0]
    
    # Get probabilities for all classes
    class_probs = {}
    for i, class_name in enumerate(target_encoder.classes_):
        class_probs[class_name] = prediction_proba[i]
    
    return predicted_class, class_probs


@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    State('age-slider', 'value'),
    State('gender-dropdown', 'value'),
    State('location-dropdown', 'value'),
    State('income-slider', 'value'),
    State('browsing-slider', 'value'),
    State('purchase-slider', 'value'),
    State('impact-dropdown', 'value'),
    State('revenue-dropdown', 'value'),
    State('competition-dropdown', 'value')
)
def update_prediction(n_clicks, age, gender, location, income, browsing, purchase, 
                     impact, revenue, competition):
    if not model_loaded:
        return html.Div([
            html.H3("Error: Model not loaded", style={'color': '#e74c3c'}),
            html.P("Please run the Jupyter notebook first to generate model files.")
        ], style={'padding': '20px', 'backgroundColor': '#fff', 'borderRadius': '10px',
                 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
    
    if n_clicks == 0:
        return html.Div()
    
    # Create feature categories (matching notebook logic)
    if income < 50000:
        income_cat = 'Low'
    elif income < 100000:
        income_cat = 'Medium'
    elif income < 150000:
        income_cat = 'High'
    else:
        income_cat = 'Very High'
    
    if age < 30:
        age_group = 'Young'
    elif age < 45:
        age_group = 'Middle'
    elif age < 60:
        age_group = 'Senior'
    else:
        age_group = 'Elderly'
    
    if browsing < 10:
        browsing_cat = 'Low'
    elif browsing < 20:
        browsing_cat = 'Medium'
    else:
        browsing_cat = 'High'
    
    if purchase < 5:
        purchase_cat = 'Low'
    elif purchase < 10:
        purchase_cat = 'Medium'
    elif purchase < 15:
        purchase_cat = 'High'
    else:
        purchase_cat = 'Very High'
    
    # Prepare customer data
    customer_data = {
        'Age': age,
        'Gender': gender,
        'Location': location,
        'Annual_Income': income,
        'Browsing_Time_per_Week_Hours': browsing,
        'Purchase_Frequency_Per_Month': purchase,
        'Impact_of_Dynamic_Pricing_on_Purchase': impact,
        'Perception_of_Amazon_Revenue_Growth_due_to_Dynamic_Pricing': revenue,
        'Perception_of_Competition_in_Amazon_Marketplace': competition,
        'Income_Category': income_cat,
        'Age_Group': age_group,
        'Browsing_Category': browsing_cat,
        'Purchase_Freq_Category': purchase_cat
    }
    
    # Make prediction
    try:
        predicted_class, probabilities = predict_recommendation(
            customer_data, model, label_encoders, target_encoder, scaler, feature_names
        )
        
        # Color mapping for predictions
        color_map = {
            'Highly Likely': '#2ecc71',
            'Likely': '#3498db',
            'Unlikely': '#f39c12',
            'Highly Unlikely': '#e74c3c'
        }
        
        # Create probability bar chart
        prob_df = pd.DataFrame({
            'Recommendation': list(probabilities.keys()),
            'Probability': list(probabilities.values())
        })
        prob_df = prob_df.sort_values('Probability', ascending=True)
        
        fig = go.Figure(data=[
            go.Bar(
                y=prob_df['Recommendation'],
                x=prob_df['Probability'],
                orientation='h',
                marker=dict(
                    color=[color_map.get(r, '#95a5a6') for r in prob_df['Recommendation']],
                    line=dict(color='white', width=1)
                ),
                text=[f"{p:.1%}" for p in prob_df['Probability']],
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title='Prediction Probabilities',
            xaxis_title='Probability',
            yaxis_title='Recommendation Likelihood',
            height=300,
            margin=dict(l=150, r=50, t=50, b=50),
            xaxis=dict(range=[0, 1])
        )
        
        return html.Div([
            html.Div([
                html.H2("Prediction Result", style={'color': '#2c3e50', 'marginBottom': '20px'}),
                html.Div([
                    html.H3("Predicted Recommendation:", style={'display': 'inline', 'marginRight': '10px'}),
                    html.H3(predicted_class, 
                           style={'display': 'inline', 'color': color_map.get(predicted_class, '#2c3e50'),
                                 'fontWeight': 'bold'})
                ], style={'marginBottom': '30px', 'textAlign': 'center'}),
                dcc.Graph(figure=fig),
                html.Div([
                    html.H4("Customer Profile Summary:", style={'marginTop': '30px', 'marginBottom': '15px'}),
                    html.P(f"Age: {age} | Gender: {gender} | Location: {location}"),
                    html.P(f"Income: ${income:,} | Browsing: {browsing} hrs/week | Purchases: {purchase}/month"),
                    html.P(f"Pricing Impact: {impact} | Revenue Perception: {revenue} | Competition: {competition}")
                ], style={'marginTop': '20px', 'padding': '15px', 'backgroundColor': '#ecf0f1',
                         'borderRadius': '5px'})
            ], style={'padding': '30px', 'backgroundColor': '#ffffff', 'borderRadius': '10px',
                     'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
        ])
    except Exception as e:
        return html.Div([
            html.H3("Prediction Error", style={'color': '#e74c3c'}),
            html.P(str(e))
        ], style={'padding': '20px', 'backgroundColor': '#fff', 'borderRadius': '10px',
                 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})


if __name__ == '__main__':
    if model_loaded:
        print("=" * 60)
        print("Dashboard starting...")
        print("Open your browser and navigate to: http://localhost:8050")
        print("=" * 60)
        app.run_server(debug=True, port=8050)
    else:
        print("=" * 60)
        print("ERROR: Model files not found!")
        print("Please run the Jupyter notebook first to generate model files.")
        print("=" * 60)

