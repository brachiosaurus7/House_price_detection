from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

app = Flask(__name__)

# Example area data and synthetic data generation for training the model
areas = {
    'C R Park': {'1 BHK': (4000000, 6700000), '2 BHK': (7400000, 12100000), '3 BHK': (12700000, 20200000)},
    'Greater Kailash I': {'1 BHK': (6060000, 10000000), '2 BHK': (11100000, 18100000), '3 BHK': (19100000, 30300000)},
    'Hauz Khas': {'1 BHK': (5700000, 9500000), '2 BHK': (10400000, 17100000), '3 BHK': (18000000, 28500000)},
    'Lajpat Nagar': {'1 BHK': (4350000, 7250000), '2 BHK': (7300000, 12000000), '3 BHK': (12700000, 20100000)},
    'Vasant Kunj': {'1 BHK': (3960000, 6600000), '2 BHK': (7260000, 11800000), '3 BHK': (12000000, 20000000)},
    'South Extension': {'1 BHK': (4500000, 7500000), '2 BHK': (7700000, 12300000), '3 BHK': (13000000, 21000000)},
    'Defence Colony': {'1 BHK': (6000000, 10500000), '2 BHK': (11300000, 18300000), '3 BHK': (19500000, 31500000)},
    'Jor Bagh': {'1 BHK': (7000000, 11000000), '2 BHK': (12000000, 19000000), '3 BHK': (20000000, 33000000)},
    'Connaught Place': {'1 BHK': (5000000, 9000000), '2 BHK': (9500000, 16000000), '3 BHK': (17000000, 28000000)},
    'Saket': {'1 BHK': (4200000, 6900000), '2 BHK': (7100000, 11700000), '3 BHK': (12500000, 20500000)},
    'Dwarka': {'1 BHK': (3000000, 5500000), '2 BHK': (6000000, 9500000), '3 BHK': (10000000, 15000000)},
    'Karol Bagh': {'1 BHK': (3500000, 6000000), '2 BHK': (6500000, 10500000), '3 BHK': (11000000, 17000000)},
    'Pitampura': {'1 BHK': (4000000, 6500000), '2 BHK': (7000000, 11500000), '3 BHK': (12000000, 18500000)},
    'Rohini': {'1 BHK': (3500000, 6000000), '2 BHK': (6500000, 10500000), '3 BHK': (11000000, 17000000)},
    'Janakpuri': {'1 BHK': (3200000, 5500000), '2 BHK': (6000000, 9500000), '3 BHK': (10000000, 15500000)},
    'Paschim Vihar': {'1 BHK': (3700000, 6200000), '2 BHK': (6700000, 11000000), '3 BHK': (11500000, 17500000)},
    'Punjabi Bagh': {'1 BHK': (4500000, 7500000), '2 BHK': (7700000, 12300000), '3 BHK': (13000000, 21000000)},
    'Rajouri Garden': {'1 BHK': (4000000, 6500000), '2 BHK': (7000000, 11500000), '3 BHK': (12000000, 18500000)},
    'Shalimar Bagh': {'1 BHK': (3700000, 6200000), '2 BHK': (6700000, 11000000), '3 BHK': (11500000, 17500000)},
    'Tagore Garden': {'1 BHK': (3500000, 6000000), '2 BHK': (6500000, 10500000), '3 BHK': (11000000, 17000000)},
    'Vikaspuri': {'1 BHK': (3200000, 5500000), '2 BHK': (6000000, 9500000), '3 BHK': (10000000, 15500000)},
    'Uttam Nagar': {'1 BHK': (2500000, 4000000), '2 BHK': (4500000, 7000000), '3 BHK': (7500000, 12000000)},
    'Malviya Nagar': {'1 BHK': (3700000, 6200000), '2 BHK': (6700000, 11000000), '3 BHK': (11500000, 17500000)},
    'Kalkaji': {'1 BHK': (3500000, 6000000), '2 BHK': (6500000, 10500000), '3 BHK': (11000000, 17000000)},
    'Nehru Place': {'1 BHK': (4000000, 7000000), '2 BHK': (7200000, 11700000), '3 BHK': (12000000, 20000000)},
    'New Friends Colony': {'1 BHK': (6000000, 10000000), '2 BHK': (11000000, 18000000), '3 BHK': (19000000, 30000000)},
    'East of Kailash': {'1 BHK': (3700000, 6200000), '2 BHK': (6700000, 11000000), '3 BHK': (11500000, 17500000)},
    'Kailash Colony': {'1 BHK': (3700000, 6200000), '2 BHK': (6700000, 11000000), '3 BHK': (11500000, 17500000)},
    'Jangpura': {'1 BHK': (4000000, 7000000), '2 BHK': (7200000, 11700000), '3 BHK': (12000000, 20000000)},
    'Lodi Colony': {'1 BHK': (5000000, 9000000), '2 BHK': (9500000, 16000000), '3 BHK': (17000000, 28000000)},
    'Chanakyapuri': {'1 BHK': (6000000, 10000000), '2 BHK': (11000000, 18000000), '3 BHK': (19000000, 30000000)},
    'Anand Niketan': {'1 BHK': (5500000, 9500000), '2 BHK': (10000000, 17000000), '3 BHK': (18000000, 29000000)},
    'Sundar Nagar': {'1 BHK': (5000000, 9000000), '2 BHK': (9500000, 16000000), '3 BHK': (17000000, 28000000)},
    'Green Park': {'1 BHK': (3700000, 6200000), '2 BHK': (6700000, 11000000), '3 BHK': (11500000, 17500000)},
    'Safdarjung Enclave': {'1 BHK': (5000000, 9000000), '2 BHK': (9500000, 16000000), '3 BHK': (17000000, 28000000)},
    'Vasant Vihar': {'1 BHK': (5500000, 9500000), '2 BHK': (10000000, 17000000), '3 BHK': (18000000, 29000000)},
    'Malcha Marg': {'1 BHK': (6000000, 10000000), '2 BHK': (11000000, 18000000), '3 BHK': (19000000, 30000000)},
    'Civil Lines': {'1 BHK': (4500000, 7500000), '2 BHK': (7700000, 12300000), '3 BHK': (13000000, 21000000)},
    'Kashmere Gate': {'1 BHK': (4000000, 6500000), '2 BHK': (7000000, 11500000), '3 BHK': (12000000, 18500000)},
    'Model Town': {'1 BHK': (4000000, 7000000), '2 BHK': (7200000, 11700000), '3 BHK': (12000000, 20000000)},
    'Ashok Vihar': {'1 BHK': (4200000, 6900000), '2 BHK': (7100000, 11700000), '3 BHK': (12500000, 20500000)},
    'Mehrauli': {'1 BHK': (3500000, 6000000), '2 BHK': (6500000, 10500000), '3 BHK': (11000000, 17000000)},
    'Munirka': {'1 BHK': (3700000, 6200000), '2 BHK': (6700000, 11000000), '3 BHK': (11500000, 17500000)},
    'Sarita Vihar': {'1 BHK': (3200000, 5500000), '2 BHK': (6000000, 9500000), '3 BHK': (10000000, 15500000)},
    'Okhla': {'1 BHK': (3500000, 6000000), '2 BHK': (6500000, 10500000), '3 BHK': (11000000, 17000000)},
    'Geeta Colony': {'1 BHK': (2800000, 4500000), '2 BHK': (5000000, 8000000), '3 BHK': (9000000, 14000000)},
    'Shahdara': {'1 BHK': (3000000, 4800000), '2 BHK': (5200000, 8500000), '3 BHK': (9500000, 15000000)},
    'Preet Vihar': {'1 BHK': (3500000, 5500000), '2 BHK': (6000000, 9500000), '3 BHK': (10000000, 16000000)},
    'Patparganj': {'1 BHK': (3800000, 6000000), '2 BHK': (6500000, 10500000), '3 BHK': (11000000, 17000000)},
    'Mayur Vihar': {'1 BHK': (3600000, 5800000), '2 BHK': (6300000, 10200000), '3 BHK': (10500000, 16500000)},
    'Indraprastha Extension': {'1 BHK': (3900000, 6200000), '2 BHK': (6800000, 11000000), '3 BHK': (11500000, 18000000)},
    'Dwarka Mor': {'1 BHK': (3000000, 5000000), '2 BHK': (5500000, 9000000), '3 BHK': (9500000, 15000000)},
    'Uttam Nagar West': {'1 BHK': (2700000, 4400000), '2 BHK': (4800000, 7600000), '3 BHK': (8200000, 13000000)},
    'Rajendra Nagar': {'1 BHK': (3500000, 5700000), '2 BHK': (6200000, 9900000), '3 BHK': (10500000, 16500000)},
    'Inderlok': {'1 BHK': (3200000, 5200000), '2 BHK': (5700000, 9200000), '3 BHK': (9700000, 15000000)},
    'Jahangirpuri': {'1 BHK': (2800000, 4500000), '2 BHK': (5000000, 8000000), '3 BHK': (8500000, 13500000)},
    'Bawana': {'1 BHK': (2500000, 4000000), '2 BHK': (4500000, 7200000), '3 BHK': (7500000, 12000000)},
    'Sadar Bazar': {'1 BHK': (3000000, 5000000), '2 BHK': (5500000, 8800000), '3 BHK': (9000000, 14000000)},
    'Nangloi': {'1 BHK': (2700000, 4500000), '2 BHK': (5000000, 8200000), '3 BHK': (8500000, 13500000)},
    'Najafgarh': {'1 BHK': (2400000, 4000000), '2 BHK': (4500000, 7500000), '3 BHK': (7800000, 12500000)},
    'Gandhi Nagar': {'1 BHK': (2900000, 4700000), '2 BHK': (5200000, 8600000), '3 BHK': (8800000, 14000000)},
    'Daryaganj': {'1 BHK': (3100000, 5200000), '2 BHK': (5700000, 9300000), '3 BHK': (9500000, 15000000)},
    'Khan Market': {'1 BHK': (6000000, 10000000), '2 BHK': (11000000, 18000000), '3 BHK': (19000000, 30000000)},
    'Sarojini Nagar': {'1 BHK': (4200000, 7000000), '2 BHK': (7300000, 12000000), '3 BHK': (12500000, 20000000)},
    'Chittaranjan Park': {'1 BHK': (4000000, 6700000), '2 BHK': (7400000, 12100000), '3 BHK': (12700000, 20200000)},
    'Govindpuri': {'1 BHK': (2800000, 4500000), '2 BHK': (5000000, 8000000), '3 BHK': (8500000, 13500000)},
    'Badarpur': {'1 BHK': (2500000, 4200000), '2 BHK': (4700000, 7500000), '3 BHK': (7800000, 12500000)},
    'Khanpur': {'1 BHK': (2600000, 4300000), '2 BHK': (4800000, 7700000), '3 BHK': (8000000, 12800000)},
    'Madangir': {'1 BHK': (2700000, 4500000), '2 BHK': (5000000, 8000000), '3 BHK': (8500000, 13500000)},
    'Tughlakabad': {'1 BHK': (2800000, 4600000), '2 BHK': (5100000, 8200000), '3 BHK': (8600000, 13800000)},
    'Chhatarpur': {'1 BHK': (2900000, 4800000), '2 BHK': (5200000, 8500000), '3 BHK': (8800000, 14000000)},
    'Neb Sarai': {'1 BHK': (3000000, 5000000), '2 BHK': (5500000, 9000000), '3 BHK': (9500000, 15000000)},
    'Sangam Vihar': {'1 BHK': (2500000, 4200000), '2 BHK': (4700000, 7500000), '3 BHK': (7800000, 12500000)},
    'Rangpuri': {'1 BHK': (2600000, 4300000), '2 BHK': (4800000, 7700000), '3 BHK': (8000000, 12800000)},
    'Kapashera': {'1 BHK': (2700000, 4500000), '2 BHK': (5000000, 8000000), '3 BHK': (8500000, 13500000)},
    'Mahipalpur': {'1 BHK': (2800000, 4600000), '2 BHK': (5100000, 8200000), '3 BHK': (8600000, 13800000)},
    'Palam': {'1 BHK': (2900000, 4800000), '2 BHK': (5200000, 8500000), '3 BHK': (8800000, 14000000)},
    'Kishangarh': {'1 BHK': (3000000, 5000000), '2 BHK': (5500000, 9000000), '3 BHK': (9500000, 15000000)},
    'Arjun Nagar': {'1 BHK': (2500000, 4200000), '2 BHK': (4700000, 7500000), '3 BHK': (7800000, 12500000)},
    'Bijwasan': {'1 BHK': (2600000, 4300000), '2 BHK': (4800000, 7700000), '3 BHK': (8000000, 12800000)},
    'Dabri': {'1 BHK': (2700000, 4500000), '2 BHK': (5000000, 8000000), '3 BHK': (8500000, 13500000)},
    'Dwarka Sector 7': {'1 BHK': (2800000, 4600000), '2 BHK': (5100000, 8200000), '3 BHK': (8600000, 13800000)},
}

# Generate synthetic data
np.random.seed(0)
data = []
for area, types in areas.items():
    for bhk, (min_price, max_price) in types.items():
        for _ in range(3):  # Adding 3 samples for each type in each area
            area_size = np.random.randint(500, 1500)
            bedrooms = int(bhk.split()[0])
            bathrooms = np.random.randint(1, 3)
            age = np.random.randint(1, 30)
            price = np.random.randint(min_price, max_price)
            data.append([area, area_size, bedrooms, bathrooms, age, price])

df = pd.DataFrame(data, columns=['Locality', 'Area', 'Bedrooms', 'Bathrooms', 'Age', 'Price'])

# Model pipeline for preprocessing and prediction
preprocessor = ColumnTransformer(
    transformers=[
        ('locality', OneHotEncoder(), ['Locality'])
    ],
    remainder='passthrough'
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

# Split the data and train the model
X = df[['Locality', 'Area', 'Bedrooms', 'Bathrooms', 'Age']]
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    locality = data['locality']
    area = float(data['area'])
    bedrooms = int(data['bedrooms'])
    bathrooms = int(data['bathrooms'])
    age = int(data['age'])
    
    input_data = pd.DataFrame([[locality, area, bedrooms, bathrooms, age]], 
                              columns=['Locality', 'Area', 'Bedrooms', 'Bathrooms', 'Age'])
    predicted_price = pipeline.predict(input_data)[0]
    return jsonify({'predicted_price': predicted_price})

if __name__ == '__main__':
    app.run(debug=True, port=5000)  # Ensure this is the correct port
