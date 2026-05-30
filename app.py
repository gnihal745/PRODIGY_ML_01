from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load dataset
df = pd.read_csv("train.csv")

# Features and target
X = df[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
y = df['SalePrice']

# Train model
model = LinearRegression()
model.fit(X, y)

@app.route('/')
def home():
    return render_template(
        'index.html',
        area=None,
        bedrooms=None,
        bathrooms=None,
        predicted_price=None
    )

@app.route('/predict', methods=['POST'])
def predict():

    try:
        area = float(request.form['area'])
        bedrooms = int(request.form['bedrooms'])
        bathrooms = int(request.form['bathrooms'])

        prediction = model.predict(
            [[area, bedrooms, bathrooms]]
        )

        predicted_price = round(prediction[0], 2)

        return render_template(
            'index.html',
            area=area,
            bedrooms=bedrooms,
            bathrooms=bathrooms,
            predicted_price=predicted_price
        )

    except Exception as e:
        return render_template(
            'index.html',
            error=str(e)
        )

if __name__ == '__main__':
    app.run(debug=True)