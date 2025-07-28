import pandas as pd
from sklearn.linear_model import LinearRegression

def purchase_prediction(past_purchases):
    df = pd.DataFrame(past_purchases)
    X = df[['Month']]
    y = df['Purchases']
    model = LinearRegression()
    model.fit(X, y)
    future_month = [[13]]  # Predicting the 13th month
    prediction = model.predict(future_month)
    return prediction[0]

past_purchases = {
    "Month": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "Purchases": [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650]
}

prediction = purchase_prediction(past_purchases)
print(f"Next month's predicted purchases: {prediction}")
