# Purchase-Prediction-System
This project predicts the number of purchases for the next month using past purchase data and linear regression.

## Features
- Uses a linear regression model to predict future purchases.
- Accepts monthly purchase data as input.
- Provides easy-to-use functionality for business forecasting.

## Requirements
The following libraries are required to run the project:
- `pandas`
- `scikit-learn`

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/Quantmbot-AI/Purchase-Prediction.git
   cd Purchase-Prediction
   ```

2. Define your past purchase data in the script:
   ```python
   past_purchases = {
       "Month": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
       "Purchases": [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650]
   }
   ```

3. Run the script to predict next month's purchases:
   ```bash
   python purchase_prediction.py
   ```

4. The script outputs the predicted purchase count for the next month:
   ```
   Next month's predicted purchases: 700.0
   ```

## Customization
- **Past Purchases Data:** Modify the `past_purchases` dictionary to input your own data.
- **Prediction Month:** Adjust the `future_month` variable in the script to predict for other months.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

### Notes
- Ensure your past purchase data is in a dictionary format with "Month" and "Purchases" as keys.
- The model assumes a linear trend in purchase data over time.

