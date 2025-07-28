import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Keras/TensorFlow imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
torch.manual_seed(42)

def load_and_prepare_data(file_path):
    """Load and prepare the sales data"""
    df = pd.read_csv(file_path)
    df['invoiceDate'] = pd.to_datetime(df['invoiceDate'])
    df = df.sort_values('invoiceDate')
    df = df.set_index('invoiceDate')
    
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Amount statistics:")
    print(df['amount'].describe())
    
    return df

def create_sequences(data, seq_length):
    """Create sequences for time series prediction"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def smape(y_true, y_pred):
    """Calculate Symmetric Mean Absolute Percentage Error"""
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

def build_keras_model(input_shape):
    """Build Keras LSTM model"""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

class SalesDataset(Dataset):
    """PyTorch Dataset for sales data"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class PyTorchLSTM(nn.Module):
    """PyTorch LSTM model"""
    def __init__(self, input_size=1, hidden_size=50, num_layers=3, dropout=0.2):
        super(PyTorchLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 25)
        self.fc2 = nn.Linear(25, 1)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        lstm_out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(lstm_out[:, -1, :])
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out

def train_pytorch_model(model, train_loader, val_loader, epochs=50):
    """Train PyTorch model"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred = model(X_batch)
                loss = criterion(y_pred.squeeze(), y_batch)
                val_loss += loss.item()
        
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Train Loss: {train_losses[-1]:.6f}, Val Loss: {val_losses[-1]:.6f}')
    
    return train_losses, val_losses

def plot_results(y_true, keras_pred, pytorch_pred, dates, keras_smape, pytorch_smape):
    """Plot the results"""
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Predictions vs Actual
    plt.subplot(2, 2, 1)
    plt.plot(dates, y_true, label='Actual', marker='o', linewidth=2)
    plt.plot(dates, keras_pred, label=f'Keras (SMAPE: {keras_smape:.2f}%)', marker='s', linewidth=2)
    plt.plot(dates, pytorch_pred, label=f'PyTorch (SMAPE: {pytorch_smape:.2f}%)', marker='^', linewidth=2)
    plt.title('Sales Forecasting: Actual vs Predicted')
    plt.xlabel('Date')
    plt.ylabel('Amount')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Residuals for Keras
    plt.subplot(2, 2, 2)
    keras_residuals = y_true - keras_pred
    plt.plot(dates, keras_residuals, marker='o', color='red', alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.title('Keras Model Residuals')
    plt.xlabel('Date')
    plt.ylabel('Residual')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Residuals for PyTorch
    plt.subplot(2, 2, 3)
    pytorch_residuals = y_true - pytorch_pred
    plt.plot(dates, pytorch_residuals, marker='o', color='blue', alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.title('PyTorch Model Residuals')
    plt.xlabel('Date')
    plt.ylabel('Residual')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Model Comparison
    plt.subplot(2, 2, 4)
    models = ['Keras', 'PyTorch']
    smape_values = [keras_smape, pytorch_smape]
    colors = ['orange', 'green']
    bars = plt.bar(models, smape_values, color=colors, alpha=0.7)
    plt.title('Model Performance Comparison (SMAPE)')
    plt.ylabel('SMAPE (%)')
    plt.ylim(0, max(smape_values) * 1.2)
    
    # Add value labels on bars
    for bar, value in zip(bars, smape_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(smape_values)*0.01,
                f'{value:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def main():
    # Load and prepare data
    df = load_and_prepare_data('merged_sum_daily_agg_filled.csv')
    
    # Prepare data for modeling
    data = df['amount'].values
    
    # Split data (last 10 for testing)
    train_size = len(data) - 10
    train_data = data[:train_size]
    test_data = data[train_size:]
    test_dates = df.index[train_size:]
    
    # Scale the data
    scaler = MinMaxScaler()
    train_data_scaled = scaler.fit_transform(train_data.reshape(-1, 1)).flatten()
    test_data_scaled = scaler.transform(test_data.reshape(-1, 1)).flatten()
    
    # Create sequences
    seq_length = 7  # Use 7 days to predict next day
    X_train, y_train = create_sequences(train_data_scaled, seq_length)
    
    print(f"Training sequences shape: {X_train.shape}")
    print(f"Training targets shape: {y_train.shape}")
    
    # Split training data into train and validation
    val_size = int(0.2 * len(X_train))
    X_val, y_val = X_train[-val_size:], y_train[-val_size:]
    X_train, y_train = X_train[:-val_size], y_train[:-val_size]
    
    # Reshape for LSTM (samples, time steps, features)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
    
    # ==================== KERAS MODEL ====================
    print("\n" + "="*50)
    print("TRAINING KERAS MODEL")
    print("="*50)
    
    keras_model = build_keras_model((seq_length, 1))
    
    # Train Keras model
    history = keras_model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val),
        verbose=1
    )
    
    # ==================== PYTORCH MODEL ====================
    print("\n" + "="*50)
    print("TRAINING PYTORCH MODEL")
    print("="*50)
    
    # Prepare PyTorch data
    train_dataset = SalesDataset(X_train.squeeze(), y_train)
    val_dataset = SalesDataset(X_val.squeeze(), y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    pytorch_model = PyTorchLSTM()
    train_losses, val_losses = train_pytorch_model(pytorch_model, train_loader, val_loader)
    
    # ==================== PREDICTIONS ====================
    print("\n" + "="*50)
    print("MAKING PREDICTIONS")
    print("="*50)
    
    # Make predictions on test data
    keras_predictions = []
    pytorch_predictions = []
    
    # Use the last seq_length points from training for initial prediction
    current_sequence = train_data_scaled[-seq_length:].reshape(1, seq_length, 1)
    current_sequence_torch = torch.FloatTensor(train_data_scaled[-seq_length:]).unsqueeze(0)
    
    for i in range(10):  # Predict next 10 days
        # Keras prediction
        keras_pred = keras_model.predict(current_sequence, verbose=0)[0, 0]
        keras_predictions.append(keras_pred)
        
        # PyTorch prediction
        pytorch_model.eval()
        with torch.no_grad():
            pytorch_pred = pytorch_model(current_sequence_torch).item()
        pytorch_predictions.append(pytorch_pred)
        
        # Update sequences for next prediction
        current_sequence = np.roll(current_sequence, -1, axis=1)
        current_sequence[0, -1, 0] = keras_pred
        
        current_sequence_torch = torch.roll(current_sequence_torch, -1, dims=1)
        current_sequence_torch[0, -1] = pytorch_pred
    
    # Inverse transform predictions
    keras_predictions = scaler.inverse_transform(np.array(keras_predictions).reshape(-1, 1)).flatten()
    pytorch_predictions = scaler.inverse_transform(np.array(pytorch_predictions).reshape(-1, 1)).flatten()
    
    # ==================== EVALUATION ====================
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    # Calculate SMAPE
    keras_smape = smape(test_data, keras_predictions)
    pytorch_smape = smape(test_data, pytorch_predictions)
    
    print(f"Keras Model SMAPE: {keras_smape:.2f}%")
    print(f"PyTorch Model SMAPE: {pytorch_smape:.2f}%")
    
    # Calculate other metrics
    keras_mae = mean_absolute_error(test_data, keras_predictions)
    pytorch_mae = mean_absolute_error(test_data, pytorch_predictions)
    
    keras_rmse = np.sqrt(mean_squared_error(test_data, keras_predictions))
    pytorch_rmse = np.sqrt(mean_squared_error(test_data, pytorch_predictions))
    
    print(f"\nKeras Model:")
    print(f"  MAE: {keras_mae:.2f}")
    print(f"  RMSE: {keras_rmse:.2f}")
    
    print(f"\nPyTorch Model:")
    print(f"  MAE: {pytorch_mae:.2f}")
    print(f"  RMSE: {pytorch_rmse:.2f}")
    
    # Print actual vs predicted values
    print(f"\nActual vs Predicted Values:")
    print(f"{'Date':<12} {'Actual':<12} {'Keras':<12} {'PyTorch':<12}")
    print("-" * 50)
    for i, date in enumerate(test_dates):
        print(f"{date.strftime('%Y-%m-%d'):<12} {test_data[i]:<12.2f} {keras_predictions[i]:<12.2f} {pytorch_predictions[i]:<12.2f}")
    
    # Plot results
    plot_results(test_data, keras_predictions, pytorch_predictions, test_dates, keras_smape, pytorch_smape)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Keras Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('PyTorch Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()