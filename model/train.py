import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, root_mean_squared_log_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from model import MLP

def train_mlp_vcra(X_sub, y_sub, y_bin_sub, device='cpu', epochs=100, batch_size=64, lr=1e-3, tag=''):
    mlp_vcra_features = ['own_speed', 'own_course_rad', 'target_speed', 'target_course_rad', 'dist_euclid', 'azimuth_angle_target_to_own', 'rel_movement_direction']
    X_data = X_sub.loc[:, mlp_vcra_features].values
    y_data = y_sub.values

    scaler = StandardScaler().fit(X_data)
    X_data = scaler.transform(X_data)
    
    # Initialize model and move to device
    model = MLP(input_size=X_data.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Cross-validations
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)
    results = []

    for train_index, test_index in skf.split(X_data, y_bin_sub):
        X_train, X_test = torch.tensor(X_data[train_index], dtype=torch.float32).to(device), torch.tensor(X_data[test_index], dtype=torch.float32).to(device)
        y_train, y_test = torch.tensor(y_data[train_index], dtype=torch.float32).to(device), torch.tensor(y_data[test_index], dtype=torch.float32).to(device)
        
        # Training loop
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            output = model(X_train).squeeze()
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test).squeeze().cpu().numpy()
            y_true = y_test.cpu().numpy()
        
        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred, squared=False)
        rmsle = root_mean_squared_log_error(y_true, y_pred, squared=False)

        # Store results
        results.append({
            'mae': mae,
            'rmse': rmse,
            'rmsle': rmsle
        })
        
    results_df = pd.DataFrame(results)
    
    data_dir = "data"

    os.makedirs(data_dir, exist_ok=True)
    
    results_df.to_feather(f'{data_dir}/mlp_vcra_skf_results_v14{tag}.feather')

if __name__ == "__main__":
    train_mlp_vcra()
    pass