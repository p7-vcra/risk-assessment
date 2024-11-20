import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import argparse

from sklearn.metrics import mean_absolute_error, root_mean_squared_error, root_mean_squared_log_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from model import MLP
from loguru import logger

CHECKPOINT_DIR = 'checkpoints'

def train_mlp_vcra(X_sub, y_sub, y_bin_sub, device='cpu', n_splits=5, epochs=100, batch_size=64, lr=1e-3, tag='', use_checkpoint=False):
    mlp_vcra_features = ['own_speed', 'own_course_rad', 'target_speed', 'target_course_rad', 
                         'dist_euclid', 'azimuth_angle_target_to_own', 'rel_movement_direction']
    X_data = X_sub.loc[:, mlp_vcra_features].values
    y_data = y_sub.values

    scaler = StandardScaler().fit(X_data)
    X_data = scaler.transform(X_data)
    
    model = MLP(input_size=X_data.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Check if checkpoints should be used and if there are existing checkpoints
    if use_checkpoint and os.listdir(CHECKPOINT_DIR):
        # Find the latest checkpoint based on lowest error
        checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pt')]
        latest_checkpoint = min(
            checkpoints, 
            key=lambda x: float(x.split('_')[-1].replace('.pt', ''))
        )
        
        checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, f'{latest_checkpoint}'), weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        latest_fold = int(checkpoint['fold'])
        latest_epoch = int(checkpoint['epoch'])
        
        start_fold = latest_fold + 1 if latest_epoch == epochs else latest_fold
        start_epoch = latest_epoch + 1 if latest_epoch < epochs else 1

        if start_fold <= n_splits:
            logger.info(f'Resuming training from fold: {start_fold}, epoch: {start_epoch}, current MSE loss {checkpoint["val_loss"]}')
    else:
        start_fold = 1
        start_epoch = 1

    skf = StratifiedKFold(n_splits, shuffle=True, random_state=10)
    results = []

    # Iterate over each fold, starting from the loaded fold if resuming
    for current_fold, (train_index, test_index) in enumerate(skf.split(X_data, y_bin_sub), start=1):
        # Skip completed folds if resuming from a checkpoint
        if current_fold < start_fold:
            continue
        
        logger.info(f'------------FOLD-{current_fold}------------')

        X_train, X_test = torch.tensor(X_data[train_index], dtype=torch.float32).to(device), torch.tensor(X_data[test_index], dtype=torch.float32).to(device)
        y_train, y_test = torch.tensor(y_data[train_index], dtype=torch.float32).to(device), torch.tensor(y_data[test_index], dtype=torch.float32).to(device)

        best_val_loss = float('inf')

        # Epoch loop starting from start_epoch for the current fold
        for epoch in range(start_epoch, epochs+1):
            logger.info(f"Training epoch {epoch}/{epochs} started")

            model.train()
            optimizer.zero_grad()
            output = model(X_train).squeeze()
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()

            # Evaluate on validation set
            model.eval()
            with torch.no_grad():
                val_output = model(X_test).squeeze()
                val_loss = criterion(val_output, y_test).item()

            # Save checkpoint if validation loss improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_filename = f'{n_splits}fold_{epochs}epoch_model_f{current_fold}_e{epoch}_{val_loss:.4f}.pt'
                torch.save({
                    'fold': current_fold,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss
                }, os.path.join(CHECKPOINT_DIR, checkpoint_filename))
                logger.info(f"Training epoch {epoch}/{epochs} finished. Validation loss improved to {val_loss:.4f}. Saved checkpoint: {checkpoint_filename}")
            else:
                logger.info(f"Training eEpoch {epoch}/{epochs}. Current validation loss: {val_loss:.4f}")

        # Reset start_epoch for the next fold
        start_epoch = 1

        # Final evaluation metrics
        y_pred = val_output.cpu().numpy()
        y_true = y_test.cpu().numpy()
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)
        rmsle = root_mean_squared_log_error(y_true, y_pred)

        results.append({'mae': mae, 'rmse': rmse, 'rmsle': rmsle})
        
    logger.info('Training completed. Saving results...')

    results_df = pd.DataFrame(results)
    
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    results_df.to_feather(f'{data_dir}/mlp_vcra_skf_results_v14{tag}.feather')


def generate_sample_data(num_samples=1000):    
    # Generate random data for features
    X_data = pd.DataFrame({
        'own_speed': np.random.rand(num_samples),
        'own_course_rad': np.random.rand(num_samples) * 2 * np.pi,  # Radians [0, 2π]
        'target_speed': np.random.rand(num_samples),
        'target_course_rad': np.random.rand(num_samples) * 2 * np.pi,
        'dist_euclid': np.random.rand(num_samples) * 100,  # Distance in some unit
        'azimuth_angle_target_to_own': np.random.rand(num_samples) * 2 * np.pi,
        'rel_movement_direction': np.random.rand(num_samples) * 2 * np.pi
    })
    
    # Generate target values (continuous) and binary labels for stratification
    y_data = pd.Series(np.random.rand(num_samples) * 100)  # Continuous target for regression
    y_bin_data = pd.Series(np.random.randint(0, 2, size=num_samples))  # Binary labels
    
    return X_data, y_data, y_bin_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MLP VCRA model')
    parser.add_argument('--use-checkpoint', '-uc', action='store_true', help='Flag to use checkpoint for resuming training')
    args = parser.parse_args()

    logger.add("logs/risk-assessment-{time:YYYYMMDD}.log", rotation="00:00")
    
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info(f"Using {device} device")

    # Generate sample data
    X_sub, y_sub, y_bin_sub = generate_sample_data(num_samples=10000)
    
    # Train the model with sample data
    train_mlp_vcra(X_sub, y_sub, y_bin_sub, device=device, epochs=10, use_checkpoint=args.use_checkpoint)
