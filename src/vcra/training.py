import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    root_mean_squared_log_error,
)
from loguru import logger

pd.set_option("display.max_columns", None)


def evaluate_clf(clf, X, y, train_index, test_index, include_indices=False):
    logger.info(
        f"Training with {len(train_index)} samples; Testing with {len(test_index)} samples"
    )

    # Get Train/Test Sets
    X_train, X_test = X.iloc[train_index].values, X.iloc[test_index].values
    y_train, y_test = y.iloc[train_index].values, y.iloc[test_index].values

    # Train Model on Selected Fold
    clf.fit(X_train, y_train)
    y_pred = np.clip(clf.predict(X_test), 0, 1)

    # Organize and Return Results
    result = dict(
        instance=clf,
        train_indices=train_index,
        test_indices=test_index,
        y_true=y_test,
        y_pred=y_pred,
        acc=clf.score(X_test, y_test),
        mae=mean_absolute_error(y_test, y_pred),
        rmse=root_mean_squared_error(y_test, y_pred),
        rmsle=root_mean_squared_log_error(y_test, y_pred),
    )

    if include_indices:
        result.update({"train_indices": train_index, "test_indices": test_index})

    return result


def train_mlp_vcra(data_dir, X_sub, y_sub, y_bin_sub, tag=""):
    # %% mlp_vcra_features = ['own_speed', 'own_course', 'target_speed', 'target_course', 'dist_euclid', 'azimuth_angle_target_to_own', 'rel_movement_direction']
    mlp_vcra_features = [
        "vessel_1_speed",
        "vessel_1_course",
        "vessel_2_speed",
        "vessel_2_course",
        "euclidian_dist",
        "azimuth_target_to_own",
        "rel_movement_direction",
    ]
    mlp_vcra_training_data = X_sub.loc[:, mlp_vcra_features].copy()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)
    regr = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            random_state=10,
            max_iter=300,
            hidden_layer_sizes=(256, 32),
            verbose=True,
            early_stopping=True,
            n_iter_no_change=7,
        ),
    )

    mlp_vcra_skf_results = Parallel(n_jobs=-1)(
        delayed(evaluate_clf)(
            regr, mlp_vcra_training_data, y_sub, train_index, test_index
        )
        for (train_index, test_index) in tqdm(
            skf.split(X_sub, y_bin_sub), total=skf.get_n_splits(X_sub, y_bin_sub)
        )
    )

    logger.info("Training completed. Saving results...")
    mlp_vcra_skf_results_df = pd.DataFrame(mlp_vcra_skf_results)
    mlp_vcra_skf_results_df.to_pickle(
        f"{data_dir}/mlp_vcra_skf_results_v14{tag}.pickle"
    )


def run(data_dir):
    # %% Loading and Preparing CRI Dataset
    gdf_vcra = pd.read_csv(f"{data_dir}/training_aisdk.csv")
    gdf_vcra.loc[:, "ves_cri_bin"] = pd.cut(
        gdf_vcra.ves_cri, bins=np.arange(0, 1.1, 0.2), right=True, include_lowest=True
    )

    ves_cri_bin_val_counts = gdf_vcra.ves_cri_bin.value_counts(sort=False)
    logger.info(ves_cri_bin_val_counts)
    ax = ves_cri_bin_val_counts.plot.bar()
    ax.set_yscale("log")
    plt.savefig(f"{data_dir}/training_aisdk.ves_cri.distribution.pdf", dpi=300)

    # %% Get a Stratified Subset (to ensure a "fair" comparison)
    X, y, y_bin = (
        gdf_vcra.iloc[:, :-2],
        gdf_vcra.iloc[:, -2],
        gdf_vcra.iloc[:, -1].astype("str"),
    )
    # X_sub, _, y_sub, _, y_bin_sub, _ = train_test_split(X, y, y_bin, train_size=0.35, random_state=10, stratify=y_bin)

    args = data_dir, X, y, y_bin, ".trained_on_all_data"
    train_mlp_vcra(*args)
