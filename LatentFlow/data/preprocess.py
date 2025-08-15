import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

def preprocess_and_save(data_dir):
    # Load data
    cp_1200 = np.load(os.path.join(data_dir, "Cp_1200.npy"))       # (1200, 30)
    u = np.load(os.path.join(data_dir, "velocities_u.npy"))        # (1200, 367, 383)
    v = np.load(os.path.join(data_dir, "velocities_v.npy"))         # (1200, 367, 383)

    assert u.shape == v.shape and u.shape[0] == cp_1200.shape[0]

    # Flatten for scaler
    u_flat = u.reshape((u.shape[0], -1))
    v_flat = v.reshape((v.shape[0], -1))

    # Standardize
    scaler_u = StandardScaler().fit(u_flat)
    scaler_v = StandardScaler().fit(v_flat)

    u_norm = scaler_u.transform(u_flat).reshape(u.shape)
    v_norm = scaler_v.transform(v_flat).reshape(v.shape)

    # Stack to get (N, 2, H, W)
    vel_stack = np.stack([u_norm, v_norm], axis=1)

    # Save normalized data
    np.save(os.path.join(data_dir, "velocities_normalized.npy"), vel_stack)

    # Save scalers
    joblib.dump(scaler_u, os.path.join(data_dir, "scaler_u.pkl"))
    joblib.dump(scaler_v, os.path.join(data_dir, "scaler_v.pkl"))

    print("Normalized data saved:")
    print(f"velocities_normalized.npy: shape = {vel_stack.shape}")
    print("scaler_u.pkl, scaler_v.pkl saved")

if __name__ == "__main__":
    preprocess_and_save("/workspace/ljl/Junle_PIV_data/LatentFlow/data")  # current directory