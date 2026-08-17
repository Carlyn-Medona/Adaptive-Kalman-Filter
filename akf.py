"""
Adaptive Kalman Filtering for State Estimation
------------------------------------------------
Filters:
1. Standard Kalman Filter
2. Innovation-based Adaptive Kalman Filter
3. Sage-Husa Adaptive Kalman Filter
4. Wavelet-MRA + Kalman Filter


"""

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. SYSTEM MODEL
# ============================================================

def create_system(dt=0.1):

    # State:
    # x = [position, velocity]^T

    F = np.array([
        [1.0, dt],
        [0.0, 1.0]
    ])

    H = np.array([
        [1.0, 0.0]
    ])

    # Process noise covariance
    Q = np.array([
        [0.002, 0.0],
        [0.0, 0.001]
    ])

    return F, H, Q


# ============================================================
# 2. TRUE SYSTEM + MEASUREMENT GENERATION
# ============================================================

def simulate_system(N=500, dt=0.1, seed=42):

    rng = np.random.default_rng(seed)

    F, H, Q = create_system(dt)

    # True state
    x_true = np.zeros((N, 2))

    # Initial condition
    x_true[0] = [0.0, 1.0]

    # Generate system trajectory
    for k in range(1, N):

        process_noise = rng.multivariate_normal(
            mean=[0.0, 0.0],
            cov=Q
        )

        x_true[k] = F @ x_true[k-1] + process_noise

    # --------------------------------------------------------
    # CURRENTLY: simple Gaussian measurement noise
    #
    # This section is intentionally kept modular.
    # Later you can replace this with:
    # Gaussian / impulsive / coloured / periodic /
    # time-varying noise.
    # --------------------------------------------------------

    measurement_std = 0.5

    measurement_noise = rng.normal(
        0.0,
        measurement_std,
        N
    )

    z = x_true[:, 0] + measurement_noise

    R = np.array([[measurement_std**2]])

    return x_true, z, F, H, Q, R


# ============================================================
# 3. STANDARD KALMAN FILTER
# ============================================================

def standard_kf(z, F, H, Q, R, x0, P0):

    N = len(z)

    x_est = np.zeros((N, 2))

    x = x0.copy()
    P = P0.copy()

    for k in range(N):

        # Prediction
        if k == 0:
            x_pred = x
            P_pred = P
        else:
            x_pred = F @ x
            P_pred = F @ P @ F.T + Q

        # Innovation
        innovation = z[k] - (H @ x_pred)[0]

        # Innovation covariance
        S = H @ P_pred @ H.T + R

        # Kalman gain
        K = P_pred @ H.T @ np.linalg.inv(S)

        # Update
        x = x_pred + K[:, 0] * innovation

        P = (
            np.eye(2) - K @ H
        ) @ P_pred

        x_est[k] = x

    return x_est


# ============================================================
# 4. INNOVATION-BASED ADAPTIVE KALMAN FILTER
# ============================================================

def innovation_adaptive_kf(
        z,
        F,
        H,
        Q,
        R_initial,
        x0,
        P0,
        alpha=0.02):

    N = len(z)

    x_est = np.zeros((N, 2))

    x = x0.copy()
    P = P0.copy()

    R = R_initial.copy()

    estimated_R = np.zeros(N)

    for k in range(N):

        # Prediction
        if k == 0:
            x_pred = x
            P_pred = P
        else:
            x_pred = F @ x
            P_pred = F @ P @ F.T + Q

        # Innovation
        innovation = z[k] - (H @ x_pred)[0]

        # Part of innovation covariance explained
        # by state uncertainty
        state_uncertainty = (
            H @ P_pred @ H.T
        )[0, 0]

        # Estimate measurement noise variance
        R_measured = (
            innovation**2 - state_uncertainty
        )

        # Prevent negative covariance
        R_measured = max(
            R_measured,
            1e-5
        )

        # Exponential adaptation
        R[0, 0] = (
            (1 - alpha) * R[0, 0]
            + alpha * R_measured
        )

        # Bound R for numerical stability
        R[0, 0] = np.clip(
            R[0, 0],
            1e-4,
            100.0
        )

        estimated_R[k] = R[0, 0]

        # Innovation covariance
        S = (
            H @ P_pred @ H.T
            + R
        )

        # Kalman gain
        K = (
            P_pred @ H.T
            @ np.linalg.inv(S)
        )

        # Update
        x = (
            x_pred
            + K[:, 0] * innovation
        )

        P = (
            np.eye(2) - K @ H
        ) @ P_pred

        x_est[k] = x

    return x_est, estimated_R


# ============================================================
# 5. SAGE-HUSA ADAPTIVE KALMAN FILTER
# ============================================================

def sage_husa_kf(
        z,
        F,
        H,
        Q,
        R_initial,
        x0,
        P0,
        forgetting_factor=0.02):

    N = len(z)

    x_est = np.zeros((N, 2))

    x = x0.copy()
    P = P0.copy()

    R = R_initial.copy()

    estimated_R = np.zeros(N)

    for k in range(N):

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        if k == 0:
            x_pred = x
            P_pred = P
        else:
            x_pred = F @ x
            P_pred = F @ P @ F.T + Q

        # ----------------------------------------------------
        # Innovation
        # ----------------------------------------------------

        innovation = (
            z[k] - (H @ x_pred)[0]
        )

        state_uncertainty = (
            H @ P_pred @ H.T
        )[0, 0]

        # ----------------------------------------------------
        # Sage-Husa style covariance estimation
        # ----------------------------------------------------

        R_instantaneous = (
            innovation**2
            - state_uncertainty
        )

        R_instantaneous = np.clip(
            R_instantaneous,
            1e-4,
            100.0
        )

        # Recursive covariance adaptation
        R[0, 0] = (
            (1 - forgetting_factor)
            * R[0, 0]
            +
            forgetting_factor
            * R_instantaneous
        )

        R[0, 0] = np.clip(
            R[0, 0],
            1e-4,
            100.0
        )

        estimated_R[k] = R[0, 0]

        # ----------------------------------------------------
        # Kalman update
        # ----------------------------------------------------

        S = (
            H @ P_pred @ H.T
            + R
        )

        K = (
            P_pred @ H.T
            @ np.linalg.inv(S)
        )

        x = (
            x_pred
            + K[:, 0] * innovation
        )

        P = (
            np.eye(2) - K @ H
        ) @ P_pred

        x_est[k] = x

    return x_est, estimated_R


# ============================================================
# 6. SIMPLE HAAR MULTI-RESOLUTION ANALYSIS
# ============================================================

def haar_mra_denoise(
        signal,
        level=3,
        threshold_scale=1.0):

    """
    Simple Haar-wavelet MRA.

    Decomposes the signal into:
        Approximation coefficients
        Detail coefficients

    High-frequency detail coefficients are
    thresholded and the signal is reconstructed.

    This is intentionally implemented without PyWavelets,
    so the prototype only needs NumPy + Matplotlib.
    """

    signal = np.asarray(signal, dtype=float)

    original_length = len(signal)

    # Pad signal to make its length divisible by 2^level
    block_size = 2**level

    padding = (
        -original_length
    ) % block_size

    if padding > 0:
        working_signal = np.pad(
            signal,
            (0, padding),
            mode="edge"
        )
    else:
        working_signal = signal.copy()

    approximation = working_signal.copy()

    details = []

    # --------------------------------------------------------
    # Forward Haar decomposition
    # --------------------------------------------------------

    for _ in range(level):

        even = approximation[::2]
        odd = approximation[1::2]

        detail = (
            even - odd
        ) / np.sqrt(2)

        approximation = (
            even + odd
        ) / np.sqrt(2)

        details.append(detail)

    # --------------------------------------------------------
    # Threshold detail coefficients
    # --------------------------------------------------------

    for i in range(len(details)):

        d = details[i]

        # Median absolute deviation estimate
        sigma = (
            np.median(
                np.abs(
                    d - np.median(d)
                )
            ) / 0.6745
        )

        threshold = (
            threshold_scale
            * sigma
            * np.sqrt(
                2 * np.log(
                    max(len(signal), 2)
                )
            )
        )

        # Soft thresholding
        details[i] = (
            np.sign(d)
            * np.maximum(
                np.abs(d) - threshold,
                0
            )
        )

    # --------------------------------------------------------
    # Inverse Haar transform
    # --------------------------------------------------------

    for detail in reversed(details):

        reconstructed = np.empty(
            2 * len(detail)
        )

        reconstructed[::2] = (
            approximation + detail
        ) / np.sqrt(2)

        reconstructed[1::2] = (
            approximation - detail
        ) / np.sqrt(2)

        approximation = reconstructed

    return approximation[:original_length]


# ============================================================
# 7. WAVELET-MRA + KALMAN FILTER
# ============================================================

def wavelet_mra_kf(
        z,
        F,
        H,
        Q,
        R,
        x0,
        P0,
        level=3):

    # MRA-based measurement preprocessing
    z_mra = haar_mra_denoise(
        z,
        level=level
    )

    # Feed reconstructed measurement
    # into standard Kalman filter
    x_est = standard_kf(
        z_mra,
        F,
        H,
        Q,
        R,
        x0,
        P0
    )

    return x_est, z_mra


# ============================================================
# 8. PERFORMANCE METRICS
# ============================================================

def calculate_rmse(true_state, estimated_state):

    position_rmse = np.sqrt(
        np.mean(
            (
                true_state[:, 0]
                - estimated_state[:, 0]
            )**2
        )
    )

    velocity_rmse = np.sqrt(
        np.mean(
            (
                true_state[:, 1]
                - estimated_state[:, 1]
            )**2
        )
    )

    return position_rmse, velocity_rmse


# ============================================================
# 9. MAIN PROGRAM
# ============================================================

def main():

    # --------------------------------------------------------
    # Generate test system
    # --------------------------------------------------------

    x_true, z, F, H, Q, R = (
        simulate_system()
    )

    N = len(z)

    # Initial state estimate
    x0 = np.array([
        0.0,
        0.0
    ])

    # Initial covariance
    P0 = np.eye(2)

    # --------------------------------------------------------
    # Run filters
    # --------------------------------------------------------

    print("\nRunning filters...\n")

    # Standard KF
    x_kf = standard_kf(
        z,
        F,
        H,
        Q,
        R,
        x0,
        P0
    )

    # Innovation-based adaptive KF
    x_innovation, R_innovation = (
        innovation_adaptive_kf(
            z,
            F,
            H,
            Q,
            R,
            x0,
            P0
        )
    )

    # Sage-Husa KF
    x_sage, R_sage = (
        sage_husa_kf(
            z,
            F,
            H,
            Q,
            R,
            x0,
            P0
        )
    )

    # Wavelet-MRA KF
    x_mra, z_mra = (
        wavelet_mra_kf(
            z,
            F,
            H,
            Q,
            R,
            x0,
            P0
        )
    )

    # --------------------------------------------------------
    # Calculate RMSE
    # --------------------------------------------------------

    results = {}

    results["Standard KF"] = (
        calculate_rmse(
            x_true,
            x_kf
        )
    )

    results["Innovation Adaptive KF"] = (
        calculate_rmse(
            x_true,
            x_innovation
        )
    )

    results["Sage-Husa KF"] = (
        calculate_rmse(
            x_true,
            x_sage
        )
    )

    results["Wavelet-MRA KF"] = (
        calculate_rmse(
            x_true,
            x_mra
        )
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("=" * 60)
    print("FILTER PERFORMANCE")
    print("=" * 60)

    print(
        f"{'Filter':<30}"
        f"{'Position RMSE':>15}"
        f"{'Velocity RMSE':>15}"
    )

    print("-" * 60)

    for name, values in results.items():

        print(
            f"{name:<30}"
            f"{values[0]:>15.4f}"
            f"{values[1]:>15.4f}"
        )

    print("=" * 60)

    # --------------------------------------------------------
    # Time vector
    # --------------------------------------------------------

    t = np.arange(N) * 0.1

    # ========================================================
    # PLOT 1: POSITION ESTIMATION
    # ========================================================

    plt.figure(figsize=(10, 5))

    plt.plot(
        t,
        x_true[:, 0],
        label="True Position",
        linewidth=2
    )

    plt.plot(
        t,
        z,
        label="Measurement",
        alpha=0.4
    )

    plt.plot(
        t,
        x_kf[:, 0],
        label="Standard KF"
    )

    plt.plot(
        t,
        x_innovation[:, 0],
        label="Innovation Adaptive KF"
    )

    plt.plot(
        t,
        x_sage[:, 0],
        label="Sage-Husa KF"
    )

    plt.plot(
        t,
        x_mra[:, 0],
        label="Wavelet-MRA KF"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Position")

    plt.title(
        "Position State Estimation"
    )

    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # ========================================================
    # PLOT 2: VELOCITY ESTIMATION
    # ========================================================

    plt.figure(figsize=(10, 5))

    plt.plot(
        t,
        x_true[:, 1],
        label="True Velocity",
        linewidth=2
    )

    plt.plot(
        t,
        x_kf[:, 1],
        label="Standard KF"
    )

    plt.plot(
        t,
        x_innovation[:, 1],
        label="Innovation Adaptive KF"
    )

    plt.plot(
        t,
        x_sage[:, 1],
        label="Sage-Husa KF"
    )

    plt.plot(
        t,
        x_mra[:, 1],
        label="Wavelet-MRA KF"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Velocity")

    plt.title(
        "Velocity State Estimation"
    )

    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # ========================================================
    # PLOT 3: ADAPTIVE R ESTIMATION
    # ========================================================

    plt.figure(figsize=(10, 5))

    plt.plot(
        t,
        R_innovation,
        label="Innovation-based R"
    )

    plt.plot(
        t,
        R_sage,
        label="Sage-Husa R"
    )

    plt.axhline(
        R[0, 0],
        linestyle="--",
        label="True R"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Estimated Measurement Variance")

    plt.title(
        "Adaptive Measurement Noise Covariance"
    )

    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # ========================================================
    # PLOT 4: MRA MEASUREMENT
    # ========================================================

    plt.figure(figsize=(10, 5))

    plt.plot(
        t,
        z,
        label="Original Measurement",
        alpha=0.5
    )

    plt.plot(
        t,
        z_mra,
        label="MRA-Processed Measurement",
        linewidth=2
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Measurement")

    plt.title(
        "Wavelet-MRA Measurement Preprocessing"
    )

    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()