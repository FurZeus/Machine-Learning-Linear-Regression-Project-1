
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_df = pd.read_csv('trainDATA.csv')
test_df  = pd.read_csv('testDATA.csv')

FUEL_CATEGORIES   = ['CNG', 'Diesel', 'LPG', 'Petrol']
SELLER_CATEGORIES = ['Dealer', 'Individual', 'Trustmark Dealer']

OWNER_MAP = {
    'Test Drive Car'      : 0,
    'First Owner'         : 1,
    'Second Owner'        : 2,
    'Third Owner'         : 3,
    'Fourth & Above Owner': 4
}

SCALE_COLS = ['year', 'km_driven', 'owner']

def preprocess(df, fit_scalers=None):
    df = df.copy()

    df.drop(columns=['name'], inplace=True)

    df['owner'] = df['owner'].map(OWNER_MAP).fillna(1)

    df['transmission'] = df['transmission'].map({'Manual': 0, 'Automatic': 1}).fillna(0)

    for cat in FUEL_CATEGORIES:
        df[f'fuel_{cat}'] = (df['fuel'] == cat).astype(int)
    df.drop(columns=['fuel'], inplace=True)

    for cat in SELLER_CATEGORIES:
        df[f'seller_{cat}'] = (df['seller_type'] == cat).astype(int)
    df.drop(columns=['seller_type'], inplace=True)

    if fit_scalers is None:
        scalers = {}
        for col in SCALE_COLS:
            mn, mx = df[col].min(), df[col].max()
            scalers[col] = (mn, mx)
            df[col] = (df[col] - mn) / (mx - mn + 1e-8)

        mn_sp, mx_sp = df['selling_price'].min(), df['selling_price'].max()
        scalers['selling_price'] = (mn_sp, mx_sp)
        df['selling_price'] = (df['selling_price'] - mn_sp) / (mx_sp - mn_sp + 1e-8)

        return df, scalers
    else:
        for col in SCALE_COLS:
            mn, mx = fit_scalers[col]
            df[col] = (df[col] - mn) / (mx - mn + 1e-8)

        mn_sp, mx_sp = fit_scalers['selling_price']
        df['selling_price'] = (df['selling_price'] - mn_sp) / (mx_sp - mn_sp + 1e-8)

        return df

train_processed, scalers = preprocess(train_df, fit_scalers=None)
test_processed  = preprocess(test_df, fit_scalers=scalers)

target_cols    = train_processed.columns.tolist()
test_processed = test_processed.reindex(columns=target_cols, fill_value=0)

print("\nIslenmiş sütunlar:", train_processed.columns.tolist())
print("Feature sayısı   :", train_processed.shape[1] - 1)

feature_cols = [c for c in train_processed.columns if c != 'selling_price']

X_train = train_processed[feature_cols].values
y_train = train_processed['selling_price'].values
X_test  = test_processed[feature_cols].values
y_test  = test_processed['selling_price'].values

X_train_b = X_train
X_test_b  = X_test

N, p = X_train_b.shape
print(f"\nX_train shape: {X_train_b.shape}  |  y_train shape: {y_train.shape}")

def compute_cost(X, y, theta):
    m = len(y)
    errors = X @ theta - y
    return (1 / (2 * m)) * np.dot(errors, errors)


def compute_gradient(X, y, theta):
    m = len(y)
    errors = X @ theta - y
    return (1 / m) * (X.T @ errors)

def gradient_descent(X, y, alpha, max_iter=10000, tol=1e-6):
    m, n = X.shape
    theta = np.zeros(n)
    cost_history = []

    for i in range(max_iter):
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)

        grad  = compute_gradient(X, y, theta)
        theta = theta - alpha * grad

        if i > 0 and abs(cost_history[-2] - cost_history[-1]) < tol:
            print(f"  → Yakınsama sağlandı: iterasyon {i+1}, J={cost:.6f}")
            break

    return theta, cost_history

learning_rates = [0.001, 0.01, 0.05, 0.1]
results = {}

print("\n" + "="*55)
print(f"{'Learning Rate':>15} | {'İter':>6} | {'Son J (train)':>14} | {'Test MSE':>10}")
print("="*55)

for alpha in learning_rates:
    theta, cost_hist = gradient_descent(X_train_b, y_train, alpha=alpha,
                                        max_iter=10000, tol=1e-6)
    y_pred_test = X_test_b @ theta
    test_mse    = np.mean((y_pred_test - y_test) ** 2)
    results[alpha] = {
        'theta'       : theta,
        'cost_history': cost_hist,
        'test_mse'    : test_mse
    }
    print(f"  α = {alpha:<10} | {len(cost_hist):>6} | {cost_hist[-1]:>14.6f} | {test_mse:>10.6f}")

best_alpha = min(results, key=lambda a: results[a]['test_mse'])
best_theta = results[best_alpha]['theta']

y_pred_scaled = X_test_b @ best_theta

mn_sp, mx_sp = scalers['selling_price']
y_pred_real  = y_pred_scaled * (mx_sp - mn_sp) + mn_sp
y_test_real  = y_test        * (mx_sp - mn_sp) + mn_sp

mse_s  = np.mean((y_pred_scaled - y_test) ** 2)
rmse_s = np.sqrt(mse_s)
mae_s  = np.mean(np.abs(y_pred_scaled - y_test))

mse_r  = np.mean((y_pred_real - y_test_real) ** 2)
rmse_r = np.sqrt(mse_r)
mae_r  = np.mean(np.abs(y_pred_real - y_test_real))

ss_res = np.sum((y_test_real - y_pred_real) ** 2)
ss_tot = np.sum((y_test_real - np.mean(y_test_real)) ** 2)
r2 = 1 - ss_res / ss_tot

print(f"\n{'='*40}")
print(f"  En iyi α             : {best_alpha}")
print(f"  Test MSE  (scaled)   : {mse_s:.6f}")
print(f"  Test RMSE (scaled)   : {rmse_s:.6f}")
print(f"  Test MAE  (scaled)   : {mae_s:.6f}")
print(f"  Test RMSE (gerçek)   : {rmse_r:,.0f} INR")
print(f"  Test MAE  (gerçek)   : {mae_r:,.0f} INR")
print(f"  R² Skoru             : {r2:.4f}")

print(f"\n--- Theta ({len(feature_cols)} feature) ---")
col_names = feature_cols
for name, val in zip(col_names, best_theta):
    print(f"  {name:35s}: {val:+.4f}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Linear Regression - Gradient Descent Analizi', fontsize=14, fontweight='bold')

colors = ['red', 'blue', 'green', 'orange']

ax1 = axes[0, 0]
for i, alpha in enumerate(learning_rates):
    hist = results[alpha]['cost_history']
    ax1.plot(range(len(hist)), hist, label=f'α={alpha}', color=colors[i], linewidth=2)
ax1.set_xlabel('İterasyon')
ax1.set_ylabel('J (Maliyet)')
ax1.set_title('Farklı Öğrenme Hızları: Maliyet')
ax1.legend()
ax1.set_yscale('log')
ax1.grid()

ax2 = axes[0, 1]
best_hist = results[best_alpha]['cost_history']
ax2.plot(range(len(best_hist)), best_hist, color='green', linewidth=2)
ax2.set_xlabel('İterasyon')
ax2.set_ylabel('J (Maliyet)')
ax2.set_title(f'En İyi Öğrenme Hızı (α={best_alpha}): Maliyet')
ax2.grid()

ax3 = axes[1, 0]
ax3.scatter(y_test_real, y_pred_real, alpha=0.4, s=20, color='blue')
lim_min = min(y_test_real.min(), y_pred_real.min())
lim_max = max(y_test_real.max(), y_pred_real.max())
ax3.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', linewidth=2, label='Mükemmel tahmin')
ax3.set_xlabel('Gerçek Fiyat (INR)')
ax3.set_ylabel('Tahmin Fiyat (INR)')
ax3.set_title(f'Tahmin vs Gerçek  (R²={r2:.3f})')
ax3.legend()
ax3.grid()

ax4 = axes[1, 1]
alphas_str = [str(a) for a in learning_rates]
test_mses  = [results[a]['test_mse'] for a in learning_rates]
bars = ax4.bar(alphas_str, test_mses, color=colors, edgecolor='black', linewidth=0.8)
ax4.set_xlabel('Öğrenme Hızı (α)')
ax4.set_ylabel('Test MSE (scaled)')
ax4.set_title('Öğrenme Hızlarına Göre Test MSE')
for bar, val in zip(bars, test_mses):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.00001,
             f'{val:.5f}', ha='center', va='bottom', fontsize=9)
ax4.grid()

plt.tight_layout()
plt.show()

def gradient_descent_track_theta(X, y, alpha, max_iter=10000, tol=1e-6, n_track=5):
    m, n = X.shape
    theta = np.zeros(n)
    cost_history  = []
    theta_history = []

    for i in range(max_iter):
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)
        theta_history.append(theta[:n_track].copy())

        grad  = compute_gradient(X, y, theta)
        theta = theta - alpha * grad

        if i > 0 and abs(cost_history[-2] - cost_history[-1]) < tol:
            break

    return theta, cost_history, np.array(theta_history)


theta_best, cost_hist_best, theta_hist = gradient_descent_track_theta(
    X_train_b, y_train, alpha=best_alpha, max_iter=10000)

fig2, ax = plt.subplots(figsize=(10, 5))
labels = feature_cols[:5]
for i in range(5):
    ax.plot(range(len(theta_hist)), theta_hist[:, i], label=labels[i], linewidth=2)
ax.set_xlabel('İterasyon')
ax.set_ylabel('θ Değeri')
ax.set_title(f'Theta Evrim Grafiği (α={best_alpha}, ilk 5 parametre)')
ax.legend(fontsize=8)
ax.grid()
plt.tight_layout()
plt.show()