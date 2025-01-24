#1
import numpy as np
from ct_support_code import *

data = np.load('ct_data.npz')
X_train = data['X_train']; X_val = data['X_val']; X_test = data['X_test']
y_train = data['y_train']; y_val = data['y_val']; y_test = data['y_test']
#a
# Mean of training output data and SE
mu_y_train = np.mean(y_train)
se_y_train = np.std(y_train, ddof=1) / np.sqrt(len(y_train))

# Mean of validation output data and SE
mu_y_val = np.mean(y_val)
se_y_val = np.std(y_val, ddof=1) / np.sqrt(len(y_val))

# Mean of first 5785 entries in training output data and SE
mu_y_first = np.mean(y_train[:5785])
se_y_first = np.std(y_train[:5785], ddof=1) / np.sqrt(len(y_train[:5785]))

print("Mean of y_train is {:.4f} (SE = {:.4f}).".format(mu_y_train, se_y_train))
print("Mean of y_val is {:.4f} (SE = {:.4f}).".format(mu_y_val, se_y_val))
print("Mean of first 5,785 of y_train is {:.4f} (SE = {:.4f}).".format(mu_y_first, se_y_first))
#b

# Indices of constant features
zero_std_ind = np.where(np.std(X_train, axis=0) == 0)[0]
print("Indexes of constant features in original training data:\n", zero_std_ind)

# Remove constant features from training data
X_train = np.delete(X_train, zero_std_ind, axis=1)

# Find unique features
_, unique_ind = np.unique(X_train, axis=1, return_index=True)

# Indices of duplicate features
duplicate_ind = np.setdiff1d(np.arange(X_train.shape[1]), unique_ind)
print("Indexes of duplicated features after removing constant features:\n", duplicate_ind)

# Keep only unique features
X_train = X_train[:, np.sort(unique_ind)]

# Apply the same filters to validation and test data
X_val = np.delete(X_val, zero_std_ind, axis=1)
X_val = X_val[:, np.sort(unique_ind)]
X_test = np.delete(X_test, zero_std_ind, axis=1)
X_test = X_test[:, np.sort(unique_ind)]
#2
def fit_linreg(X, yy, alpha):
    """
    Fits a linear regression model with regularization using least squares.
    """
    N, K = X.shape

    # Add column for bias
    X_design = np.hstack((np.ones((N, 1)), X))

    # Regularization matrix
    reg_block = np.hstack((np.zeros((K, 1)), np.sqrt(alpha) * np.eye(K)))
    reg_matrix = np.vstack((np.zeros((1, K+1)), reg_block))

    # Augment design matrix with regularization matrix
    X_aug = np.vstack((X_design, reg_matrix))

    # Augment output vector
    yy_aug = np.hstack((yy, np.zeros((K+1))))

    # Solve using least squares
    w = np.linalg.lstsq(X_aug, yy_aug, rcond=None)[0]

    return w

# Fit training data using least squares
w_lstsq = fit_linreg(X_train, y_train, alpha=30)

def predict_lstsq(X, w):
    """
    Predicts using fitted weights.
    """
    pred = np.dot(X, w)
    return pred

def rmse(target, prediction):
    """
    Computes Root Mean Squared Error.
    """
    return np.sqrt(np.mean((target - prediction) ** 2))

# Training predictions
X_train_design = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
pred_train_lstsq = predict_lstsq(X_train_design, w_lstsq)

# Training RMSE
rmse_train_lstsq = rmse(y_train, pred_train_lstsq)

# Validation predictions
X_val_design = np.hstack((np.ones((X_val.shape[0], 1)), X_val))
pred_val_lstsq = predict_lstsq(X_val_design, w_lstsq)

# Validation RMSE
rmse_val_lstsq = rmse(y_val, pred_val_lstsq)

print("Least squares: Training RMSE = {:.4f}; Validation RMSE = {:.4f}.".format(rmse_train_lstsq, rmse_val_lstsq))



# Fit training data using gradient-based optimization
ww_gradopt, bb_gradopt = fit_linreg_gradopt(X_train, y_train, alpha=30)

# Training predictions
pred_train_gradopt = np.dot(X_train, ww_gradopt) + bb_gradopt

# Training RMSE
rmse_train_gradopt = rmse(y_train, pred_train_gradopt)

# Validation predictions
pred_val_gradopt = np.dot(X_val, ww_gradopt) + bb_gradopt

# Validation RMSE
rmse_val_gradopt = rmse(y_val, pred_val_gradopt)

print("Gradient-based opt.: Training RMSE = {:.4f}; Validation RMSE = {:.4f}.".format(rmse_train_gradopt, rmse_val_gradopt))
#3
# Number of thresholds
K = 20

mx = np.max(y_train); mn = np.min(y_train); hh = (mx - mn) / (K + 1)
thresholds = np.linspace(mn + hh, mx - hh, num=K, endpoint=True)

def fit_logreg_gradopt(X, yy, alpha):
    """
    Fits logistic regression using gradient-based optimization.
    """
    D = X.shape[1]
    args = (X, yy, alpha)

    # Initialize weights and bias
    init = (np.zeros(D), np.array(0))

    # Minimize cost function
    ww, bb = minimize_list(logreg_cost, init, args)

    return ww, bb

# Initialize weights for logistic regression models
W_hidden = np.zeros((K, X_train.shape[1]))

# Initialize biases for logistic regression models
bb_hidden = np.zeros((K,))

for kk in range(K):
    labels = y_train > thresholds[kk]
    ww, bb = fit_logreg_gradopt(X_train, labels, alpha=30)
    W_hidden[kk, :] = ww
    bb_hidden[kk] = bb

def logistic(x):
    """
    Computes the logistic (sigmoid) function.
    """
    return 1. / (1. + np.exp(-x))

# Predicted probabilities on training
preds_train = logistic(np.dot(X_train, W_hidden.T) + bb_hidden[None, :])

# Predicted probabilities on validation
preds_val = logistic(np.dot(X_val, W_hidden.T) + bb_hidden[None, :])

# Fit linear regression on predicted probabilities
ww_out, bb_out = fit_linreg_gradopt(preds_train, y_train, alpha=30)

# Output predictions on training
pred_train_out = np.dot(preds_train, ww_out) + bb_out

# Training RMSE
rmse_train_out = rmse(y_train, pred_train_out)

# Output predictions on validation
pred_val_out = np.dot(preds_val, ww_out) + bb_out

# Validation RMSE
rmse_val_out = rmse(y_val, pred_val_out)

print("Classification approach: Training RMSE = {:.4f}; Validation RMSE = {:.4f}.".format(rmse_train_out, rmse_val_out))

#4
np.random.seed(100)

# Initialize weights for output layer
ww_0 = np.random.randn(K,) / np.sqrt(K)

# Initialize bias for output layer
bb_0 = np.random.randn(1)

# Initialize weights for hidden layer
V_0 = np.random.rand(K, X_train.shape[1]) / np.sqrt(X_train.shape[1])

# Initialize biases for hidden layer
bk_0 = np.random.randn(K,)

# Random initialization
init_rand = (ww_0, bb_0, V_0, bk_0)

def fit_nn_gradopt(X, yy, alpha, init):
    """
    Fits a neural network with one hidden layer using gradient optimization.
    """
    args = (X, yy, alpha)
    ww, bb, V, bk = minimize_list(nn_cost, init, args)
    return ww, bb, V, bk

# Fit neural network with random initialization
ww_rand, bb_rand, V_rand, bk_rand = fit_nn_gradopt(X_train, y_train, alpha=0, init=init_rand)

# Fit neural network using parameters from Q3 as initialization
init_q3 = (ww_out, bb_out, W_hidden, bb_hidden)
ww_q3, bb_q3, V_q3, bk_q3 = fit_nn_gradopt(X_train, y_train, alpha=0, init=init_q3)


# Predictions and RMSE for random initialization
pred_train_rand = nn_cost(params=(ww_rand, bb_rand, V_rand, bk_rand), X=X_train)
rmse_train_rand = rmse(y_train, pred_train_rand)
pred_val_rand = nn_cost(params=(ww_rand, bb_rand, V_rand, bk_rand), X=X_val)
rmse_val_rand = rmse(y_val, pred_val_rand)

print("Random init.: Training RMSE = {:.4f}; Validation RMSE = {:.4f}.".format(rmse_train_rand, rmse_val_rand))

# Predictions and RMSE for Q3 initialization
pred_train_q3 = nn_cost(params=(ww_q3, bb_q3, V_q3, bk_q3), X=X_train)
rmse_train_q3 = rmse(y_train, pred_train_q3)
pred_val_q3 = nn_cost(params=(ww_q3, bb_q3, V_q3, bk_q3), X=X_val)
rmse_val_q3 = rmse(y_val, pred_val_q3)

print("Q3 init.: Training RMSE = {:.4f}; Validation RMSE = {:.4f}.".format(rmse_train_q3, rmse_val_q3))

# Fit regularized neural network with random initialization
ww_rand_reg, bb_rand_reg, V_rand_reg, bk_rand_reg = fit_nn_gradopt(X_train, y_train, alpha=30, init=init_rand)

# Fit regularized neural network using parameters from Q3 as initialization and
ww_q3_reg, bb_q3_reg, V_q3_reg, bk_q3_reg = fit_nn_gradopt(X_train, y_train, alpha=30, init=init_q3)

# Predictions and RMSE for random initialization with regularization
pred_train_rand_reg = nn_cost(params=(ww_rand_reg, bb_rand_reg, V_rand_reg, bk_rand_reg), X=X_train)
rmse_train_rand_reg = rmse(y_train, pred_train_rand_reg)
pred_val_rand_reg = nn_cost(params=(ww_rand_reg, bb_rand_reg, V_rand_reg, bk_rand_reg), X=X_val)
rmse_val_rand_reg = rmse(y_val, pred_val_rand_reg)

print("Random init. with reg.: Training RMSE = {:.4f}; Validation RMSE = {:.4f}.".format(rmse_train_rand_reg, rmse_val_rand_reg))

# Predictions and RMSE for Q3 initialization with regularization
pred_train_q3_reg = nn_cost(params=(ww_q3_reg, bb_q3_reg, V_q3_reg, bk_q3_reg), X=X_train)
rmse_train_q3_reg = rmse(y_train, pred_train_q3_reg)
pred_val_q3_reg = nn_cost(params=(ww_q3_reg, bb_q3_reg, V_q3_reg, bk_q3_reg), X=X_val)
rmse_val_q3_reg = rmse(y_val, pred_val_q3_reg)

print("Q3 init. with reg.: Training RMSE = {:.4f}; Validation RMSE = {:.4f}.".format(rmse_train_q3_reg, rmse_val_q3_reg))
#5
def train_nn_reg(X_train, y_train, alpha, init, X_val, y_val):
    """
    Trains a neural network with regularization and returns validation RMSE.
    """
    ww, bb, V, bk = fit_nn_gradopt(X_train, y_train, alpha, init)
    pred_val = nn_cost(params=(ww, bb, V, bk), X=X_val)
    rmse_val = rmse(y_val, pred_val)
    return rmse_val

from scipy.stats import norm

def pi(mu, sigma, yy):
    """
    Computes the probability of improvement (PI).
    """
    best_yy = max(yy)
    z = (mu - best_yy) / sigma
    return norm.cdf(z)

# Initialize alpha values
alpha = np.arange(0, 50, 0.02)

np.random.seed(100)

# Randomly select initial alphas as observations
alpha_obs = np.random.choice(alpha, 3, replace=False)

# Compute log-RMSE observations
logrmse_obs = np.array([train_nn_reg(X_train, y_train, alpha, init_rand, X_val, y_val) for alpha in alpha_obs])

# Subtract log-RMSE from baseline log-RMSE
f_obs = np.log(rmse_val_rand) - np.log(logrmse_obs)

pi_max_vec = []
alpha_next_vec = []

for i in np.arange(5):
    alpha_rest = np.setdiff1d(alpha, alpha_obs)

    # Compute posterior mean and covariance
    rest_cond_mu, rest_cond_cov = gp_post_par(alpha_rest, alpha_obs, f_obs)
    rest_cond_sigma = np.sqrt(np.diag(rest_cond_cov))

    # Compute PI and select next alpha
    pi_rest = pi(rest_cond_mu, rest_cond_sigma, f_obs)
    alpha_next = alpha_rest[np.argmax(pi_rest)]
    alpha_obs = np.concatenate((alpha_obs, alpha_next.reshape((1,))))
    logrmse_next = train_nn_reg(X_train, y_train, alpha_next, init_rand, X_val, y_val)
    f_next = np.log(rmse_val_rand) - np.log(logrmse_next)
    f_obs = np.concatenate((f_obs, f_next.reshape((1,))))

    pi_max_vec.append(pi_rest[np.argmax(pi_rest)])
    alpha_next_vec.append(alpha_next)

# Determine the best alpha and train the model
alpha_best = alpha_obs[np.argmax(f_obs)]
ww_best, bb_best, V_best, bk_best = fit_nn_gradopt(X_train, y_train, alpha_best, init_rand)

# Compute RMSE for the best alpha on validation and test
pred_val_best = nn_cost(params=(ww_best, bb_best, V_best, bk_best), X=X_val)
rmse_val_best = rmse(y_val, pred_val_best)
pred_test_best = nn_cost(params=(ww_best, bb_best, V_best, bk_best), X=X_test)
rmse_test_best = rmse(y_test, pred_test_best)

print("Probabilities of improvement in each iteration:\n", np.round(pi_max_vec, 4))
print("Alphas chosen in each iteration:\n", np.round(alpha_next_vec, 4))
print("Best alpha found: {:.4f}.".format(alpha_best))
print("Bayesian opt.: Validation RMSE = {:.4f}, Test RMSE = {:.4f}.".format(rmse_val_best, rmse_test_best))
#6
np.random.seed(100)

noise_sd = np.std(X_train, ddof=1, axis=0)*1.35
noise = np.random.normal(0, scale=noise_sd, size=X_train.shape)
X_train_noise = X_train + noise

ww_noise, bb_noise, V_noise, bk_noise = fit_nn_gradopt(X_train_noise, y_train, alpha=0, init=init_rand)

pred_train_noise = nn_cost(params=(ww_noise, bb_noise, V_noise, bk_noise), X=X_train_noise)
rmse_train_noise = rmse(y_train, pred_train_noise)
pred_val_noise = nn_cost(params=(ww_noise, bb_noise, V_noise, bk_noise), X=X_val)
rmse_val_noise = rmse(y_val, pred_val_noise)
pred_test_noise = nn_cost(params=(ww_noise, bb_noise, V_noise, bk_noise), X=X_test)
rmse_test_noise = rmse(y_test, pred_test_noise)

print("Noise: Training RMSE = {:.4f}; Validation RMSE = {:.4f}; Test RMSE = {:.4f}".format(rmse_train_noise, rmse_val_noise, rmse_test_noise))
