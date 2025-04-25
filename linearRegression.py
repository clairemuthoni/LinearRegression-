import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Generate 200 samples of x linearly spaced between 0 and 10
x = np.linspace(0, 10, 200)

# Generate corresponding y values using a function with some noise
# For this example, let's use: y = 2x² + 3x + 5 + noise
noise = np.random.normal(0, 5, 200)  # Gaussian noise with mean 0 and standard deviation 5
y = 2 * x**2 + 3 * x + 5 + noise

# Print the first few samples
print("First 5 samples:")
for i in range(5):
    print(f"x[{i}] = {x[i]:.4f}, y[{i}] = {y[i]:.4f}")

print("========= Ordinary Least Mean Square Method ===========")

# Calculate the means 
x_mean = np.mean(x)
y_mean = np.mean(y)

# Calculate the slope using the formula :
# m = sum((x - x_mean) * (y - y_mean)) / sum((x - x_mean) ** 2)

numerator = np.sum((x - x_mean)* (y - y_mean))
denominator = np.sum((x - x_mean) ** 2)
m = numerator / denominator 

# Calculate the y-intercept using the  formula :
# c = y_mean - m * x_mean 
c = y_mean - m*x_mean

print(f"Slope (m): {m:.4f}")
print(f"Intercept (c): {c:.4f}")

# Make predictions 
y_pred_ols = m*x+c 

# Calculate MSE 
mse_ols = np.mean((y - y_pred_ols)** 2)
print(f"Mean Squared Error: {mse_ols:.4f}")

print("\n ===== Graadient Descent Approach =====")

# Initialize parameters 
learning_rate = 0.01
num_iterations = 1000
m_gd = 0 # Initial slope
c_gd = 0 # Initial intercept 
n = len(x) # Number of data points 
costs = [] # To store cost history 

#Perform gradient descent 
for i in range(num_iterations):
    # Compute predictions 
    y_pred = m_gd * x + c_gd

    # Compute gradeints using formulas :
    # dj/dm = (2/n) * sum((y_pred -y) * x)
    # dj/dc = (2/n) * sum(y_pred -y )
    dj_dm = (2/n) * np.sum((y_pred - y) * x)
    dj_dc = (2/n) * np.sum(y_pred - y)

    # Update parameters using the update rules:
    # m = m - alpha * (dj/dm)
    # c = c - alpha * (dj/dc)
    m_gd = m_gd - learning_rate * dj_dm 
    c_gd = c_gd - learning_rate * dj_dc

    # Compute cost (MSE)
    cost = (1/n) * np.sum((y_pred -y)** 2)
    costs.append(cost)

    # Print progress occasionally 
    if i % 100 == 0:
        print(f"Iteration {i}: COst = {cost:4f}, m = {m_gd:.4f}, c = {c_gd:.4f}")

    # Final parameters and predictions 
    print(f"\n Final paramters after {num_iterations} iterations:")
    print(f"Slope (m): {m_gd:.4f}")
    print(f"Intercept (c): {c_gd:.4f}")

    y_pred_gd = m_gd * x + c_gd 
    mse_gd = np.mean((y - y_pred_gd) ** 2)
    print(f"Mean Squared Error: {mse_gd:.4f}")

# VISUALIZATION 
    plt.figure(figsize=(15,10))

    # Plot the data points 
    plt.subplot(2,2,1)
    plt.scatter(x, y, alpha=0.5, color='blue', label='Data Points')
    plt.plot(x, y_pred_ols, color='red', linewidth=2, label=f'OLS: y = {m:.4f}x + {c:.4f}')
    plt.plot(x, y_pred_gd, color='green', linewidth=2, label=f'GD: y = {m_gd:.4f}x + {c_gd:.4f}')
    plt.title('Linear Regression: OLS vs GD')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)

    # PLot the true function vs linear models 
    plt.subplot(2, 2, 2)
    x_smooth = np.linspace(0, 10, 1000)
    y_true = 2 * x_smooth**2 + 3 * x_smooth + 5 
    plt.plot(x_smooth, y_true, 'b-', linewidth=2, label='True function (quadratic)')
    plt.plot(x, y_pred_ols, 'r-', linewidth=2, label='OLS Linear Regression')
    plt.plot(x, y_pred_gd, 'g-',linewidth=2 , label='GD Linear Regression')
    plt.title('True Function vs Linear Models ')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)

    # Plot the cost history for GD 
    plt.subplot(2,2,3)
    plt.plot(costs)
    plt.title('Cost History during Gradient Descent')
    plt.xlabel('Iterations')
    plt.ylabel('Cost (MSE)')
    plt.grid(True)

    # Plot the residuals 
    plt.subplot(2,2,4)
    plt.scatter(x, y-y_pred_ols, color='red', alpha=0.5, label='OLS Residuals')
    plt.scatter(x, y - y_pred_gd, color='green',alpha=0.5, label='GD Residuals')
    plt.axhline(y=0, color='black', linestyle='-',alpha=0.3)
    plt.title('Residual Plot')
    plt.xlabel('x')
    plt.ylabel('Residuals')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()