def plot_cost(J_history):
    print("plot")

ss_res = np.sum((Y_test - y_pred) ** 2)
ss_tot = np.sum((Y_test - np.mean(Y_test)) ** 2)

r2 = 1 - ss_res / ss_tot

print(theta)
print("R²:", r2)