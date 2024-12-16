import matplotlib.pyplot as plt

# Data for plotting
x = [0.6899, 0.6996, 0.6368, 0.6302, 0.6743, 0.6219, 0.6996, 0.6774, 0.5708, 0.5343]
y = [0.3678, 0.3534, 0.4265, 0.4406, 0.4092, 0.4451, 0.3534, 0.3866, 0.4906, 0.4837]

# Create the plot
plt.figure(figsize=(8, 5))  # Set the figure size
plt.scatter(x, y, color='blue', label='Points')

# Add labels and title
plt.title('Adversarial vs Standard Acc')
plt.xlabel('Standard Acc')
plt.ylabel('Adv Acc')

# Add a grid
plt.grid(True, linestyle='--', alpha=0.7)

# Add a legend
plt.legend()

# Show the plot
plt.show()
