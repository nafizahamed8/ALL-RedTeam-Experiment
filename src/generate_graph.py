import matplotlib.pyplot as plt

# 1. Hardcoded Accuracy numbers extracted directly from your Colab screenshot
epochs = [1, 2, 3, 4, 5]
train_acc = [85.88, 89.65, 92.42, 94.95, 96.82]
val_acc = [87.76, 89.36, 89.69, 89.78, 90.86]

# 2. Generate the IEEE-formatted Accuracy plot
plt.figure(figsize=(8, 6))
plt.plot(epochs, train_acc, 'bo-', label='Training Accuracy', linewidth=2)
plt.plot(epochs, val_acc, 'ro-', label='Validation Accuracy', linewidth=2)

plt.title('Fine-Tuning Optimization: Training vs. Validation Accuracy', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.xticks(epochs)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# 3. Save directly to your paper folder
save_path = './paper/accuracy_graph.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Graph successfully saved to {save_path}")

plt.show()