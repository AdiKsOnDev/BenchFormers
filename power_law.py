import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

dataset_sizes = [1000, 2500, 5000]

scores = [[0.9506, 0.6407, 0.6365], 
          [0.9822, 0.9689, 0.9556],
          [0.9992, 0.9978, 0.6450],
          [0.9730, 0.6342, 0.9835],
          [0.9951, 0.9978, 0.9992]]

model_names = ["Roformer", "Longformer", "BigBird", "LegalBERT", "ModernBERT"]
colors = ["blue", "green", "orange", "purple", "red"]

def power_law(x, a, b):
    return a * np.power(x, b)

plt.figure(figsize=(10, 6))
x_fit = np.linspace(900, 5100, 200)

for i, (y_data, name, color) in enumerate(zip(scores, model_names, colors)):
    y_data = np.array(y_data)
    
    # Fit power law
    params, _ = curve_fit(power_law, dataset_sizes, y_data, maxfev=10000)
    y_fit = power_law(x_fit, *params)
    
    # plt.scatter(dataset_sizes, y_data, label=f"{name} (data)", color=color)
    
    plt.plot(x_fit, y_fit, label=f"{name}", linestyle="-", color=color)

plt.title("Power Law Fit of Macro F1-Score vs Dataset Size")
plt.xlabel("Dataset Size")
plt.ylabel("Macro F1-Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("power_law.png")
plt.show()
