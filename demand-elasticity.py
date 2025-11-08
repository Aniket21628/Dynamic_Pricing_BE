import matplotlib.pyplot as plt
import numpy as np

prices = np.linspace(0.8, 1.6, 100)
demand_exp = np.exp(-0.4 * (prices - 1.0)) * 100
demand_linear = (2.0 - prices) * 100

plt.figure(figsize=(8, 5))
plt.plot(prices, demand_exp, 'b-', linewidth=2, label='Exponential (Our Model)')
plt.plot(prices, demand_linear, 'r--', linewidth=2, label='Linear (Naive)')
plt.axhline(100, color='gray', linestyle=':', alpha=0.5)
plt.axvline(1.0, color='gray', linestyle=':', alpha=0.5)
plt.scatter([1.0, 1.2, 1.5], [100, 92, 70], color='blue', s=100, zorder=5)
plt.xlabel('Price Multiplier', fontsize=12)
plt.ylabel('Demand (%)', fontsize=12)
plt.title('Price Elasticity of Demand', fontsize=14)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('results/demand_elasticity.png', dpi=300)
plt.show()