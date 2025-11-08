import matplotlib.pyplot as plt
import numpy as np

components = ['Revenue\n(1.0×)', 'Rides\n(3.0×)', 'Fulfillment\nBonus', 
              'Unfulfilled\nPenalty', 'Empty Trips\nPenalty']
values = [12.43, 3*213.03, 0.857*12.43*0.5, -2*8.97, -1.5*40.65]
colors = ['green', 'blue', 'lightgreen', 'red', 'orange']

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(components, values, color=colors, alpha=0.7, edgecolor='black')
ax.axhline(0, color='black', linewidth=0.8)
ax.set_ylabel('Contribution to Total Reward', fontsize=12)
ax.set_title('Multi-Objective Reward Function Components', fontsize=14)
ax.grid(axis='y', alpha=0.3)

for bar, val in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1f}', ha='center', va='bottom' if val > 0 else 'top', fontsize=10)

plt.tight_layout()
plt.savefig('results/reward_components.png', dpi=300)
plt.show()