import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(8, 8))

# Create 4x4 grid
for i in range(16):
    row = i // 4
    col = i % 4
    x, y = col * 100, row * 100
    
    # Color by demand (example values)
    demand_levels = [18, 20, 15, 12, 22, 20, 18, 14, 16, 18, 15, 13, 10, 12, 11, 9]
    color = plt.cm.YlOrRd(demand_levels[i] / 25)
    
    rect = plt.Rectangle((x, y), 100, 100, facecolor=color, edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(x+50, y+50, f'Zone {i}\n{demand_levels[i]} rides/hr', 
            ha='center', va='center', fontsize=10, fontweight='bold')

ax.set_xlim(0, 400)
ax.set_ylim(0, 400)
ax.set_xlabel('X Coordinate (meters)', fontsize=12)
ax.set_ylabel('Y Coordinate (meters)', fontsize=12)
ax.set_title('16-Zone Grid Configuration with Average Demand', fontsize=14)
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('results/zone_layout.png', dpi=300)
plt.show()