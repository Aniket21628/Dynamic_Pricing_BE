import matplotlib.pyplot as plt
import numpy as np

# Ablation study data
configurations = [
    'Original\n(Failed)',
    'Original\n+ Multi-Obj\nReward',
    '+ Balanced\nWeights\n(3× rides)',
    '+ Action\nConstraint\n[0.90, 1.35]',
    'Static\nBaseline',
    'Surge\nBaseline'
]

revenues = [11.25, 11.63, 11.95, 12.43, 11.69, 12.12]  # in millions
rides = [85, 168, 198, 213, 222, 219]  # in thousands
colors = ['#e74c3c', '#e67e22', '#f39c12', '#2ecc71', '#95a5a6', '#95a5a6']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Plot 1: Revenue Comparison
bars1 = ax1.bar(range(len(configurations)), revenues, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
ax1.set_ylabel('Total Revenue (Million $)', fontsize=13, fontweight='bold')
ax1.set_title('Ablation Study: Revenue Impact of Design Choices', fontsize=16, fontweight='bold')
ax1.set_xticks(range(len(configurations)))
ax1.set_xticklabels(configurations, fontsize=10)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_ylim(10.5, 13)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars1, revenues)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
            f'${val:.2f}M', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Show improvement percentage
    if i > 0 and i < 4:
        improvement = ((revenues[i] - revenues[0]) / revenues[0]) * 100
        ax1.text(bar.get_x() + bar.get_width()/2., height - 0.3,
                f'+{improvement:.1f}%', ha='center', va='top', fontsize=9, 
                color='white', fontweight='bold')

# Highlight final optimized version
bars1[3].set_linewidth(4)
bars1[3].set_edgecolor('darkgreen')

# Plot 2: Rides Comparison
bars2 = ax2.bar(range(len(configurations)), rides, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
ax2.set_ylabel('Completed Rides (Thousands)', fontsize=13, fontweight='bold')
ax2.set_title('Ablation Study: Ride Volume Impact', fontsize=16, fontweight='bold')
ax2.set_xticks(range(len(configurations)))
ax2.set_xticklabels(configurations, fontsize=10)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.set_ylim(50, 240)

# Add value labels
for i, (bar, val) in enumerate(zip(bars2, rides)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 3,
            f'{val}k', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Highlight final optimized version
bars2[3].set_linewidth(4)
bars2[3].set_edgecolor('darkgreen')

# Add annotations
ax1.annotate('Failed: Extreme pricing\nkilled demand', xy=(0, 11.25), xytext=(0.5, 10.8),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=10, color='red', fontweight='bold')

ax1.annotate('Final Optimized:\n+10.4% vs Original\n+6.35% vs Static', xy=(3, 12.43), xytext=(3.5, 12.8),
            arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2),
            fontsize=10, color='darkgreen', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

ax2.annotate('Original: Only 85k rides\n(extreme pricing)', xy=(0, 85), xytext=(0.5, 120),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=10, color='red', fontweight='bold')

ax2.annotate('Optimized: 213k rides\n(96% of baseline)', xy=(3, 213), xytext=(1.5, 180),
            arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2),
            fontsize=10, color='darkgreen', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

# Summary box
summary_text = """
ABLATION STUDY INSIGHTS:
1. Multi-Objective Reward: +3.4% revenue, +98% rides (prevents extreme pricing)
2. Balanced Weights (3× rides): +2.8% revenue, +18% rides (better trade-off)
3. Action Constraint [0.90, 1.35]: +4.0% revenue, +7.6% rides (efficient exploration)
TOTAL IMPROVEMENT: +10.4% revenue vs original, +150% rides
"""

fig.text(0.5, -0.08, summary_text, ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
         family='monospace', fontweight='bold')

plt.tight_layout()
plt.savefig('results/ablation_study.png', dpi=300, bbox_inches='tight')
print("✓ Ablation Study diagram saved!")
plt.show()
