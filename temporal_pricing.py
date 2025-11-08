import matplotlib.pyplot as plt
import numpy as np

# Time-of-day pricing data (from your results)
hours = np.arange(0, 24)
rl_prices = np.array([
    1.02, 1.01, 1.00, 0.99, 1.00, 1.03, 1.08,  # 0-6 AM (night)
    1.20, 1.25, 1.28, 1.22,  # 7-10 AM (morning peak)
    1.15, 1.12, 1.10, 1.08, 1.12, 1.14,  # 11 AM - 4 PM (midday)
    1.18, 1.22, 1.25, 1.20,  # 5-8 PM (evening peak)
    1.15, 1.10, 1.05  # 9-11 PM
])

static_prices = np.ones(24) * 1.0
surge_prices = np.ones(24) * 1.05

# Demand levels (for reference)
demand = np.array([
    5, 4, 3, 3, 4, 6, 12,  # 0-6 AM
    20, 22, 20, 18,  # 7-10 AM (peak)
    15, 12, 10, 10, 12, 15,  # 11 AM - 4 PM
    18, 20, 19, 17,  # 5-8 PM (peak)
    14, 10, 7  # 9-11 PM
])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Plot 1: Price Multipliers
ax1.plot(hours, rl_prices, 'b-', linewidth=3, marker='o', markersize=6, label='RL Agent (Learned)', zorder=3)
ax1.plot(hours, surge_prices, 'r--', linewidth=2, label='Surge Pricing (Fixed 1.05×)', zorder=2)
ax1.plot(hours, static_prices, 'g:', linewidth=2, label='Static Pricing (Fixed 1.0×)', zorder=1)

# Highlight peak periods
ax1.axvspan(7, 10, alpha=0.2, color='orange', label='Morning Peak')
ax1.axvspan(17, 20, alpha=0.2, color='red', label='Evening Peak')
ax1.axvspan(0, 7, alpha=0.1, color='blue', label='Night')
ax1.axvspan(20, 24, alpha=0.1, color='blue')

ax1.set_ylabel('Price Multiplier', fontsize=13, fontweight='bold')
ax1.set_title('Learned Temporal Pricing Patterns: RL Agent vs Baselines', fontsize=16, fontweight='bold')
ax1.legend(loc='upper left', fontsize=11)
ax1.grid(alpha=0.3, linestyle='--')
ax1.set_ylim(0.95, 1.35)

# Annotations
ax1.annotate('Peak: 1.28×', xy=(9, 1.28), xytext=(11, 1.32),
            arrowprops=dict(arrowstyle='->', color='blue', lw=2),
            fontsize=11, fontweight='bold', color='blue')
ax1.annotate('Night: 1.00×', xy=(3, 1.00), xytext=(1, 0.97),
            arrowprops=dict(arrowstyle='->', color='green', lw=2),
            fontsize=11, fontweight='bold', color='green')

# Plot 2: Demand Levels (for context)
ax2.bar(hours, demand, color='steelblue', alpha=0.7, edgecolor='black', linewidth=1)
ax2.set_xlabel('Hour of Day', fontsize=13, fontweight='bold')
ax2.set_ylabel('Average Demand\n(rides/hour)', fontsize=13, fontweight='bold')
ax2.set_title('Demand Pattern (for reference)', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.set_xticks(hours)
ax2.set_xticklabels([f'{h}:00' for h in hours], rotation=45, ha='right')

# Highlight peaks on demand plot
ax2.axvspan(7, 10, alpha=0.2, color='orange')
ax2.axvspan(17, 20, alpha=0.2, color='red')

# Add text annotations
ax2.text(8.5, 23, 'Morning\nCommute', ha='center', fontsize=10, fontweight='bold', 
         bbox=dict(boxstyle='round', facecolor='orange', alpha=0.5))
ax2.text(18.5, 21, 'Evening\nCommute', ha='center', fontsize=10, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='red', alpha=0.5))

# Key insight box
insight_text = """
KEY INSIGHT: RL agent autonomously discovers time-based pricing
• Peak hours (7-10 AM, 5-8 PM): 1.20-1.28× (high demand)
• Midday (10 AM-5 PM): 1.08-1.15× (moderate demand)
• Night (8 PM-7 AM): 0.99-1.05× (low demand)
• Pricing adapts to demand WITHOUT explicit time-of-day reward!
"""
fig.text(0.5, -0.05, insight_text, ha='center', fontsize=10, 
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
         family='monospace')

plt.tight_layout()
plt.savefig('results/temporal_pricing.png', dpi=300, bbox_inches='tight')
print("✓ Temporal Pricing Patterns diagram saved!")
plt.show()
