import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch

fig, ax = plt.subplots(figsize=(12, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(5, 9.5, 'Ride-Hailing Environment Interaction Loop', ha='center', fontsize=18, fontweight='bold')

# Environment Box (Center)
env_box = FancyBboxPatch((3, 4.5), 4, 2.5, boxstyle="round,pad=0.15",
                         edgecolor='black', facecolor='#2ecc71', linewidth=3)
ax.add_patch(env_box)
ax.text(5, 6.5, 'ENVIRONMENT', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
ax.text(5, 6, '16 Zones | 100 Vehicles', ha='center', va='center', fontsize=10, color='white')
ax.text(5, 5.5, 'Demand Generation', ha='center', va='center', fontsize=9, color='white', style='italic')
ax.text(5, 5.1, 'Vehicle Matching', ha='center', va='center', fontsize=9, color='white', style='italic')

# Agent Box (Left)
agent_box = FancyBboxPatch((0.5, 5.5), 1.8, 1.5, boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor='#3498db', linewidth=2.5)
ax.add_patch(agent_box)
ax.text(1.4, 6.5, 'TD3 AGENT', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
ax.text(1.4, 6, 'Actor-Critic', ha='center', va='center', fontsize=9, color='white')

# State Box (Top)
state_box = FancyBboxPatch((3.5, 7.8), 3, 1, boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor='#f39c12', linewidth=2)
ax.add_patch(state_box)
ax.text(5, 8.5, 'STATE (49-dim)', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(5, 8.1, 'Vehicles(16) + Demand(16) + Prices(16) + Time(1)', ha='center', va='center', fontsize=8)

# Action Box (Bottom)
action_box = FancyBboxPatch((3.5, 2.5), 3, 1, boxstyle="round,pad=0.1",
                           edgecolor='black', facecolor='#9b59b6', linewidth=2)
ax.add_patch(action_box)
ax.text(5, 3.2, 'ACTION (16-dim)', ha='center', va='center', fontsize=11, fontweight='bold', color='white')
ax.text(5, 2.8, 'Price Multipliers: [0.90, 1.35] per zone', ha='center', va='center', fontsize=8, color='white')

# Reward Box (Right)
reward_box = FancyBboxPatch((7.7, 5.5), 1.8, 1.5, boxstyle="round,pad=0.1",
                           edgecolor='black', facecolor='#e74c3c', linewidth=2.5)
ax.add_patch(reward_box)
ax.text(8.6, 6.5, 'REWARD', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
ax.text(8.6, 6.1, 'Revenue', ha='center', va='center', fontsize=9, color='white')
ax.text(8.6, 5.8, '+ 3×Rides', ha='center', va='center', fontsize=9, color='white')

# Arrows with labels
# State to Agent
arrow1 = FancyArrowPatch((3.5, 8.3), (2.3, 7),
                        arrowstyle='->', mutation_scale=30, lw=3, color='#f39c12')
ax.add_patch(arrow1)
ax.text(2.7, 7.8, 'Observe', ha='center', fontsize=10, fontweight='bold', color='#f39c12')

# Agent to Action
arrow2 = FancyArrowPatch((1.4, 5.5), (4.5, 3.5),
                        arrowstyle='->', mutation_scale=30, lw=3, color='#9b59b6')
ax.add_patch(arrow2)
ax.text(2.5, 4.3, 'Decide', ha='center', fontsize=10, fontweight='bold', color='#9b59b6')

# Action to Environment
arrow3 = FancyArrowPatch((5, 3.5), (5, 4.5),
                        arrowstyle='->', mutation_scale=30, lw=3, color='#9b59b6')
ax.add_patch(arrow3)
ax.text(5.5, 4, 'Apply\nPrices', ha='center', fontsize=9, fontweight='bold', color='#9b59b6')

# Environment to Reward
arrow4 = FancyArrowPatch((7, 6), (7.7, 6),
                        arrowstyle='->', mutation_scale=30, lw=3, color='#e74c3c')
ax.add_patch(arrow4)
ax.text(7.35, 6.4, 'Compute', ha='center', fontsize=9, fontweight='bold', color='#e74c3c')

# Reward to Agent (feedback loop)
arrow5 = FancyArrowPatch((8.6, 5.5), (2.3, 6.2),
                        arrowstyle='->', mutation_scale=30, lw=3, color='#e74c3c',
                        connectionstyle="arc3,rad=0.3")
ax.add_patch(arrow5)
ax.text(6, 4, 'Learn', ha='center', fontsize=10, fontweight='bold', color='#e74c3c')

# Environment to State (next state)
arrow6 = FancyArrowPatch((5, 7), (5, 7.8),
                        arrowstyle='->', mutation_scale=30, lw=3, color='#f39c12')
ax.add_patch(arrow6)
ax.text(4.3, 7.4, 'Update\nState', ha='center', fontsize=9, fontweight='bold', color='#f39c12')

# Timestep indicator
timestep = Circle((5, 1.5), 0.4, facecolor='lightgray', edgecolor='black', linewidth=2)
ax.add_patch(timestep)
ax.text(5, 1.5, 't', ha='center', va='center', fontsize=14, fontweight='bold')
ax.text(5, 0.8, 'Repeat every timestep (1,440 steps/episode)', ha='center', fontsize=9, style='italic')

# Details boxes
details1 = ax.text(0.3, 3.5, 'Demand Elasticity:\nd = base × exp(-0.4×(p-1))', 
                  fontsize=8, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

details2 = ax.text(0.3, 2.3, 'Reward Formula:\nR = revenue + 3×rides\n    - 2×unfulfilled\n    - 1.5×empty', 
                  fontsize=8, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

details3 = ax.text(7.2, 1.8, 'Training:\n50 episodes\n72,000 steps\n~45 minutes', 
                  fontsize=8, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

plt.tight_layout()
plt.savefig('results/environment_flow.png', dpi=300, bbox_inches='tight')
print("✓ Environment Flow diagram saved!")
plt.show()
