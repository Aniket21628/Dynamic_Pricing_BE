import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Colors
actor_color = '#3498db'
critic_color = '#e74c3c'
buffer_color = '#95a5a6'
target_color = '#f39c12'

# Replay Buffer
buffer = FancyBboxPatch((0.5, 4), 1.5, 1.5, boxstyle="round,pad=0.1", 
                        edgecolor='black', facecolor=buffer_color, linewidth=2)
ax.add_patch(buffer)
ax.text(1.25, 4.75, 'Replay\nBuffer', ha='center', va='center', fontsize=11, fontweight='bold')

# Actor Network
actor = FancyBboxPatch((3, 7), 1.8, 1.2, boxstyle="round,pad=0.1",
                       edgecolor='black', facecolor=actor_color, linewidth=2)
ax.add_patch(actor)
ax.text(3.9, 7.6, 'Actor π(s)', ha='center', va='center', fontsize=12, fontweight='bold', color='white')

# Twin Critics
critic1 = FancyBboxPatch((3, 5), 1.8, 0.9, boxstyle="round,pad=0.1",
                        edgecolor='black', facecolor=critic_color, linewidth=2)
ax.add_patch(critic1)
ax.text(3.9, 5.45, 'Critic Q₁(s,a)', ha='center', va='center', fontsize=11, fontweight='bold', color='white')

critic2 = FancyBboxPatch((3, 3.5), 1.8, 0.9, boxstyle="round,pad=0.1",
                        edgecolor='black', facecolor=critic_color, linewidth=2)
ax.add_patch(critic2)
ax.text(3.9, 3.95, 'Critic Q₂(s,a)', ha='center', va='center', fontsize=11, fontweight='bold', color='white')

# Target Networks
target_actor = FancyBboxPatch((6.5, 7), 1.8, 1.2, boxstyle="round,pad=0.1",
                             edgecolor='black', facecolor=target_color, linewidth=2, linestyle='--')
ax.add_patch(target_actor)
ax.text(7.4, 7.6, "Target π'(s')", ha='center', va='center', fontsize=11, fontweight='bold')

target_critic1 = FancyBboxPatch((6.5, 5), 1.8, 0.9, boxstyle="round,pad=0.1",
                               edgecolor='black', facecolor=target_color, linewidth=2, linestyle='--')
ax.add_patch(target_critic1)
ax.text(7.4, 5.45, "Target Q₁'", ha='center', va='center', fontsize=10, fontweight='bold')

target_critic2 = FancyBboxPatch((6.5, 3.5), 1.8, 0.9, boxstyle="round,pad=0.1",
                               edgecolor='black', facecolor=target_color, linewidth=2, linestyle='--')
ax.add_patch(target_critic2)
ax.text(7.4, 3.95, "Target Q₂'", ha='center', va='center', fontsize=10, fontweight='bold')

# Environment
env = FancyBboxPatch((3, 1), 1.8, 1, boxstyle="round,pad=0.1",
                     edgecolor='black', facecolor='#2ecc71', linewidth=2)
ax.add_patch(env)
ax.text(3.9, 1.5, 'Environment', ha='center', va='center', fontsize=11, fontweight='bold', color='white')

# Arrows
# Buffer to Networks
ax.annotate('', xy=(3, 7.6), xytext=(2, 4.75), 
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))
ax.annotate('', xy=(3, 5.45), xytext=(2, 4.75),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))
ax.annotate('', xy=(3, 3.95), xytext=(2, 4.75),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))

# Actor to Environment
ax.annotate('', xy=(3.9, 2), xytext=(3.9, 7),
            arrowprops=dict(arrowstyle='->', lw=2.5, color=actor_color))
ax.text(3.3, 4.5, 'Action', ha='center', fontsize=9, rotation=90, color=actor_color, fontweight='bold')

# Environment to Buffer
ax.annotate('', xy=(2, 4.75), xytext=(3, 1.5),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))
ax.text(2.3, 3, 'Store\n(s,a,r,s\')', ha='center', fontsize=8)

# Soft Updates (tau)
ax.annotate('', xy=(6.5, 7.6), xytext=(4.8, 7.6),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='purple', linestyle='--'))
ax.text(5.65, 7.9, 'τ=0.005', ha='center', fontsize=8, color='purple')

ax.annotate('', xy=(6.5, 5.45), xytext=(4.8, 5.45),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='purple', linestyle='--'))

ax.annotate('', xy=(6.5, 3.95), xytext=(4.8, 3.95),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='purple', linestyle='--'))

# Min operation
ax.text(7.4, 2.8, 'min(Q₁\', Q₂\')', ha='center', fontsize=10, 
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# Labels
ax.text(5, 9.3, 'TD3 Algorithm Architecture', ha='center', fontsize=16, fontweight='bold')
ax.text(0.5, 0.3, 'Policy Networks (solid)', fontsize=9, color=actor_color, fontweight='bold')
ax.text(0.5, 0.1, 'Target Networks (dashed)', fontsize=9, color=target_color, fontweight='bold')

# Annotations
ax.text(8.5, 8.5, 'Delayed Update\n(every 2 steps)', ha='left', fontsize=8,
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
ax.text(8.5, 2.5, 'Clipped Double\nQ-Learning', ha='left', fontsize=8,
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

plt.tight_layout()
plt.savefig('results/td3_architecture.png', dpi=300, bbox_inches='tight')
print("✓ TD3 Architecture diagram saved!")
plt.show()
