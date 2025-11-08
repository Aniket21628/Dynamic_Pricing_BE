import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Actor Network
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 8)
ax1.axis('off')
ax1.set_title('Actor Network Architecture', fontsize=16, fontweight='bold', pad=20)

# Layer positions
layers_actor = [
    ('Input\nState\n(49)', 1, 4, 0.6, 2, '#e8f4f8'),
    ('Linear\n512', 2, 4, 0.6, 2, '#3498db'),
    ('BatchNorm\n512', 3, 4, 0.4, 2, '#9b59b6'),
    ('ReLU', 3.6, 4, 0.3, 2, '#f39c12'),
    ('Dropout\n0.1', 4.1, 4, 0.4, 2, '#e74c3c'),
    ('Linear\n384', 4.7, 4, 0.6, 2, '#3498db'),
    ('BatchNorm\n384', 5.5, 4, 0.4, 2, '#9b59b6'),
    ('ReLU', 6.1, 4, 0.3, 2, '#f39c12'),
    ('Dropout\n0.1', 6.6, 4, 0.4, 2, '#e74c3c'),
    ('Linear\n256', 7.2, 4, 0.6, 2, '#3498db'),
    ('ReLU', 8, 4, 0.3, 2, '#f39c12'),
    ('Linear\n16', 8.5, 4, 0.5, 2, '#3498db'),
    ('Tanh', 9.2, 4, 0.3, 2, '#f39c12'),
]

# Draw actor layers
for i, (label, x, y, w, h, color) in enumerate(layers_actor):
    rect = mpatches.FancyBboxPatch((x, y-h/2), w, h, boxstyle="round,pad=0.05",
                                   edgecolor='black', facecolor=color, linewidth=1.5)
    ax1.add_patch(rect)
    ax1.text(x + w/2, y, label, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Arrows
    if i < len(layers_actor) - 1:
        next_x = layers_actor[i+1][1]
        ax1.annotate('', xy=(next_x, y), xytext=(x + w, y),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

# Output transformation
ax1.text(9.7, 5.5, 'Action Transform:\na = 1.125 + 0.15×tanh(x)', 
         ha='left', fontsize=9, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))
ax1.text(9.7, 2.5, 'Output:\n16 Price\nMultipliers\n[0.90, 1.35]', 
         ha='left', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6))

# Critic Network
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 8)
ax2.axis('off')
ax2.set_title('Twin Critic Networks Architecture (Q₁ and Q₂ - Identical)', fontsize=16, fontweight='bold', pad=20)

# Layer positions for critic
layers_critic = [
    ('Input\nState+Action\n(49+16=65)', 1, 4, 0.8, 2.5, '#e8f4f8'),
    ('Linear\n512', 2.2, 4, 0.6, 2, '#e74c3c'),
    ('ReLU', 3, 4, 0.3, 2, '#f39c12'),
    ('Linear\n384', 3.5, 4, 0.6, 2, '#e74c3c'),
    ('ReLU', 4.3, 4, 0.3, 2, '#f39c12'),
    ('Linear\n256', 4.8, 4, 0.6, 2, '#e74c3c'),
    ('ReLU', 5.6, 4, 0.3, 2, '#f39c12'),
    ('Linear\n1', 6.1, 4, 0.5, 2, '#e74c3c'),
]

# Draw critic layers
for i, (label, x, y, w, h, color) in enumerate(layers_critic):
    rect = mpatches.FancyBboxPatch((x, y-h/2), w, h, boxstyle="round,pad=0.05",
                                   edgecolor='black', facecolor=color, linewidth=1.5)
    ax2.add_patch(rect)
    ax2.text(x + w/2, y, label, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Arrows
    if i < len(layers_critic) - 1:
        next_x = layers_critic[i+1][1]
        ax2.annotate('', xy=(next_x, y), xytext=(x + w, y),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

# Output annotation
ax2.text(7, 4, 'Output:\nQ-value\n(scalar)', ha='left', fontsize=10, 
         bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.6))

# Twin critics note
ax2.text(5, 1, 'Note: Two identical critic networks (Q₁ and Q₂) with independent parameters\nTarget computation uses: y = r + γ × min(Q₁\'(s\',a\'), Q₂\'(s\',a\'))', 
         ha='center', fontsize=9, style='italic',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

# Legend
legend_elements = [
    mpatches.Patch(facecolor='#3498db', edgecolor='black', label='Linear Layer'),
    mpatches.Patch(facecolor='#9b59b6', edgecolor='black', label='BatchNorm'),
    mpatches.Patch(facecolor='#f39c12', edgecolor='black', label='Activation (ReLU/Tanh)'),
    mpatches.Patch(facecolor='#e74c3c', edgecolor='black', label='Dropout/Critic Linear'),
]
ax1.legend(handles=legend_elements, loc='upper left', fontsize=9)

plt.tight_layout()
plt.savefig('results/network_architecture.png', dpi=300, bbox_inches='tight')
print("✓ Network Architecture diagram saved!")
plt.show()
