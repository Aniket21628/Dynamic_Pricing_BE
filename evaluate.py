# evaluate.py — Enhanced for realistic ride counts & dual RL variants
import numpy as np
import matplotlib.pyplot as plt
from environment import RideHailingEnv
from td3_agent import TD3Agent
import pandas as pd
import os


# ==============================================================
# Pricing Strategies
# ==============================================================
class PricingStrategies:
    """Different pricing strategies to compare"""

    @staticmethod
    def static_pricing(state):
        """Static 1.0 multiplier (baseline)"""
        return np.ones(16)

    @staticmethod
    def surge_pricing(state):
        """Simple surge pricing based on demand"""
        demand_start = 16
        demand_per_zone = state[demand_start:demand_start + 16]
        max_demand = np.max(demand_per_zone) + 1e-6
        multiplier = 1.0 + 0.5 * (demand_per_zone / max_demand)
        return np.clip(multiplier, 0.33, 3.0)

    @staticmethod
    def rl_pricing(state, agent, noisy_eval=False):
        """RL-based pricing (learned)"""
        noise_level = 0.1 if noisy_eval else 0.0
        action = agent.select_action(state, noise=noise_level)
        return np.clip(1.0 + action, 0.33, 3.0)


# ==============================================================
# Evaluation logic
# ==============================================================
def evaluate_strategy(strategy_name, strategy_func, episodes=10, agent=None):
    """Evaluate one pricing strategy"""
    env = RideHailingEnv(num_zones=16, num_vehicles=100)
    all_metrics, total_rewards = [], []

    print(f"\nEvaluating {strategy_name}...")
    print("-" * 55)

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0.0
        for step in range(1440):
            if "RL Agent" in strategy_name:
                action = strategy_func(state, agent)
            else:
                action = strategy_func(state)

            next_state, step_revenue, done, info, *_ = env.step(action)
            episode_reward += step_revenue
            state = next_state
            if done:
                break

        metrics = env.get_metrics()
        all_metrics.append(metrics)
        total_rewards.append(episode_reward)

        print(f"Episode {episode + 1}: "
              f"Revenue=${metrics['total_revenue'] / 1e6:.2f}M, "
              f"Rides={metrics['rides_completed']}, Empty={metrics['empty_trips']}")

    avg_revenue = np.mean([m['total_revenue'] for m in all_metrics])
    avg_rides = np.mean([m['rides_completed'] for m in all_metrics])
    avg_empty = np.mean([m['empty_trips'] for m in all_metrics])
    avg_reward = np.mean(total_rewards)

    result = {
        'strategy': strategy_name,
        'avg_revenue': avg_revenue,
        'avg_rides': avg_rides,
        'avg_empty': avg_empty,
        'avg_reward': avg_reward,
        'std_reward': np.std(total_rewards),
        'all_rewards': total_rewards,
        'all_metrics': all_metrics
    }

    print(f"\n{strategy_name} Results:")
    print(f"  Average Revenue: ${avg_revenue / 1e6:.2f}M")
    print(f"  Average Rides: {avg_rides:.1f}")
    print(f"  Average Empty Trips: {avg_empty:.1f}")
    print(f"  Average Reward: {avg_reward / 1e6:.2f}M ± {np.std(total_rewards) / 1e6:.2f}M")

    return result


# ==============================================================
# Comparison and Plotting
# ==============================================================
def compare_strategies(rl_agent_path='models/td3_pricing_agent.pth', episodes=10):
    """Compare static, surge, and RL strategies"""

    print("\n" + "=" * 70)
    print("DYNAMIC PRICING STRATEGY COMPARISON (SCALED ENVIRONMENT)")
    print("=" * 70)

    # Load RL agent
    print("Loading trained RL agent...")
    agent = TD3Agent(state_dim=49, action_dim=16, max_action=1.5)
    try:
        agent.load(rl_agent_path)
        print("✓ Agent loaded successfully\n")
    except FileNotFoundError:
        print("⚠ Agent not found, using untrained agent\n")

    results = {}

    # Static
    results['Static'] = evaluate_strategy(
        'Static Pricing (1.0x)',
        PricingStrategies.static_pricing,
        episodes=episodes
    )

    # Surge
    results['Surge'] = evaluate_strategy(
        'Demand-Based Surge Pricing',
        PricingStrategies.surge_pricing,
        episodes=episodes
    )

    # RL (Greedy)
    results['RL_Greedy'] = evaluate_strategy(
        'RL Agent (Greedy)',
        lambda state, agent: PricingStrategies.rl_pricing(state, agent, noisy_eval=False),
        episodes=episodes,
        agent=agent
    )

    # RL (Noisy Evaluation)
    results['RL_Noisy'] = evaluate_strategy(
        'RL Agent (Noisy Evaluation)',
        lambda state, agent: PricingStrategies.rl_pricing(state, agent, noisy_eval=True),
        episodes=episodes,
        agent=agent
    )

    # Build summary table
    df = pd.DataFrame({
        'Strategy': [r['strategy'] for r in results.values()],
        'Avg Revenue ($M)': [r['avg_revenue'] / 1e6 for r in results.values()],
        'Avg Rides': [r['avg_rides'] for r in results.values()],
        'Avg Empty Trips': [r['avg_empty'] for r in results.values()],
        'Avg Reward ($M)': [r['avg_reward'] / 1e6 for r in results.values()]
    })

    static_rev = results['Static']['avg_revenue']
    df['Improvement over Static (%)'] = (
        (df['Avg Revenue ($M)'] / (static_rev / 1e6) - 1) * 100
    ).round(2)

    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY (REVENUE IN MILLIONS)")
    print("=" * 70)
    print(df.to_string(index=False))

    plot_comparison(results)
    return results


def plot_comparison(results):
    """Plot strategy comparison"""
    strategies = [r['strategy'] for r in results.values()]
    revenues = [r['avg_revenue'] / 1e6 for r in results.values()]
    rides = [r['avg_rides'] for r in results.values()]
    empty = [r['avg_empty'] for r in results.values()]
    reward_data = [r['all_rewards'] for r in results.values()]

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#F7B801']
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Revenue
    axes[0, 0].bar(strategies, revenues, color=colors, alpha=0.85, edgecolor='black')
    axes[0, 0].set_title('Average Revenue per Episode ($ Millions)', fontsize=12, fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(revenues):
        axes[0, 0].text(i, v + 0.05, f"{v:.2f}M", ha='center', fontweight='bold')

    # Rides
    axes[0, 1].bar(strategies, rides, color=colors, alpha=0.85, edgecolor='black')
    axes[0, 1].set_title('Average Rides Completed', fontsize=12, fontweight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3)

    # Empty trips
    axes[1, 0].bar(strategies, empty, color=colors, alpha=0.85, edgecolor='black')
    axes[1, 0].set_title('Average Empty Trips', fontsize=12, fontweight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3)

    # Reward distribution
    bp = axes[1, 1].boxplot([[r / 1e6 for r in rewards] for rewards in reward_data],
                            labels=strategies, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    axes[1, 1].set_title('Reward Distribution ($ Millions)', fontsize=12, fontweight='bold')
    axes[1, 1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig('results/strategy_comparison_scaled.png', dpi=300, bbox_inches='tight')
    print("\n✓ Scaled comparison plot saved to results/strategy_comparison_scaled.png")
    plt.close()


# ==============================================================
# Policy Analysis
# ==============================================================
def analyze_pricing_policy(agent, episodes=3):
    """Analyze and visualize the learned pricing policy"""
    print("\n" + "="*60)
    print("LEARNED PRICING POLICY ANALYSIS")
    print("="*60)

    env = RideHailingEnv(num_zones=16, num_vehicles=100)
    policy_data = []

    for episode in range(episodes):
        state = env.reset()
        for step in range(1440):
            action = agent.select_action(state, noise=0.0)
            demand_start = 16
            demand_per_zone = state[demand_start:demand_start+16]
            idle_per_zone = state[:16]

            for zone in range(16):
                policy_data.append({
                    'zone': zone,
                    'price_multiplier': action[zone],
                    'demand': demand_per_zone[zone],
                    'idle_vehicles': idle_per_zone[zone],
                    'hour': (step % 1440) / 60
                })

            next_state, reward, done, _ = env.step(action)
            state = next_state
            if done:
                break

    df_policy = pd.DataFrame(policy_data)
    correlation = df_policy['price_multiplier'].corr(df_policy['demand'])
    print(f"\nPrice vs Demand Correlation: {correlation:.4f}")

    hourly_avg = df_policy.groupby('hour')['price_multiplier'].mean()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].plot(hourly_avg.index, hourly_avg.values, marker='o', linewidth=2)
    axes[0].set_title('Learned Hourly Pricing Pattern', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Hour of Day')
    axes[0].set_ylabel('Average Price Multiplier')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 24)

    axes[1].scatter(df_policy['demand'], df_policy['price_multiplier'], alpha=0.3, s=20)
    axes[1].set_title('Price Multiplier vs Demand', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Demand (number of requests)')
    axes[1].set_ylabel('Price Multiplier')
    axes[1].grid(True, alpha=0.3)

    z = np.polyfit(df_policy['demand'], df_policy['price_multiplier'], 2)
    p = np.poly1d(z)
    x_trend = np.linspace(df_policy['demand'].min(), df_policy['demand'].max(), 100)
    axes[1].plot(x_trend, p(x_trend), "r--", linewidth=2, label='Trend')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('results/policy_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Policy analysis saved to results/policy_analysis.png")
    plt.close()


# ==============================================================
# Main
# ==============================================================
def main():
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    results = compare_strategies(episodes=10)

    # Analyze learned policy
    print("\n" + "="*60)
    print("Analyzing learned pricing policy...")
    agent = TD3Agent(state_dim=49, action_dim=16, max_action=1.5)
    try:
        agent.load('models/td3_pricing_agent.pth')
        analyze_pricing_policy(agent, episodes=3)
    except FileNotFoundError:
        print("⚠ Trained model not found, skipping policy analysis")

    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)
    print("\nGenerated Files:")
    print("  - results/strategy_comparison_scaled.png")
    print("  - results/policy_analysis.png")


if __name__ == "__main__":
    main()
