# train.py — FINAL OPTIMIZED ELASTIC VERSION (with blue training curves)
import numpy as np
import torch
import matplotlib.pyplot as plt
import time, os
from environment import RideHailingEnv
from td3_agent import TD3Agent

def plot_training_results(rewards, metrics):
    """Plot training performance (blue curves)"""
    os.makedirs("results", exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0, 0].plot(rewards, color='steelblue')
    axes[0, 0].set_title("Episode Rewards", fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Total Reward")

    rides = [m["rides_completed"] for m in metrics]
    axes[0, 1].plot(rides, color='steelblue')
    axes[0, 1].set_title("Rides Completed per Episode", fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Rides Completed")

    revenue = [m["total_revenue"] for m in metrics]
    axes[1, 0].plot(revenue, color='steelblue')
    axes[1, 0].set_title("Total Revenue per Episode", fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Revenue ($)")

    empty = [m["empty_trips"] for m in metrics]
    axes[1, 1].plot(empty, color='steelblue')
    axes[1, 1].set_title("Empty Trips per Episode", fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Empty Trips")

    plt.tight_layout()
    plt.savefig("results/training_results.png", dpi=300)
    print("📊 Training curves saved to results/training_results.png")
    plt.close()


def train_dynamic_pricing(episodes=50, max_steps=1440):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print("🚀 Starting TD3 Training (FINAL) — FULL SIMULATION on CPU")
    print("=" * 70)

    # Environment setup
    env = RideHailingEnv(num_zones=16, num_vehicles=100)
    state_dim, action_dim, max_action = 49, 16, 1.5
    agent = TD3Agent(state_dim, action_dim, max_action)

    episode_rewards, metrics_log = [], []
    best_revenue = -np.inf

    for ep in range(1, episodes + 1):
        state = env.reset()
        ep_reward = 0
        start_time = time.time()

        for step in range(max_steps):
            # Action with exploration noise
            action = agent.select_action(state)
            noise = np.random.normal(0, 0.08, size=action_dim)
            action = np.clip(action + noise, 0.33, max_action)

            next_state, reward, done, info = env.step(action)

            # ---- Optimized Reward Shaping ----
            step_revenue = info.get("step_revenue", 0)
            fulfilled = info.get("fulfilled_requests", 0)
            total_demand = info.get("total_demand", 1)
            idle_vehicles = np.sum(info.get("idle_per_zone", np.zeros(16)))
            empty_trips = info.get("empty_trips", 0)

            demand_ratio = fulfilled / (total_demand + 1e-6)

            shaped_reward = (
                reward
                + 0.04 * step_revenue
                + 0.002 * demand_ratio * step_revenue
                - 0.0015 * idle_vehicles
                - 0.0005 * empty_trips
                - 0.001 * np.mean(np.maximum(action - 1.3, 0)) * step_revenue
            )

            agent.replay_buffer.add(state, action, shaped_reward, next_state, done)
            state = next_state
            ep_reward += shaped_reward

            if step % 2 == 0:
                agent.train(batch_size=256)

            if done:
                break

        metrics = env.get_metrics()
        ep_revenue = metrics.get("total_revenue", 0)
        ep_time = time.time() - start_time
        episode_rewards.append(ep_reward)
        metrics_log.append(metrics)

        print(
            f"\n✅ Episode {ep}/{episodes} complete in {ep_time:.1f}s\n"
            f"   → Episode Reward: {ep_reward:.2f} | Avg (last 5): {np.mean(episode_rewards[-5:]):.2f}\n"
            f"   → Rides: {metrics.get('rides_completed', 0)} | "
            f"Revenue: ${ep_revenue/1e6:,.2f}M | Empty: {metrics.get('empty_trips', 0)}"
        )

        if ep_revenue > best_revenue:
            best_revenue = ep_revenue
            agent.save("models/td3_pricing_agent.pth")
            print(f"💾 Model saved to models/td3_pricing_agent.pth")
            print(f"💾 Model improved and saved (Revenue: ${ep_revenue/1e6:,.2f}M)")

    plot_training_results(episode_rewards, metrics_log)
    print(f"\n🎯 Training complete — best revenue observed: ${best_revenue/1e6:,.2f}M")


if __name__ == "__main__":
    train_dynamic_pricing(episodes=50, max_steps=1440)
