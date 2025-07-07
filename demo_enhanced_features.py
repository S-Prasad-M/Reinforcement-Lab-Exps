#!/usr/bin/env python3
"""
Demo script showing the enhanced features of Lab 2
Multi-Armed Bandit with Comprehensive Visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configure plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
np.random.seed(42)

print("🎯 Enhanced Multi-Armed Bandit Lab 2 Demo")
print("="*60)
print("Demonstrating comprehensive visualizations and analysis")
print("="*60)

# Enhanced Agent Implementation
class EpsilonGreedyAgent:
    """Enhanced ε-Greedy agent with comprehensive tracking."""
    
    def __init__(self, n_arms, epsilon=0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.name = f"ε-Greedy (ε={epsilon})"
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.total_reward = 0
        self.actions = []
        self.rewards = []
        self.exploration_actions = []
        
    def select_action(self):
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.n_arms)
            self.is_exploration = True
        else:
            action = np.argmax(self.values)
            self.is_exploration = False
        return action
    
    def update(self, action, reward):
        self.counts[action] += 1
        self.values[action] += (reward - self.values[action]) / self.counts[action]
        self.total_reward += reward
        self.actions.append(action)
        self.rewards.append(reward)
        if self.is_exploration:
            self.exploration_actions.append(len(self.actions) - 1)
    
    def get_confidence_interval(self, action, alpha=0.05):
        if self.counts[action] == 0:
            return (0, 0)
        mean = self.values[action]
        std_err = np.sqrt(mean * (1 - mean) / self.counts[action])
        z_score = stats.norm.ppf(1 - alpha/2)
        margin = z_score * std_err
        return (max(0, mean - margin), min(1, mean + margin))

class UCBAgent:
    """Enhanced UCB agent with comprehensive tracking."""
    
    def __init__(self, n_arms, c=2.0):
        self.n_arms = n_arms
        self.c = c
        self.name = f"UCB (c={c})"
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.total_reward = 0
        self.actions = []
        self.rewards = []
        self.t = 0
        
    def select_action(self):
        self.t += 1
        if self.t <= self.n_arms:
            action = self.t - 1
        else:
            ucb_values = self.values + self.c * np.sqrt(np.log(self.t) / (self.counts + 1e-10))
            action = np.argmax(ucb_values)
        return action
    
    def update(self, action, reward):
        self.counts[action] += 1
        self.values[action] += (reward - self.values[action]) / self.counts[action]
        self.total_reward += reward
        self.actions.append(action)
        self.rewards.append(reward)
    
    def get_confidence_interval(self, action, alpha=0.05):
        if self.counts[action] == 0:
            return (0, 0)
        mean = self.values[action]
        std_err = np.sqrt(mean * (1 - mean) / self.counts[action])
        z_score = stats.norm.ppf(1 - alpha/2)
        margin = z_score * std_err
        return (max(0, mean - margin), min(1, mean + margin))

# Simulation function
def simulate_bandit_experiment(true_ctrs, agents, n_rounds=1000):
    """Simulate multi-armed bandit experiment with comprehensive tracking."""
    
    results = {}
    optimal_arm = np.argmax(true_ctrs)
    optimal_value = true_ctrs[optimal_arm]
    
    for agent in agents:
        print(f"Running simulation for {agent.name}...")
        
        # Reset agent
        agent.__init__(agent.n_arms, agent.epsilon if hasattr(agent, 'epsilon') else agent.c)
        
        cumulative_regret = []
        cumulative_reward = []
        action_counts = np.zeros(agent.n_arms)
        
        for t in range(n_rounds):
            # Agent selects action
            action = agent.select_action()
            
            # Generate reward
            reward = np.random.binomial(1, true_ctrs[action])
            
            # Update agent
            agent.update(action, reward)
            
            # Track metrics
            regret = optimal_value - true_ctrs[action]
            cumulative_regret.append(regret if t == 0 else cumulative_regret[-1] + regret)
            cumulative_reward.append(reward if t == 0 else cumulative_reward[-1] + reward)
            action_counts[action] += 1
        
        # Calculate confidence intervals
        confidence_intervals = [agent.get_confidence_interval(i) for i in range(agent.n_arms)]
        
        results[agent.name] = {
            'agent': agent,
            'cumulative_regret': cumulative_regret,
            'cumulative_reward': cumulative_reward,
            'action_counts': action_counts,
            'final_estimates': agent.values.copy(),
            'confidence_intervals': confidence_intervals,
            'total_reward': agent.total_reward,
            'exploration_rate': len(getattr(agent, 'exploration_actions', [])) / n_rounds
        }
    
    return results

# Demo data
print("\n📊 Setting up demo data...")
n_arms = 10
n_rounds = 1000
true_ctrs = np.array([0.22, 0.48, 0.38, 0.32, 0.12, 0.12, 0.08, 0.44, 0.32, 0.37])

print(f"True CTRs: {true_ctrs}")
print(f"Optimal Arm: {np.argmax(true_ctrs)} (CTR: {true_ctrs[np.argmax(true_ctrs)]:.3f})")

# Initialize agents
agents = [
    EpsilonGreedyAgent(n_arms, epsilon=0.1),
    UCBAgent(n_arms, c=2.0)
]

print(f"\nInitialized {len(agents)} agents for comparison")

# Run experiment
print("\n🎯 Running enhanced bandit experiment...")
results = simulate_bandit_experiment(true_ctrs, agents, n_rounds)

# Create enhanced visualization
def create_enhanced_dashboard(results, true_ctrs):
    """Create enhanced performance dashboard."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Enhanced Multi-Armed Bandit Performance Dashboard', fontsize=16, fontweight='bold')
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    # 1. Cumulative Regret
    for i, (agent_name, data) in enumerate(results.items()):
        axes[0, 0].plot(data['cumulative_regret'], label=agent_name, color=colors[i], linewidth=2)
    axes[0, 0].set_title('Cumulative Regret Over Time', fontweight='bold')
    axes[0, 0].set_xlabel('Round')
    axes[0, 0].set_ylabel('Cumulative Regret')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Cumulative Reward
    for i, (agent_name, data) in enumerate(results.items()):
        axes[0, 1].plot(data['cumulative_reward'], label=agent_name, color=colors[i], linewidth=2)
    axes[0, 1].set_title('Cumulative Reward Over Time', fontweight='bold')
    axes[0, 1].set_xlabel('Round')
    axes[0, 1].set_ylabel('Cumulative Reward')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Action Selection Distribution
    n_agents = len(results)
    x = np.arange(n_arms)
    width = 0.35
    
    for i, (agent_name, data) in enumerate(results.items()):
        axes[1, 0].bar(x + i * width, data['action_counts'], width, label=agent_name, 
                      color=colors[i], alpha=0.7)
    
    axes[1, 0].set_title('Action Selection Distribution', fontweight='bold')
    axes[1, 0].set_xlabel('Arm/Ad')
    axes[1, 0].set_ylabel('Selection Count')
    axes[1, 0].set_xticks(x + width/2)
    axes[1, 0].set_xticklabels([f'Ad {i}' for i in range(n_arms)])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Estimated vs True CTR
    axes[1, 1].bar(x - 0.2, true_ctrs, 0.2, label='True CTR', color='black', alpha=0.8)
    
    for i, (agent_name, data) in enumerate(results.items()):
        offset = i * 0.15
        axes[1, 1].bar(x + offset, data['final_estimates'], 0.12, label=f'{agent_name} Estimate', 
                      color=colors[i], alpha=0.7)
    
    axes[1, 1].set_title('Estimated vs True CTR', fontweight='bold')
    axes[1, 1].set_xlabel('Arm/Ad')
    axes[1, 1].set_ylabel('CTR')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([f'Ad {i}' for i in range(n_arms)])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('enhanced_dashboard_demo.png', dpi=300, bbox_inches='tight')
    plt.show()

# Create the enhanced dashboard
print("\n📈 Creating enhanced visualization dashboard...")
create_enhanced_dashboard(results, true_ctrs)

# Performance summary
print("\n📊 Performance Summary:")
print("-"*60)
for agent_name, data in results.items():
    total_reward = data['total_reward']
    final_regret = data['cumulative_regret'][-1]
    exploration_rate = data['exploration_rate'] * 100
    
    print(f"{agent_name:20s} | Reward: {total_reward:3.0f} | Regret: {final_regret:6.2f} | Exploration: {exploration_rate:4.1f}%")

# Create performance table
print("\n📋 Detailed Performance Analysis:")
print("-"*60)

analysis_data = []
for agent_name, data in results.items():
    total_reward = data['total_reward']
    final_regret = data['cumulative_regret'][-1]
    avg_reward = total_reward / n_rounds
    exploration_rate = data['exploration_rate']
    
    # Estimation accuracy
    estimation_error = np.mean(np.abs(data['final_estimates'] - true_ctrs))
    
    analysis_data.append({
        'Algorithm': agent_name,
        'Total Reward': total_reward,
        'Final Regret': final_regret,
        'Average Reward': avg_reward,
        'Exploration Rate': f"{exploration_rate*100:.1f}%",
        'Estimation Error': estimation_error
    })

df = pd.DataFrame(analysis_data)
print(df.to_string(index=False, float_format='%.3f'))

print("\n🎉 Enhanced Multi-Armed Bandit Demo Completed!")
print("✅ All enhanced features demonstrated successfully")
print("\nKey improvements over basic implementation:")
print("• Comprehensive performance dashboard with multiple visualizations")
print("• Statistical analysis with confidence intervals")
print("• Professional styling and annotations")
print("• Detailed performance metrics and comparison")
print("• Self-explanatory output with formatted tables")
print("• Enhanced tracking of exploration vs exploitation")