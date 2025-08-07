# Simple bandit
using Statistics
using LinearAlgebra

# Define a Bandit type
mutable struct Bandit
    true_mean::Float64  # Actual win probability
    estimated_mean::Float64  # Current estimate
    N::Int64  # Number of pulls
end

# Constructor
Bandit(true_mean) = Bandit(true_mean, 0.0, 0)

# Pull the bandit's arm (return 1 or 0)
function pull(b::Bandit)
    return rand() < b.true_mean ? 1 : 0
end

# Update the bandit's estimate
function update_estimate!(b::Bandit, reward::Int)
    b.N += 1
    b.estimated_mean = (1 - 1.0/b.N) * b.estimated_mean + (1.0/b.N) * reward
end

# ε-Greedy algorithm
function epsilon_greedy(bandits::Vector{Bandit}, ε::Float64, n_trials::Int)
    rewards = zeros(n_trials)
    for t in 1:n_trials
        if rand() < ε
            # Explore: random bandit
            j = rand(1:length(bandits))
        else
            # Exploit: bandit with highest estimate
            j = argmax([b.estimated_mean for b in bandits])
        end
        
        reward = pull(bandits[j])
        rewards[t] = reward
        update_estimate!(bandits[j], reward)
    end
    return rewards, bandits
end
