# Adaptive bandit
using Statistics
using LinearAlgebra


mutable struct AdaptiveBandit
    true_mean::Float64
    estimated_mean::Float64
    N::Int
    α::Float64  # Learning rate (e.g., 0.1 for fast adaptation)
end

# Constructor
AdaptiveBandit(true_mean, alpha) = AdaptiveBandit(true_mean, 0.0, 0, alpha)

# Pull the bandit's arm (return 1 or 0)
function pull(b::AdaptiveBandit)
    return rand() < b.true_mean ? 1 : 0
end

function update_ab!(b::AdaptiveBandit, reward::Int)
    b.N += 1
    b.estimated_mean = (1 - b.α) * b.estimated_mean + b.α * reward
end

function adaptive_epsilon_greedy(bandits::Vector{AdaptiveBandit}, ε=0.1, n_trials=1000)
    rewards = zeros(n_trials)
    for t in 1:n_trials
        if rand() < ε
            j = rand(1:length(bandits))
        else
            j = argmax([b.estimated_mean for b in bandits])
        end
        reward = pull(bandits[j])
        update_ab!(bandits[j], reward)
        rewards[t] = reward
    end

    return rewards, bandits
end
