# RL start
using Distributions
using Statistics
using LinearAlgebra
using Plots

abs_project_path = normpath(joinpath(@__FILE__, ".."))
include(joinpath(abs_project_path, "src", "simple_bandit.jl"))


# - Simple bandit -
# Run experiment
true_means = [0.1, 0.3, 0.5]  # True win probabilities
bandits = [Bandit(m) for m in true_means]
ε = 0.1  # Exploration rate
n_trials = 1000

rewards, bandits = epsilon_greedy(bandits, ε, n_trials)

# Results
println("Total reward: ", sum(rewards))
println("Win rate: ", mean(rewards))
println("Number of pulls per bandit:")
for (i, b) in enumerate(bandits)
    println("Bandit $i: ", b.N, " pulls (True mean: ", b.true_mean, 
            ", Estimated mean: ", round(b.estimated_mean, digits=3), ")")
end

# Plot performance
plot(cumsum(rewards) ./ (1:n_trials), 
     label="ε-Greedy (ε=$ε)",
     xlabel="Trial", 
     ylabel="Average Reward",
     title="Multi-Armed Bandit Performance")
hline!([maximum(true_means)], 
       label="Optimal Possible", 
       linestyle=:dash)



# - Adaptive bandit -
# Run experiment
adapt_average = 0.2
bandits = [AdaptiveBandit(0.3, adapt_average), 
           AdaptiveBandit(0.5, adapt_average),
           AdaptiveBandit(0.7, adapt_average)]
ε = 0.1  # Exploration rate
n_trials = 1000
rewards, bandits = adaptive_epsilon_greedy(bandits, ε, n_trials)


# Results
println("Total reward: ", sum(rewards))
println("Win rate: ", mean(rewards))
println("Number of pulls per bandit:")
for (i, b) in enumerate(bandits)
    println("Bandit $i: ", b.N, " pulls (True mean: ", b.true_mean, 
            ", Estimated mean: ", round(b.estimated_mean, digits=3), ")")
end

# Plot performance
plot(cumsum(rewards) ./ (1:n_trials), 
     label="ε-Greedy (ε=$ε)",
     xlabel="Trial", 
     ylabel="Average Reward",
     title="Multi-Armed Bandit Performance")
hline!([maximum(true_means)], 
       label="Optimal Possible", 
       linestyle=:dash)

