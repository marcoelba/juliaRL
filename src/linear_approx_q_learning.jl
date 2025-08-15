# Grid world from scratch
import Plots
using Parameters
using StatsFuns
using LinearAlgebra


@with_kw struct ContinuousGrid
    # Grid dimensions and special cells
    grid_limits::Tuple{Int64, Int64}
    # position in space
    goal::Vector{Int64}
    # positions in space
    cliffs::Vector{Tuple{Int64, Int64}} = []
    cliff_radius::Float64 = 1.0

    # Discrete Actions: UP, DOWN, LEFT, RIGHT (move by one unit in that direction)
    actions::Vector{Tuple{Int64, Int64}} = [(0, 1), (0, -1), (-1, 0), (1, 0)]
    action_names::Vector{String} = ["↑", "↓", "←", "→"]

end


@with_kw mutable struct Environment
    # Grid dimensions and special cells
    state::Vector{Int64}
    # positions in space
    action::Int64 = 0
    # previous iteration
    previous_state::Vector{Int64}=[0, 0]
    # positions in space
    previous_action::Int64 = 0
end
Environment() = Environment(state=[1, 1], action=0)
Environment(s) = Environment(state=s, action=0)


function is_terminal(grid::ContinuousGrid, state::Vector{Int64})
    # distance from the goal - stop when close to goal
    distance_from_goal = norm(state - grid.goal)
    return distance_from_goal < 1.
end

# reward function
function get_reward(grid::ContinuousGrid, state::Vector{Int64}, illegal_move::Bool, previous_state::Vector{Int64})
    dist = sum(abs.(state .- grid.goal))
    previous_dist = sum(abs.(previous_state .- grid.goal))
    jumping_back_forward = previous_state == state

    if state == grid.goal
        r = +10.0
    else
        r = -0.1
    end

    # improve reward if distance diminish
    r += 0.05 * (previous_dist - dist)

    if jumping_back_forward
        r -= 0.5
    end

    # check for illegal moves
    if illegal_move
        r -= 5.0
    end

    return r
end

# transition logic
function move(grid::ContinuousGrid, state::Vector{Int64}, action::Int, can_slip::Bool)
    x, y = state

    dx, dy = grid.actions[action]
    new_x = clamp(x + dx, 1, grid.grid_limits[1])
    new_y = clamp(y + dy, 1, grid.grid_limits[2])

    if can_slip
        # Stochastic slip (x% chance of random move)
        if rand() < 0.1
            setd = setdiff(1:length(grid.actions), action)
            action_slip = rand(grid.actions[setd])
            dx, dy = action_slip
            new_x = clamp(x + dx, 1, grid.grid_size[1])
            new_y = clamp(y + dy, 1, grid.grid_size[2])

            new_state = [new_x, new_y]
            return new_state
        end
    end

    new_state = [new_x, new_y]
    wall_crash = new_state == state

    return new_state, wall_crash
end


# ---- Q-learning ----
# Approximate the Q-values through a (linear) model
# Takes in input the states that are continuous and predict the discrete action
# here just make it binary, try to avoid overflow
function dist_discretizer(x)
    if x == 0
        return 0
    elseif x > 0
        return 1
    else
        return -1
    end
end

function features_processsing(grid::ContinuousGrid, state::Vector{Int64})
    # provide position relative to the GOAL 
    dx, dy = dist_discretizer.(grid.goal .- state)

    # norm_dist = norm(grid.goal .- state)
    # provide distance from the boundaries
    left_dist = (state[1] - 1) > 0             # Distance to left wall
    right_dist = (grid.grid_limits[1] - state[1]) > 0  # Distance to right wall
    down_dist = (state[2] - 1) > 0              # Distance to bottom wall
    up_dist = (grid.grid_limits[2] - state[2]) > 0     # Distance to top wall
    
    features = Float64.([
        dx, dy,
        left_dist, right_dist, down_dist, up_dist
    ])

    return features
end

function q_values(grid::ContinuousGrid, theta::AbstractArray, state::Vector{Int64})
    # theta has dim: #states x #actions
    features = features_processsing(grid, state)
    pred_action = theta' * features
    return pred_action
end

# ε-greedy action selection
function select_action(theta::AbstractArray, state::Vector{Int64}, ϵ::Float64)
    if rand() < ϵ
        return rand(1:size(theta, 2))  # Explore
    else
        return argmax(q_values(grid, theta, state))   # Exploit
    end
end

# Q-learning update
function q_learning!(grid::ContinuousGrid, theta, state, action, reward, next_state, α, γ; target_theta=[])
    # best_next_q = is_terminal(grid, next_state) ? 0.0 : maximum(q_table[next_state])
    
    q_current = q_values(grid, theta[:, action], state)
    # q_target = reward + γ * maximum(q_values(grid, theta, next_state))

    # using a target model (theta) updated periodically
    next_features = features_processsing(grid, next_state)
    if length(target_theta) > 0
        q_target = reward + γ * maximum(target_theta' * next_features)
    else
        q_target = reward + γ * maximum(theta' * next_features)
    end

    q_error = q_target - q_current
    # update theta
    features = features_processsing(grid, state)
    theta[:, action] += α * q_error * features
end

function train_agent(;
    grid::ContinuousGrid,
    starting_state=[1, 1],
    episodes=1000,
    update_interval=50,
    alpha_start=0.01,
    γ=0.9,
    eps_start=0.99,
    eps_end=0.01,
    can_slip::Bool=false
    )
    theta = rand(n_features, length(grid.actions)) * 0.01
    target_theta = deepcopy(theta)
    
    grid_limits = grid.grid_limits
    visit_count = zeros(grid_limits)

    ϵ = eps_start
    ϵ_decay = (eps_start - eps_end) / episodes
    
    reward_per_episode = []
    
    for ep in 1:episodes
        env = Environment(starting_state)

        alpha = max(0.001, alpha_start * exp(-ep / episodes))

        # random start every x episodes
        if rand() < ϵ
            env.state = [rand(1:grid_limits[1]), rand(1:grid_limits[2])]
        end
        current_reward = 0.0
        iter = 0
        while !is_terminal(grid, env.state)
            iter += 1
            env.action = select_action(theta, env.state, ϵ)
            next_state, wall_crash = move(grid, env.state, env.action, can_slip)
            reward = get_reward(grid, next_state, wall_crash, env.previous_state)
            q_learning!(grid, theta, env.state, env.action, reward, next_state, alpha, γ)
            env.previous_state = env.state
            env.state = next_state
            current_reward += reward

            visit_count[grid_limits[2] - next_state[2]+1, next_state[1]] += 1.
            # sync target network
            # if iter % update_interval == 0
            #     target_theta = deepcopy(theta)
            # end
        end
        ϵ = max(eps_end, ϵ - ϵ_decay)  # Decay exploration rate
        push!(reward_per_episode, current_reward)
    end
    return theta, reward_per_episode, visit_count
end

function print_policy(grid::ContinuousGrid, theta)
    for y in grid.grid_limits[2]:-1:1
        for x in 1:grid.grid_limits[1]
            if [x, y] == grid.goal
                print("G ")
            elseif [x, y] in grid.cliffs
                print("■ ")
            else
                best_action = argmax(q_values(grid, theta, [x, y]))
                print(grid.action_names[best_action], " ")
            end
        end
        println()
    end
end


function print_path_grid(grid::ContinuousGrid, theta; starting_state=[1, 1])

    # grid
    plt = Plots.plot(
        grid=true,                     # Enable grid lines
        framestyle=:box,               # Full axis box
        xlabel="X axis", ylabel="Y axis",  # Axis labels
        xticks=range(0, grid.grid_limits[1]),
        yticks=range(0, grid.grid_limits[2]),
        title="Grid",            # Title
        xlims=(0, grid.grid_limits[1]), ylims=(0, grid.grid_limits[2]), # Axis limits
        aspect_ratio=:equal
    )
    Plots.annotate!(grid.goal[1]-0.5, grid.goal[2]-0.5, Plots.text("GOAL", :blue, :center, 8))
    for cliff in grid.cliffs
        Plots.annotate!(cliff[1]-0.5, cliff[2]-0.5, Plots.text("CLIFF", :blue, :center, 8))
    end
    Plots.display(plt)

    # path
    counter = 1
    Plots.annotate!(starting_state[1]-0.5, starting_state[2]-0.5, Plots.text("START", :red, :center, 8))
    state = starting_state
    
    while !is_terminal(grid, state)

        action = argmax(q_values(grid, theta, state))
        state, crash = move(grid, state, action, false)
        counter += 1

        # annotate!(state[1]-0.5, state[2]-0.5, text(string(counter), :red, :center, 14))
        box_x = [state[1]-1, state[1], state[1], state[1]-1, state[1]-1]
        box_y = [state[2]-1, state[2]-1, state[2], state[2], state[2]-1]

        Plots.plot!(box_x, box_y, 
            fill=true,           # Enable fill
            fillalpha=0.8,       # Transparency (0=invisible, 1=opaque)
            fillcolor="red", # Shade color
            label=false,
            linecolor=false,    # Border color
            linewidth=0,         # Border thickness
        )
    end

    display(plt)
end


# -------------------------------------------------
# ------------------- Test ------------------------
# -------------------------------------------------

# Train and visualize
grid = ContinuousGrid(
    grid_limits=(10, 10),
    goal=[5, 8],
    cliffs=[]
)

n_features = length(features_processsing(grid, [1, 1]))

theta, rewards, visits = train_agent(
    grid=grid, episodes=500,
    alpha_start=0.01, eps_start=0.99, eps_end=0.1,
    γ=0.9, can_slip=false
)
print_policy(grid, theta)
Plots.plot(rewards)
Plots.heatmap(visits ./ maximum(visits))


print_path_grid(grid, theta, starting_state=[1, 1])
print_path_grid(grid, theta, starting_state=[4, 4])
print_path_grid(grid, theta, starting_state=[10, 10])
