# Grid world from scratch
import Plots
using Parameters
using StatsFuns


@with_kw struct Grid
    # Grid dimensions and special cells
    grid_size::Tuple{Int64, Int64}
    goal::Tuple{Int64, Int64}
    cliffs::Vector{Tuple{Int64, Int64}} = []

    # Actions: UP, DOWN, LEFT, RIGHT
    actions::Vector{Tuple{Int64, Int64}} = [(0, 1), (0, -1), (-1, 0), (1, 0)]
    action_names::Vector{String} = ["↑", "↓", "←", "→"]

end


# grid = Grid(grid_size=(4, 4), goal=(4, 4), cliffs=[(2,3), (1,4)])
# grid.goal
# grid.grid_size
# grid.cliffs


function is_terminal(grid::Grid, state::Tuple{Int64, Int64})
    return state == grid.goal || state in grid.cliffs
end


# reward function
function get_reward(grid::Grid, state::Tuple{Int64, Int64})
    if state == grid.goal
        return 10.0
    elseif state in grid.cliffs
        return -10.0
    else
        return -1.0  # Step penalty
    end
end

# transition logic
function move(grid::Grid, state::Tuple{Int,Int}, action::Int, can_slip::Bool)
    x, y = state

    dx, dy = grid.actions[action]
    new_x = clamp(x + dx, 1, grid.grid_size[1])
    new_y = clamp(y + dy, 1, grid.grid_size[2])

    if can_slip
        # Stochastic slip (x% chance of random move)
        if rand() < 0.1
            setd = setdiff(1:length(grid.actions), action)
            action_slip = rand(grid.actions[setd])
            dx, dy = action_slip
            new_x = clamp(x + dx, 1, grid.grid_size[1])
            new_y = clamp(y + dy, 1, grid.grid_size[2])

            return (new_x, new_y)
        end
    end

    return (new_x, new_y)
end


# ---- Q-learning ----
# Initialize Q-table: states × actions
function init_q_table(grid::Grid)
    q_table = Dict{Tuple{Int,Int}, Vector{Float64}}()
    for x in 1:grid.grid_size[1], y in 1:grid.grid_size[2]
        q_table[(x, y)] = zeros(length(grid.actions))
    end
    return q_table
end

# ε-greedy action selection
function select_action(q_table, state, ϵ)
    if rand() < ϵ
        return rand(1:length(q_table[state]))  # Explore
    else
        return argmax(q_table[state])   # Exploit
    end
end
    
# Q-learning update
function q_learning!(grid::Grid, q_table, state, action, reward, next_state, α, γ)
    best_next_q = is_terminal(grid, next_state) ? 0.0 : maximum(q_table[next_state])
    q_table[state][action] += α * (reward + γ * best_next_q - q_table[state][action])
end

function train_agent(;
    grid::Grid,
    episodes=1000,
    α=0.1,
    γ=0.9,
    ϵ_start=1.0,
    ϵ_end=0.01,
    starting_state=(1, 1),
    can_slip::Bool=false
    )
    q_table = init_q_table(grid)
    ϵ = ϵ_start
    ϵ_decay = (ϵ_start - ϵ_end) / episodes
    
    reward_per_episode = []
    
    for ep in 1:episodes
        state = starting_state
        current_reward = 0.0
        while !is_terminal(grid, state)
            action = select_action(q_table, state, ϵ)
            next_state = move(grid, state, action,can_slip)
            reward = get_reward(grid, next_state)
            q_learning!(grid, q_table, state, action, reward, next_state, α, γ)
            state = next_state
            current_reward += reward
        end
        ϵ = max(ϵ_end, ϵ - ϵ_decay)  # Decay exploration rate
        push!(reward_per_episode, current_reward)
    end
    return q_table, reward_per_episode
end

function print_policy(grid::Grid, q_table)
    for y in grid.grid_size[2]:-1:1
        for x in 1:grid.grid_size[1]
            if (x, y) == grid.goal
                print("G ")
            elseif (x, y) in grid.cliffs
                print("■ ")
            else
                best_action = argmax(q_table[(x, y)])
                print(grid.action_names[best_action], " ")
            end
        end
        println()
    end
end


function print_path_grid(grid::Grid, q_table; starting_state=(1, 1))

    # grid
    plt = Plots.plot(
        grid=true,                     # Enable grid lines
        framestyle=:box,               # Full axis box
        xlabel="X axis", ylabel="Y axis",  # Axis labels
        xticks=range(0, grid.grid_size[1]),
        yticks=range(0, grid.grid_size[2]),
        title="Grid",            # Title
        xlims=(0, grid.grid_size[1]), ylims=(0, grid.grid_size[2]), # Axis limits
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

        action = argmax(q_table[state])
        state = move(grid, state, action, false)
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
grid = Grid(
    grid_size=(8, 8),
    goal=(7, 8),
    cliffs=[(2,3), (1,4), (5,6), (6,4)],
    can_slip=true
)
q_table, rewards = train_agent(grid=grid, episodes=10000, γ=0.9)
print_policy(grid, q_table)
Plots.plot(rewards)

norm_table = init_q_table(grid)
for (ii, key) in enumerate(keys(norm_table))
    norm_table[key] = softmax(q_table[key])
end

print_path_grid(q_grid, q_table, starting_state=(1, 2))
print_path_grid(q_grid, norm_table, starting_state=(1, 1))

