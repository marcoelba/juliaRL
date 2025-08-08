# Grid world from scratch
using Plots
using Parameters


@with_kw struct Grid
    # Grid dimensions and special cells
    grid_size::Tuple{Int64, Int64}
    goal::Tuple{Int64, Int64}
    cliffs::Vector{Tuple{Int64, Int64}} = []

    # Actions: UP, DOWN, LEFT, RIGHT
    actions::Vector{Tuple{Int64, Int64}} = [(0, 1), (0, -1), (-1, 0), (1, 0)]
    action_names::Vector{String} = ["↑", "↓", "←", "→"]

end


grid = Grid(grid_size=(4, 4), goal=(4, 4), cliffs=[(2,3), (1,4)])
grid.goal
grid.grid_size
grid.cliffs


function is_terminal(grid::Grid, state::Tuple{Int64, Int64})
    return state == grid.goal || state in grid.cliffs
end
is_terminal(grid, (4, 4))


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
get_reward(grid, (4, 4))
get_reward(grid, (1, 4))

# transition logic
function move(grid::Grid, state::Tuple{Int,Int}, action::Int)
    x, y = state
    dx, dy = grid.actions[action]
    new_x = clamp(x + dx, 1, grid.grid_size[1])
    new_y = clamp(y + dy, 1, grid.grid_size[2])
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
q_table = init_q_table(grid::Grid)

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
    starting_state=(1, 1)
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
            next_state = move(grid, state, action)
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
    plt = plot(
        grid=true,                     # Enable grid lines
        framestyle=:box,               # Full axis box
        xlabel="X axis", ylabel="Y axis",  # Axis labels
        xticks=range(0, grid.grid_size[1]),
        yticks=range(0, grid.grid_size[2]),
        title="Grid",            # Title
        xlims=(0, grid.grid_size[1]), ylims=(0, grid.grid_size[2]), # Axis limits
        aspect_ratio=:equal
    )
    annotate!(grid.goal[1]-0.5, grid.goal[2]-0.5, text("GOAL", :blue, :center, 8))
    for cliff in grid.cliffs
        annotate!(cliff[1]-0.5, cliff[2]-0.5, text("CLIFF", :blue, :center, 8))
    end
    display(plt)

    # path
    counter = 1
    annotate!(starting_state[1]-0.5, starting_state[2]-0.5, text("START", :red, :center, 8))
    state = starting_state
    
    while !is_terminal(grid, state)
        action = argmax(q_table[state])
        state = move(grid, state, action)
        counter += 1
        annotate!(state[1]-0.5, state[2]-0.5, text(string(counter), :red, :center, 14))
    end

    display(plt)
end


# -------------------------------------------------
# ------------------- Test ------------------------
# -------------------------------------------------

# Train and visualize
grid = Grid(grid_size=(8, 8), goal=(7, 8), cliffs=[(2,3), (1,4), (5,6)])
q_table, rewards = train_agent(grid=grid, episodes=10000, γ=0.8)
print_policy(grid, q_table)

plot(rewards)

print_path_grid(grid, q_table, starting_state=(1, 1))
