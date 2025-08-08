# Grid world from scratch
using Plots

# Grid dimensions and special cells
const GRID_SIZE = (4, 4)
const START = (1, 1)
const GOAL = (4, 4)
const CLIFFS = [(4, 2), (2, 4)]

# Actions: UP, DOWN, LEFT, RIGHT
const ACTIONS = [
    (0, 1),   # UP
    (0, -1),  # DOWN
    (-1, 0),  # LEFT
    (1, 0)    # RIGHT
]
const ACTION_NAMES = ["↑", "↓", "←", "→"]


function is_terminal(state)
    return state == GOAL || state in CLIFFS
end


# reward function
function get_reward(next_state)
    if next_state == GOAL
        return 10.0
    elseif next_state in CLIFFS
        return -10.0
    else
        return -1.0  # Step penalty
    end
end


# ---- Q-learning ----
# Initialize Q-table: states × actions
function init_q_table()
    q_table = Dict{Tuple{Int,Int}, Vector{Float64}}()
    for x in 1:GRID_SIZE[1], y in 1:GRID_SIZE[2]
        q_table[(x, y)] = zeros(length(ACTIONS))
    end
    return q_table
end

# transition logic
function move(state::Tuple{Int,Int}, action::Int)
    x, y = state
    dx, dy = ACTIONS[action]
    new_x = clamp(x + dx, 1, GRID_SIZE[1])
    new_y = clamp(y + dy, 1, GRID_SIZE[2])
    return (new_x, new_y)
end

# ε-greedy action selection
function select_action(q_table, state, ϵ)
    if rand() < ϵ
        return rand(1:length(ACTIONS))  # Explore
    else
        return argmax(q_table[state])   # Exploit
    end
end
    
# Q-learning update
function q_learning!(q_table, state, action, reward, next_state, α, γ)
    best_next_q = is_terminal(next_state) ? 0.0 : maximum(q_table[next_state])
    q_table[state][action] += α * (reward + γ * best_next_q - q_table[state][action])
end

function train_agent(; episodes=1000, α=0.1, γ=0.9, ϵ_start=1.0, ϵ_end=0.01)
    q_table = init_q_table()
    ϵ = ϵ_start
    ϵ_decay = (ϵ_start - ϵ_end) / episodes
    
    reward_per_episode = []
    
    for ep in 1:episodes
        state = START
        current_reward = 0.0
        while !is_terminal(state)
            action = select_action(q_table, state, ϵ)
            next_state = move(state, action)
            reward = get_reward(next_state)
            q_learning!(q_table, state, action, reward, next_state, α, γ)
            state = next_state
            current_reward += reward
        end
        ϵ = max(ϵ_end, ϵ - ϵ_decay)  # Decay exploration rate
        push!(reward_per_episode, current_reward)
    end
    return q_table, reward_per_episode
end

function print_policy(q_table)
    for y in GRID_SIZE[2]:-1:1
        for x in 1:GRID_SIZE[1]
            if (x, y) == GOAL
                print("G ")
            elseif (x, y) in CLIFFS
                print("■ ")
            else
                best_action = argmax(q_table[(x, y)])
                print(ACTION_NAMES[best_action], " ")
            end
        end
        println()
    end
end

# Train and visualize
q_table, rewards = train_agent(episodes=10000, γ=0.8)
print_policy(q_table)

plot(rewards)

ticks_pos = collect(range(0.5, step=1, length=GRID_SIZE[1]))
ticks = string.(collect(1:GRID_SIZE[1]))

plot(
    grid=true,                     # Enable grid lines
    framestyle=:box,               # Full axis box
    xlabel="X axis", ylabel="Y axis",  # Axis labels
    title="Grid",            # Title
    xlims=(0, GRID_SIZE[1]), ylims=(0, GRID_SIZE[2]), # Axis limits
    aspect_ratio=:equal
)
# xticks!(ticks_pos, ticks)
# yticks!(ticks_pos, ticks)
annotate!(GOAL[1]-0.5, GOAL[2]-0.5, text("GOAL", :blue, :center, 10))
annotate!(CLIFFS[1][1]-0.5, CLIFFS[2][1]-0.5, text("CLIFF", :blue, :center, 10))
annotate!(CLIFFS[2][1]-0.5, CLIFFS[2][2]-0.5, text("CLIFF", :blue, :center, 10))

state = (4, 1)
counter = 1
annotate!(state[1]-0.5, state[2]-0.5, text(string(counter), :red, :center, 14))
action = argmax(q_table[state])
state = move(state, action)
counter += 1
annotate!(state[1]-0.5, state[2]-0.5, text(string(counter), :red, :center, 14))

