# Evolutionary Strategies
using Statistics


const GRID_SIZE = (4, 4)
const START = (1, 1)
const GOAL = (4, 4)
const CLIFFS = [(4, 2), (2, 4)]

const ACTIONS = [:UP, :DOWN, :LEFT, :RIGHT]
const ACTION_DIRS = Dict(:UP => (0,1), :DOWN => (0,-1), :LEFT => (-1,0), :RIGHT => (1,0))
const ACTION_NAMES2 = Dict(:UP => "↑", :DOWN => "↓", :LEFT => "←", :RIGHT => "→")

# Policy representation: Matrix of actions (one per state)
policy = fill(:RIGHT, GRID_SIZE...)  # Initial policy (arbitrary)

function move(state::Tuple{Int,Int}, action::Symbol)
    x, y = state
    dx, dy = ACTION_DIRS[action]
    new_x, new_y = x + dx, y + dy
    # Clip to grid bounds
    new_x = clamp(new_x, 1, GRID_SIZE[1])
    new_y = clamp(new_y, 1, GRID_SIZE[2])
    # Stochastic slip (x% chance of random move)
    # if rand() < 0.1
    #     a_slip = rand(setdiff(ACTIONS, [action]))
    #     return move(state, a_slip)
    # end
    return (new_x, new_y)
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


# Fitness function: Evaluate cumulative reward of a policy
function evaluate(policy, episodes=10)
    total_reward = 0.0
    for _ in 1:episodes
        state = START
        episode_reward = 0.0
        while state != GOAL
            action = policy[state...]
            state_new = move(state, action)
            r = get_reward(state_new)
            episode_reward += r
            state = state_new
        end
        total_reward += episode_reward
    end
    return total_reward / episodes
end

# Evolution Strategies (ES) training
function evolve!(policy, generations=100, pop_size=50, σ=0.3)
    best_policy = copy(policy)
    best_fitness = evaluate(best_policy)
    
    for _ in 1:generations
        # Generate population by perturbing the best policy
        population = []
        for _ in 1:pop_size
            candidate = copy(best_policy)
            # Mutate each state's action with probability σ
            for x in 1:GRID_SIZE[1], y in 1:GRID_SIZE[2]
                if rand() < σ
                    candidate[x, y] = rand(ACTIONS)
                end
            end
            push!(population, candidate)
        end
        
        # Evaluate fitness
        fitness = [evaluate(p) for p in population]
        best_idx = argmax(fitness)
        
        # Update best policy
        if fitness[best_idx] > best_fitness
            best_policy = population[best_idx]
            best_fitness = fitness[best_idx]
        end
    end
    return best_policy
end

function print_policy(policy)
    for y in GRID_SIZE[2]:-1:1
        for x in 1:GRID_SIZE[1]
            if (x, y) == GOAL
                print("G ")
            elseif (x, y) in CLIFFS
                print("■ ")
            else
                print(ACTION_NAMES2[policy[x, y]], " ")
            end
        end
        println()
    end
end

# Train and print ES policy
policy = evolve!(policy)
println("\nES policy:")
print_policy(policy)
