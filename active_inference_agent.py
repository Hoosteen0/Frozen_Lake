import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import numpy as np
import matplotlib.pyplot as plt
from pymdp import utils
from pymdp.agent import Agent

def create_a(observations, states, env_map):
    env_size = len(env_map)

    num_obs = [len(o_factor) for o_factor in observations]
    num_states = [len(state_factor) for state_factor in states]

    A = utils.initialize_empty_A(num_obs=num_obs, num_states=num_states)

    # filling out location obs given location state
    A[0][:, :] = np.eye(A[0].shape[0])

    # filling out tile type obs (row) given location state (col)
    for i in range(env_size):
        for j in range(env_size):
            curr_pos = (i * env_size) + j

            if env_map[i][j] == 'G':
                A[1][3, curr_pos] = 1.0
            elif env_map[i][j] == 'S':
                A[1][0, curr_pos] = 1.0
            elif env_map[i][j] == 'H':
                A[1][2, curr_pos] = 1.0
            else:
                A[1][1, curr_pos] = 1.0
    # print(f'A:\n{A[1][:,:]}')
    return A

def create_b(states, actions, env_map):
    env_size = len(env_map)
    num_states = [len(state_factor) for state_factor in states]
    num_controls = [len(control_factor) for control_factor in actions]
    B = utils.initialize_empty_B(num_states, num_controls)

    # actions list for reference: [left, down, right, up]
    for i in range(env_size):
        for j in range(env_size):
            curr_pos = (i * env_size) + j

            # going up
            if i == 0:
                B[0][curr_pos, curr_pos, 3] = 1.0
            if i < env_size - 1:
                # print(curr_pos, i, j)
                B[0][curr_pos, curr_pos+env_size, 3] = 1.0

            # going left
            if j == 0:
                B[0][curr_pos, curr_pos:curr_pos+2, 0] = 1.0
            elif j == env_size - 1:
                pass
                # B[0][curr_pos, curr_pos-1, 0] = 1.0
            else:
                B[0][curr_pos, curr_pos+1, 0] = 1.0

    # going right looks like a flip of left horizontally and then vertically
    B[0][:,:,2] = np.fliplr(np.flipud(B[0][:,:,0]))

    # going down is up flipped over horizontal and vertical
    B[0][:, :, 1] = np.fliplr(np.flipud(B[0][:, :, 3]))
    # for mat in B:
    #     for action in actions[0]:
    #         print(f'B-transitions given action {action}:\n{mat[:, :, actions[0].index(action)]}')
    return B

def create_c(observations, env_size):
    num_obs = [len(obs) for obs in observations]
    C = utils.obj_array_zeros(num_obs)

    # we are concerned with C[1], as that specifies the preferred tile type observations to see
    # C structure is: C[0] = array of len 16. C[1] = array of len 4
    # C1 = ['S', 'F' 'H' 'G']

    # 2/6/2025 test idea: calculate the manhattan distance from each position to the goal, and use that to specify
    #   the agent's preferences.
    # num_positions = len(observations[0])
    # for i in range(env_size**2):
    #     row = num_positions // env_size
    #     col = num_positions % env_size
    #     manhattan_dist = (env_size - row) + (env_size - col)
    #     C[0][i] = -1 * manhattan_dist
    C[1] = np.array([0, 0, -2.1, 2.2]) # the values of these are in log probability space
    # -0.1, -0.0001, -2.1, 2.2
    # print(f'C={C}')
    return C

def create_d(states):
    num_states = [len(state_factor) for state_factor in states]
    D = utils.obj_array_zeros(num_states)
    D[0][0:num_states[0]] = 1 / num_states[0] # start in first position. Why is this important to the agent working??
    D[0] = D[0] / (D[0].sum(axis=0)) # normalize the matrix
    # print(f'D={D}')
    return D

def specify_pomdp(env_map):
    env_size = len(env_map)

    num_loc = env_size**2

    s_loc = [i for i in range(num_loc)]
    S = [s_loc]

    o_loc = [i for i in range(num_loc)]
    o_tile_type = ['S', 'F', 'H', 'G']
    O = [o_loc, o_tile_type]

    u_mov = ['left', 'down', 'right', 'up']
    U = [u_mov]

    A = create_a(observations=O, states=S, env_map=env_map)
    B = create_b(states=S, actions=U, env_map=env_map)
    C = create_c(observations=O, env_size=env_size)
    D = create_d(states=S)

    return S, O, U, A, B, C, D

def main():
    # specify the environment size here
    env_size = 4 # works up to size 4.
    seed = 422
    env_map = generate_random_map(env_size)
    for row in env_map:
        print(row)
    """
    map size = 6
    seed = 422
    desc =  SFFFFF
            FFFHHH
            HFFHFF
            FFHFFF
            FFFFFH
            FFHFFG
    desc size = 6 =
            SHFFFF
            FFHFHF
            FFFFFH
            FFHFFF
            HHFFHH
            HFFFFG
            
    desc size = 3
            SHF
            FFF
            FHG
    
    """
    env_map = ['SHFFFF',
                'FFHFHF',
                'FFFFFH',
                'FFHFFF',
                'HHFFHH',
                'HFFFFG']
    env = gym.make('FrozenLake-v1', is_slippery=False, max_episode_steps=20, desc=env_map, render_mode='human')
    T = 10

    S, O, U, A, B, C, D = specify_pomdp(env_map=env_map)

    my_agent = Agent(A=A, B=B, C=C, D=D, policy_len=1, inference_horizon=1, save_belief_hist=True) # adjust policy length to plan x number of steps in future.

    try:
        observation, info = env.reset(seed=None)
        print(f'info: {info}')
        position = observation
        tile_type = O[1].index(env_map[observation // env_size][observation % env_size]) # using the env_map to get the
        # tile_type for the agent.
        obs = [position, tile_type]

        t = 0
        done = False
        while not done:
            qs = my_agent.infer_states(obs)

            q_pi, efe = my_agent.infer_policies()
            chosen_action_id = my_agent.sample_action()
            # print(chosen_action_id)
            chosen_action = int(chosen_action_id[0])
            print(f'chosen action: {chosen_action}')

            observation, reward, terminated, truncated, info = env.step(chosen_action)

            tile_type = O[1].index(env_map[observation // env_size][observation % env_size])
            obs = [position, tile_type]
            # print(f'At timestep={t}\n\tqs={qs}\n\tq_pi={q_pi}\n\tefe={efe}\n\taction={chosen_action}\n\tobs={obs}')
            if terminated or truncated:
                done = True
            t += 1
    finally:
        env.close()
        print(my_agent.qs_hist)
        print(my_agent.q_pi_hist)


main()
