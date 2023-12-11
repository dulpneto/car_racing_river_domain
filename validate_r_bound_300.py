
from collections import deque
from CarRacingDQNAgent import CarRacingDQNAgent
from car_river_crossing import CarRiverCrossing
from common_functions import generate_state_frame_stack_from_queue
from common_functions import process_state_image
import numpy as np

SKIP_FRAMES = 3
RENDER = True

FILE_NAME = './rewards_fixed_policy_20230522_4.csv'

def create_log():
    with open(FILE_NAME, 'w') as f:
        f.write('policy,reward\n')

def log(policy_id, e, play_episodes, total_reward):
    with open(FILE_NAME, 'a') as f:
        f.write('{},{}\n'.format(policy_id, total_reward))
    print('Policy: {}, Episode: {}/{}, Total Rewards: {}'.format(policy_id, e + 1, play_episodes, total_reward))

if __name__ == '__main__':

    play_area = 300
    zoom = 1.8

    if RENDER:
        env = CarRiverCrossing(render_mode='human', play_field_area=play_area, zoom=zoom)
    else:
        env = CarRiverCrossing(play_field_area=play_area, zoom=zoom)

    create_log()

    all_rewards = {}

    train_models = [
        './save_fixed_model_test/trial_-2.0_30000.h5',
        './save_fixed_model_test/trial_-1.0_50000.h5',
        #'./save_fixed_model_test/trial_0.0_60000.h5',
        #'./save_fixed_model_test/trial_1.0_34500.h5',
        #'./save_fixed_model_test/trial_1.0_35000.h5',
    ]

    policies = range(len(train_models))

    for policy_id in policies:

        all_rewards[policy_id] = []

        train_model = train_models[policy_id]

        print('Model', train_model)

        # Set epsilon to 0 to ensure all actions are instructed by the agent
        agent = CarRacingDQNAgent(epsilon=0, lamb=0.0)
        agent.load(train_model)

        play_episodes = 2

        e= 0

        reset_count = 0

        while e < play_episodes:
            init_state, info = env.reset()
            init_state = process_state_image(init_state)

            total_reward = 0
            punishment_counter = 0
            state_frame_stack_queue = deque([init_state] * agent.frame_stack_num, maxlen=agent.frame_stack_num)
            time_frame_counter = 1

            rewards = []

            #if reset_count > 4:
            #    print('NOT ', train_model)
            #    break

            while True:

                current_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)
                action = agent.act(current_state_frame_stack)
                #if 'trial_0.0_60000' in train_model:
                #    env.step(action)
                #    print('ACTION', action)
                #env.step(action)

                if 'trial_1.0_34500' in train_model and total_reward > -8.5:
                    action = 3

                reward = 0
                skip_frames_adjusted = SKIP_FRAMES + 1
                if 'trial_0.0_60000' in train_model:
                    skip_frames_adjusted += 1

                for _ in range(skip_frames_adjusted):
                    next_state, r, terminated, truncated, info = env.step(action)
                    reward += r
                    if terminated or truncated:
                        break

                time_frame_counter +=1

                next_state = process_state_image(next_state)
                done = terminated

                init_state = next_state

                total_reward += reward
                rewards.append(reward)

                if time_frame_counter > 200:
                    print('reset')
                    reset_count += 1
                    break

                if truncated:
                    total_reward = 0
                    rewards = []

                state_frame_stack_queue.append(next_state)

                if done:
                    e += 1

                    log(policy_id, e, play_episodes, total_reward)

                    break

            all_rewards[policy_id].append(total_reward)

    print('\n\n*** FINAL ****\n')
    print('RISK\tMEAN\tVAR\tMIN\tMAX')
    for policy_id in policies:
        print('{}\t{}\t{}\t{}\t{}'.format(policy_id, round(np.mean(all_rewards[policy_id]), 2), round(np.var(all_rewards[policy_id]), 2),round(np.min(all_rewards[policy_id]), 2),round(np.max(all_rewards[policy_id]), 2)))

