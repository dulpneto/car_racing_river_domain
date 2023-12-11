
import numpy as np
import math

from collections import deque
from CarRacingDQNAgent import CarRacingDQNAgent
from car_river_crossing import CarRiverCrossing
from common_functions import generate_state_frame_stack_from_queue
from common_functions import process_state_image

SKIP_FRAMES = 3
RENDER = False
TRUNCATED_THRESHOLD = 5
PLAY_EPISODES = 10


def log(txt):
    with open('./result_size_3.log', 'a') as f:
        f.write(txt + '\n')
    print(txt)


if __name__ == '__main__':

    log('size\trisk\tcompleted\ttotal')

    zoom = 1.8
    play_area = 300

    while True:

        play_area += 50

        if play_area > 1000:
            break

        if RENDER:
            env = CarRiverCrossing(render_mode='human', play_field_area=play_area, zoom=zoom)
        else:
            env = CarRiverCrossing(play_field_area=play_area, zoom=zoom)

        all_rewards = {}
        all_rewards_utility = {}
        gamma = 0.95

        for l in range(-100, 101, 25):
            lamb = l/100

            all_rewards[lamb] = []
            all_rewards_utility[lamb] = {}

            train_model = './save_200/trial_{}_10000.h5'.format(lamb)

            # Set epsilon to 0 to ensure all actions are instructed by the agent
            agent = CarRacingDQNAgent(epsilon=0, lamb=lamb)
            agent.load(train_model)

            task_completed = 0
            for e in range(PLAY_EPISODES):
                init_state, info = env.reset()
                init_state = process_state_image(init_state)

                total_reward = 0
                punishment_counter = 0
                state_frame_stack_queue = deque([init_state] * agent.frame_stack_num, maxlen=agent.frame_stack_num)
                time_frame_counter = 1

                rewards = []

                truncated_counter = 0

                frames = 0

                while True:
                    current_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)
                    action = agent.act(current_state_frame_stack)
                    reward = 0
                    for _ in range(SKIP_FRAMES + 1):
                        next_state, r, terminated, truncated, info = env.step(action)
                        reward += r
                        if terminated or truncated:
                            break

                    if truncated:
                        truncated_counter += 1

                    next_state = process_state_image(next_state)
                    done = terminated

                    init_state = next_state

                    total_reward += reward
                    rewards.append(reward)

                    state_frame_stack_queue.append(next_state)

                    frames += 1

                    if frames > 100:
                        print('frames')
                        break


                    # print('{} REWARD {} TILES {}'.format(environment.frames, reward, environment.tiles_visited))

                    if done or truncated_counter >= TRUNCATED_THRESHOLD:
                        break

                if truncated_counter < TRUNCATED_THRESHOLD:
                    task_completed += 1

            log('{}\t{}\t{}\t{}'.format(play_area, lamb, task_completed, PLAY_EPISODES))


