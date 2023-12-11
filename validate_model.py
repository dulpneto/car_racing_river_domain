
import numpy as np
import math

from collections import deque
from CarRacingDQNAgent import CarRacingDQNAgent
from car_river_crossing import CarRiverCrossing
from common_functions import generate_state_frame_stack_from_queue
from common_functions import process_state_image

SKIP_FRAMES = 3
RENDER = False
def log(txt):
    with open('./result.log', 'a') as f:
        f.write(txt + '\n')
    print(txt)

if __name__ == '__main__':

    play_area = 300
    zoom = 1.8

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

        train_model = './save/trial_{}_10000.h5'.format(lamb)

        # Set epsilon to 0 to ensure all actions are instructed by the agent
        agent = CarRacingDQNAgent(epsilon=0, lamb=lamb)
        agent.load(train_model)

        play_episodes = 100

        for e in range(play_episodes):
            init_state, info = env.reset()
            init_state = process_state_image(init_state)

            total_reward = 0
            punishment_counter = 0
            state_frame_stack_queue = deque([init_state] * agent.frame_stack_num, maxlen=agent.frame_stack_num)
            time_frame_counter = 1

            rewards = []

            while True:
                current_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)
                action = agent.act(current_state_frame_stack)
                reward = 0
                for _ in range(SKIP_FRAMES + 1):
                    next_state, r, terminated, truncated, info = env.step(action)
                    reward += r
                    if terminated or truncated:
                        break

                next_state = process_state_image(next_state)
                done = terminated

                init_state = next_state

                total_reward += reward
                rewards.append(reward)

                state_frame_stack_queue.append(next_state)

                # print('{} REWARD {} TILES {}'.format(environment.frames, reward, environment.tiles_visited))

                if done:

                    log('Risk: {}, Episode: {}/{}, Total Rewards: {}'.format(lamb, e + 1, play_episodes, total_reward))

                    for l2 in range(-100, 101, 25):
                        lamb2 = l2 / 100
                        all_rewards_utility[lamb][lamb2] = []

                        t = 0
                        try:
                            for i in range(len(rewards)-1, -1, -1):
                                u = np.sign(lamb2) * math.exp(lamb2 * rewards[i])
                                t = u + gamma * t

                            all_rewards_utility[lamb][lamb2].append(t)
                        except:
                            log("EXP error for {} {}".format(lamb, lamb2))

                    break

            all_rewards[lamb].append(total_reward)

    log('\n\n*** FINAL ****\n')
    log('RISK\tMEAN\tVAR')
    for l in range(-100, 101, 25):
        lamb = l/100
        log('{}\t{}\t{}'.format(lamb, round(np.mean(all_rewards[lamb]), 2), round(np.var(all_rewards[lamb]), 2)))

    log('\n\n*** UTILITY ****\n')

    v = 'RISK'.format(lamb)
    for l in range(-100, 101, 25):
        lamb = l / 100
        v = v+'\t{}'.format(lamb)
    log(v)
    for l in range(-100, 101, 25):
        lamb = l / 100
        v = '{}'.format(lamb)
        for l2 in range(-100, 101, 25):
            lamb2 = l2 / 100
            v = v + '\t{}'.format(round(np.mean(all_rewards_utility[lamb][lamb2]), 2))
        log(v)
