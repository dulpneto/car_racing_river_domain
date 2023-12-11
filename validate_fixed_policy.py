
import numpy as np
import math

from collections import deque
from CarRacingDQNAgent import CarRacingDQNAgent
from car_river_crossing import CarRiverCrossing
from common_functions import process_state_image

SKIP_FRAMES = 3
RENDER = False
def log(txt):
    with open('./result_fixed_policy.log', 'a') as f:
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
    gammas = [0.95, 0.98, 0.99, 0.999, 1.0]

    for gamma in gammas:
        all_rewards_utility[gamma] = {}

    for policy_id in range(1, 6):

        all_rewards[policy_id] = []

        for gamma in gammas:
            all_rewards_utility[gamma][policy_id] = {}

        # Set epsilon to 0 to ensure all actions are instructed by the agent
        agent = CarRacingDQNAgent(epsilon=0, lamb=0.0)

        play_episodes = 10

        for e in range(play_episodes):
            init_state, info = env.reset()
            init_state = process_state_image(init_state)

            total_reward = 0
            punishment_counter = 0
            state_frame_stack_queue = deque([init_state] * agent.frame_stack_num, maxlen=agent.frame_stack_num)
            time_frame_counter = 1

            rewards = []

            time_frame_counter_without_reset = 1

            while True:
                action = CarRacingDQNAgent.get_fixed_policy(policy_id=policy_id, time_frame_counter_without_reset=time_frame_counter_without_reset)
                reward = 0
                for _ in range(SKIP_FRAMES + 1):
                    next_state, r, terminated, truncated, info = env.step(action)
                    reward += r
                    if terminated or truncated:
                        break

                next_state = process_state_image(next_state)
                done = terminated

                if truncated:
                    time_frame_counter_without_reset = 0

                time_frame_counter_without_reset += 1

                init_state = next_state

                total_reward += reward
                rewards.append(reward)

                state_frame_stack_queue.append(next_state)

                # print('{} REWARD {} TILES {}'.format(environment.frames, reward, environment.tiles_visited))

                if done:

                    log('Policy: {}, Episode: {}/{}, Total Rewards: {}'.format(policy_id, e + 1, play_episodes, total_reward))

                    for l2 in range(-10, 11):
                        lamb2 = l2
                        for gamma in gammas:
                            all_rewards_utility[gamma][policy_id][lamb2] = []

                            t = 0
                            for i in range(len(rewards)-1, -1, -1):
                                u = rewards[i]
                                if lamb2 != 0:
                                    u = np.sign(lamb2) * math.exp(lamb2 * rewards[i])
                                t = u + gamma * t

                            all_rewards_utility[gamma][policy_id][lamb2].append(t)

                    break

            all_rewards[policy_id].append(total_reward)

    log('\n\n*** FINAL ****\n')
    log('POLICY\tMEAN\tVAR')
    for policy_id in range(1, 6):
        log('{}\t{}\t{}'.format(policy_id, round(np.mean(all_rewards[policy_id]), 2), round(np.var(all_rewards[policy_id]), 2)))

    log('\n\n*** UTILITY ****\n')

    for gamma in gammas:

        log('\n\n*** UTILITY WITH DISCOUNT {} ****\n'.format(gamma))

        v = ''
        for l in range(-10, 11):
            lamb = l
            v = v + '\t{}'.format(lamb)
        log(v)
        for policy_id in range(1, 6):
            v = '{}'.format(policy_id)
            for l2 in range(-10, 11):
                lamb2 = l2
                v = v + '\t{}'.format(round(np.mean(all_rewards_utility[gamma][policy_id][lamb2]), 2))
            log(v)
