import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)

import argparse
from collections import deque
from CarRacingDQNAgent import CarRacingDQNAgent
from car_river_crossing import CarRiverCrossing

from common_functions import generate_state_frame_stack_from_queue
from common_functions import process_state_image
import cv2

from datetime import datetime

STARTING_EPISODE              = 1
ENDING_EPISODE                = 100000
TRAINING_BATCH_SIZE           = 64
TRAINING_MODEL_FREQUENCY      = 4
UPDATE_TARGET_MODEL_FREQUENCY = 1
RESETS_BEFORE_FIXED_POLICY    = 2
SKIP_FRAMES                   = 3
MAXIMUM_FRAMES                = 1000
RESULT_FOLDER                 = 'save_fixed_model'
SAVE_IMG = False

def log(txt, lamb, gamma):
    with open('./{}/result_2_train_{}_{}.log'.format(RESULT_FOLDER, lamb, gamma), 'a') as f:
        f.write(txt + '\n')
    print(txt)

def save_image(img, frame, action):
    if SAVE_IMG:
        cv2.imwrite("./img_train/frame_{}.png".format(frame), img)
        with open('./img_train/result.log', 'a') as f:
            f.write('frame {}, action {}\n'.format(frame, action))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training a DQN agent to play CarRacing.')
    parser.add_argument('-m', '--model', help='Specify the last trained model path if you want to continue training after it.')
    parser.add_argument('-s', '--start', type=int, help='The starting episode, default to 1.')
    parser.add_argument('-e', '--end', type=int, help='The ending episode, default to 1000.')
    parser.add_argument('-p', '--epsilon', type=float, default=0.1, help='The starting epsilon of the agent, default to 1.0.')
    parser.add_argument('-l', '--lamb', type=float, default=0.0, help='The risk param, default to 0.0.')
    parser.add_argument('-g', '--gamma', type=float, default=0.99, help='The discount factor, default to 0.99.')
    parser.add_argument('-r', '--render', type=bool, default=False, help='Render while training, default to False.')
    parser.add_argument('-f', '--frequency', type=int, default=100, help='Save training frequency, defautl to 1000.')
    args = parser.parse_args()

    print('Training with risk factor', args.lamb)
    print('Training with discount factor', args.gamma)

    play_area = 300
    zoom = 1.8

    if args.render:
        env = CarRiverCrossing(render_mode='human', play_field_area=play_area, zoom=zoom)
    else:
        env = CarRiverCrossing(play_field_area=play_area, zoom=zoom)

    bias_initializer = -10.0

    agent = CarRacingDQNAgent(epsilon=args.epsilon, lamb=args.lamb, gamma=args.gamma, bias_initializer=bias_initializer)
    if args.model:
        agent.load(args.model)
    if args.start:
        STARTING_EPISODE = args.start
    if args.end:
        ENDING_EPISODE = args.end

    policy_id = 2

    saved_already = False

    for e in range(STARTING_EPISODE, ENDING_EPISODE+1):

        time_frame_counter = 1
        time_frame_counter_without_reset = 1

        current_state, info = env.reset()
        state_img = current_state
        current_state = process_state_image(current_state)
        state_frame_stack_queue = deque([current_state] * agent.frame_stack_num, maxlen=agent.frame_stack_num)


        total_reward = 0
        done = False

        policy_id += 1
        policy_id = 3

        if policy_id > 6:
            policy_id = 3

        while True:
            current_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)

            if policy_id <= 5:
                action = agent.get_fixed_policy(policy_id, time_frame_counter_without_reset)
            else:
                action = agent.act(current_state_frame_stack)
            save_image(state_img, time_frame_counter, action)
            model_value = agent.get_value(current_state_frame_stack, action)

            #log('Frame {}, Value {}'.format(time_frame_counter_without_reset, model_value), args.lamb, args.gamma)

            reward = 0
            for _ in range(SKIP_FRAMES + 1):
                next_state, r, terminated, truncated, info = env.step(action)
                reward += r
                if terminated or truncated:
                    break
            state_img = next_state
            next_state = process_state_image(next_state)
            done = terminated

            total_reward += reward
            time_frame_counter += 1

            if truncated:
                time_frame_counter_without_reset = 0
            time_frame_counter_without_reset += 1

            state_frame_stack_queue.append(next_state)
            next_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)

            agent.memorize(current_state_frame_stack, action, reward, next_state_frame_stack, done)

            current_state = next_state

            # finishes no fixed policy
            if policy_id > 5 and time_frame_counter > MAXIMUM_FRAMES:
                done = True
                truncated = True

            if done:
                policy_type = 'FIXED_{}'.format(policy_id)
                if policy_id > 5:
                    if truncated:
                        policy_type = 'AGENT_TRUNC'
                    else:
                        policy_type = 'AGENT_DONE'
                        agent.save('./{}/trial_{}_{}_{}.h5'.format(RESULT_FOLDER, args.lamb, args.gamma, e))
                        saved_already = True
                #log('{} - Episode: {}/{}, Total Frames: {}, Tiles Visited: {}, Total Rewards: {}, Epsilon: {:.2}, Policy: {}'.format(datetime.now(), e, ENDING_EPISODE, time_frame_counter, env.tile_visited_count, total_reward, float(agent.epsilon), policy_type), args.lamb, args.gamma)

                agent.replay_batch(len(agent.memory))
                agent.flush_memory()
                break

        if e % UPDATE_TARGET_MODEL_FREQUENCY == 0:
            agent.update_target_model()

        if e % args.frequency == 0:
            if saved_already:
                saved_already = False
            else:
                saved_already = False
                #agent.save('./{}/trial_{}_{}_{}.h5'.format(RESULT_FOLDER, args.lamb, args.gamma, e))

        #run once when save img
        if SAVE_IMG:
            break

    env.close()
