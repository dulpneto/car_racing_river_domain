import argparse
import cv2
from collections import deque
from CarRacingDQNAgent import CarRacingDQNAgent
from car_river_crossing import CarRiverCrossing
from common_functions import generate_state_frame_stack_from_queue
from common_functions import process_state_image

SKIP_FRAMES                   = 3
SAVE_IMG = False

def log(txt, lamb):
    #with open('./save_fixed_model_2/run_position_{}.log'.format(lamb), 'a') as f:
    #    f.write(txt + '\n')
    print(txt)

def log2(txt, lamb):
    #with open('./save_fixed_model_2/run_reward_{}.log'.format(lamb), 'a') as f:
    #    f.write(txt + '\n')
    print(txt)

def save_image(img, frame, train, model):
    if SAVE_IMG:
        cv2.imwrite("./img_run/frame_{}.png".format(frame), img)
        with open('./img_run/result.log', 'a') as f:
            f.write('frame {}, train {}, model {}\n'.format(frame, train, model))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Play CarRacing by the trained model.')
    parser.add_argument('-m', '--model', required=True, help='The `.h5` file of the trained model.')
    parser.add_argument('-e', '--episodes', type=int, default=1, help='The number of episodes should the model plays.')
    parser.add_argument('-l', '--lamb', type=float, default=0.0, help='The risk param, default to 0.0.')
    parser.add_argument('-r', '--render', type=bool, default=True, help='Render while training, default to True.')
    args = parser.parse_args()
    train_model = args.model
    play_episodes = args.episodes
    lamb = args.lamb

    play_area = 300
    zoom = 1.8

    if args.render:
        env = CarRiverCrossing(render_mode='human', play_field_area=play_area, zoom=zoom)
        #env = CarRiverCrossing(play_field_area=play_area, zoom=zoom)
    else:
        env = CarRiverCrossing(play_field_area=play_area, zoom=zoom)

    # Set epsilon to 0 to ensure all actions are instructed by the agent
    agent = CarRacingDQNAgent(epsilon=0, lamb=args.lamb)
    agent.load(train_model)

    for e in range(24, play_episodes):
        current_state, info = env.reset()
        state_img = current_state
        current_state = process_state_image(current_state)
        state_frame_stack_queue = deque([current_state] * agent.frame_stack_num, maxlen=agent.frame_stack_num)

        total_reward = 0
        time_frame_counter = 1
        
        while True:
            current_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)
            action = agent.act(current_state_frame_stack)

            policy_id = 1
            action_fixed = agent.get_fixed_policy(policy_id, time_frame_counter)

            save_image(state_img, time_frame_counter, action_fixed, action)
            print('FRAME: {}, FIXED: {}, MODEL:{}'.format(time_frame_counter, action_fixed, action))

            reward = 0
            for _ in range(SKIP_FRAMES + 1):
                next_state, r, terminated, truncated, info = env.step(action)
                x, y = info['position']
                log('{},{},{},{}'.format(e, train_model, x,y), lamb)
                reward += r
                if terminated or truncated:
                    break
            state_img = next_state
            next_state = process_state_image(next_state)
            done = terminated

            init_state = next_state

            total_reward += reward

            state_frame_stack_queue.append(next_state)

            time_frame_counter += 1

            if done:
                print('Episode: {}/{}, Total Frames: {}, Tiles Visited: {}, Total Rewards: {}'.format(e+1, play_episodes,
                                                                                                                      time_frame_counter,
                                                                                                                      env.tile_visited_count,
                                                                                                                      total_reward))
                log2('{},{},{}'.format(e,train_model, total_reward), lamb)
                break
