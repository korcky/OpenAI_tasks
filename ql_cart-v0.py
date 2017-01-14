import gym

import numpy as np

import matplotlib.pyplot as plt

# Implementation of Q-learning
class QAgent(object):

    def __init__(self, sampling, max_val, num_actions,
                 alpha=0.3, gamma=0.7,
                 random_action_rate=0.5,
                 random_action_decay_rate=0.8):
        """
        Q-learning agent

        :param sampling: <array> of sampling rate for each variable in observation space
        :param max_val: <array> of maximum values or each variable in observation space
        :param num_actions: <int> of possible actions
        :param alpha: <float> learning rate (default: 0.3)
        :param gamma: <float> discount rate (default: 0.7)
        :param random_action_rate: <float> random action rate (default: 0.5)
        :param random_action_decay_rate: <float> decay rate for random action rate (default: 0.99)
        """

        # Input check
        if len(sampling) != len(max_val):
            raise ValueError('length of sampling must correspond to length of borders')

        # Initialization of agent variables
        self.Q = np.random.uniform(low=-1, high=1,
                                   size=(tuple(sampling) + tuple([num_actions])))
        self.alpha = alpha
        self.gamma = gamma
        self.random_action_rate = random_action_rate
        self.random_action_decay_rate = random_action_decay_rate
        self.num_actions = num_actions
        self.state = 0
        self.action = 0
        self.borders = []

        # Sampling of space
        for i in range(len(sampling)):
            self.borders.append([])
            if sampling[i] == 2:
                self.borders[i].append(0)
                continue
            part = 2 * max_val[i] / (sampling[i] - 2)
            self.borders[i].append(-max_val[i])
            for j in range(sampling[i] - 3):
                self.borders[i].append(self.borders[i][j] + part)
            self.borders[i].append(max_val[i])

    def sampling_state(self, observation):
        """
        Sampling of observation

        :param observation: <array> of observation values
        :return: <tuple> state corresponded to observation
        """

        state = []
        for i in range(len(observation)):
            for j in range(len(self.borders[i])):

                if observation[i] < self.borders[i][j]:
                    state.append(j)
                    break

                if (j + 1) == len(self.borders[i]):
                    state.append(j + 1)

        return tuple(state)

    def init_state(self, observation):
        """
        Initialize first state in episode

        :param observation: <array> of observation values
        :return: <int> first action
        """

        self.state = self.sampling_state(observation)
        self.action = self.Q[self.state].argmax()
        return self.action

    def act(self, observation, reward, done):
        """
        Action corresponding to the current observation

        :param observation: observation: <array> of observation values
        :param reward: <int> reward for previous action
        :param done: <bool> is environment finished
        """

        new_sate = self.sampling_state(observation)

        # Whether the action is selected randomly
        choose_random_action = self.random_action_rate > np.random.uniform(0, 1)
        if choose_random_action:
            new_action = np.random.randint(0, self.num_actions - 1)
        else:
            new_action = self.Q[new_sate].argmax()

        # Updating Q matrix
        self.Q[self.state][self.action] = (1 - alpha) * self.Q[self.state][self.action] + \
                                          alpha * (reward + gamma * self.Q[new_sate][new_action])

        self.state = new_sate
        self.action = new_action
        return self.action

if __name__ == '__main__':
    # cart sampling parameters
    cart_position_sr = 10  # sampling rate for position of cart
    cart_velocity_sr = 10  # sampling rate for velocity of cart
    pole_angle_sr = 10  # sampling rate for angle of pole
    pole_rotation_rate_sr = 10  # sampling rate for rotation rate of pole
    sampling = [cart_position_sr, cart_velocity_sr, pole_angle_sr, pole_rotation_rate_sr]

    # Q-learning parameters
    alpha = 0.2
    gamma = 1
    random_action_rate = 0.5
    random_action_decay_rate = 0.99

    # Env parameters
    goal_steps = 195
    max_steps = 200
    average_iterations = 100
    num_episodes = 50000

    last_time_steps = np.ndarray(0)

    env = gym.make('CartPole-v0')
    agent = QAgent(alpha=alpha, gamma=gamma,
                   sampling=sampling, num_actions=2,
                   max_val=[2.4, 2, 0.20943951, 2.5],
                   random_action_rate=random_action_rate,
                   random_action_decay_rate=random_action_decay_rate)

    x, y = [], []
    for episode in range(num_episodes):

        observation = env.reset()
        action = agent.init_state(observation)
        #action = agent.act(observation, 1, False)

        for step in range(max_steps):
            observation, reward, done, info = env.step(action)
            if done:
                reward = -1000
            action = agent.act(observation, reward, done)
            if done:
                agent.random_action_rate *= agent.random_action_decay_rate
                last_time_steps = np.append(last_time_steps, [int(step + 1)])
                if len(last_time_steps) > average_iterations:
                    last_time_steps = np.delete(last_time_steps, 0)
                y.append(step + 1)
                x.append(episode)
                break

        if last_time_steps.mean() > goal_steps:
            print('Episodes before solve: ', episode + 1)
            print('Best 100-episode performance {} +- {}'.format(last_time_steps.max(),
                                                            last_time_steps.std()))
            break


    plt.plot(x, y)
    plt.show()
