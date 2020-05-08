import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #supress tensorflow messages

import gym #OpenAI Gym
import retro #Gym Retro
import numpy as np #NumPy
from time import time #calculate runtime
from collections import deque #needed for replay memory
from random import sample #used to get random minibacth

#TensorFlow 2.0
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Lambda, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution() #Disable Eager, IMPORTANT!

from utils import convert_frames, Now, get_latest_file #My utilities, useful re-usable functions 

class DDDQNAgent(object):
    def __init__(self, game, num_actions, batch_size=32, learn_every=10, update_every=10000, alpha=1e-4,
                 gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.9999, memory_size=100000):
        #retro environment
        self.game = game #game rom name
        self.num_actions = num_actions #number of possible actions for env
        self.env = self.build_env() #gym environment
        self.state_shape = self.env.observation_space.shape #env state dims
        self.state = self.reset() #initialize state
        #training
        self.batch_size=batch_size #batch size
        self.steps = 0 #number of steps ran
        self.learn_every = learn_every #interval of steps to fit model
        self.update_every = update_every #interval of steps to update target model
        self.alpha = alpha #learning rate
        self.gamma = gamma #discount factor
        self.epsilon = epsilon #exploration factor
        self.epsilon_min = epsilon_min #minimum exploration probability
        self.epsilon_decay = epsilon_decay #exponential decay rate of epsilon
        #memory
        self.memory_size = memory_size #replay memory size
        self.memory = deque(maxlen=memory_size) #replay memory
        self.log = [] #stores information from training
        #models
        self.q_eval = self.build_network() #Q eval model
        self.q_target = self.build_network() #Q target model
        
    def build_env(self, time_limit=None, downsampleRatio=2, numStack=4):
    #Build the gym retro environment.
        env = gym.make(self.game)
        return env

    def build_network(self):
    #Build the Dueling DQN Network
        X_input = Input(self.state_shape, name='input')
        X = Dense(512, activation='relu', kernel_initializer='he_uniform')(X_input)
        X = Dense(256, activation='relu', kernel_initializer='he_uniform')(X)
        #value layer
        V = Dense(1, activation='linear', name='V')(X) #V(S)
        #advantage layer
        A = Dense(self.num_actions, activation='linear', name='Ai')(X) #A(s,a)
        A = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), name='Ao')(A) #A(s,a)
        #Q layer (V + A)
        Q = Add(name='Q')([V, A]) #Q(s,a)
        Q_model = Model(inputs=[X_input], outputs=[Q], name='qvalue')
        Q_model.compile(loss='mse', optimizer=Adam(lr=self.alpha))
        return Q_model
    
    def update_target(self):
    #Update the q_target model to weights of q_eval model.
        #don't learn until we have at least one batch and update interval steps reached
        if len(self.memory) < self.batch_size or self.steps % self.update_every != 0: return
        self.q_target.set_weights(self.q_eval.get_weights())
    
    def remember(self, action, next_state, reward, done):
    #Store data in memory and update current state.
        self.memory.append([self.state, action, next_state, reward, done])
        self.state = next_state

    def choose_action(self, training):
    #Predict next action based on current state and decay epsilon.
        if training: #when training allow random exploration
            if np.random.random() < self.epsilon: #get random action
                action = np.random.randint(self.num_actions)
            else: #predict best actions
                action = np.argmax(self.q_eval.predict(self.state)[0])
            #decay epsilon, if epsilon falls below min then set to min
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            elif self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min
                print('Epsilon mininum of {} reached'.format(self.epsilon_min))
        else: #if not training then always get best action
            action = np.argmax(self.q_eval.predict(self.state)[0])
        return action
                
    def learn(self):
        #don't learn until we have at least one batch and learn interval steps reached
        if len(self.memory) < self.batch_size or self.steps % self.learn_every != 0: return
        #sample memory for a minibatch
        mini_batch = sample(self.memory, self.batch_size)
        #separate minibatch into elements
        state, action, next_state, reward, done = [np.squeeze(i) for i in zip(*mini_batch)]
        Q = self.q_eval.predict(state) #get Q values for starting states
        Q_next = self.q_eval.predict(next_state) #get Q values for ending states
        Q_target = self.q_target.predict(next_state) #get Q values from target model
        #update q values
        for i in range(self.batch_size):
            if done[i]:
                Q[i][action[i]] = 0.0 #terminal state
            else:
                a = np.argmax(Q_next[i]) ## a'_max = argmax(Q(s',a'))
                Q[i][action[i]] = reward[i] + self.gamma * Q_target[i][a] #Q_max = Q_target(s',a'_max)
        #fit network on batch_size = minibatch_size
        self.q_eval.fit(state, Q, batch_size=self.batch_size, verbose=0, shuffle=False)
    
    def load(self, directory, qeval_name=None, qtarget_name=None):
    #Load the actor and critic weights.
        print('Loading models ...', end=' ')
        #if no names supplied try to load most recent
        if qeval_name is not None and qtarget_name is not None:
            qeval_path = os.path.join(directory, qeval_name)
            qtarget_path = os.path.join(directory, qtarget_name)
        elif qeval_name is None and qtarget_name is None:
            qeval_path = get_latest_file(directory + '/*QEval.h5')
            qtarget_path = get_latest_file(directory + '/*QTarget.h5')
        self.q_eval.load_weights(qeval_path)
        self.q_target.load_weights(qtarget_path)
        print('Done. Models loaded from {}'.format(directory))
        print('Loaded Q_Eval model {}'.format(qeval_path))
        print('Loaded Q_Target model {}'.format(qtarget_path))

    def save(self, directory, fileName):
    #Save the actor and critic weights.
        print('Saving models ...', end=' ')
        if not os.path.exists(directory): os.makedirs(directory)
        qeval_name = fileName + '_QEval.h5'
        qtarget_name = fileName + '_QTarget.h5'
        self.q_eval.save_weights(os.path.join(directory, qeval_name))
        self.q_target.save_weights(os.path.join(directory, qtarget_name))
        print('Done. Saved to {}'.format(os.path.abspath(directory)))
        
    def save_log(self, directory, fileName, clear=False):
    #Save the information currently stored in log list.
        print('Saving log ...', end=' ')
        if not os.path.exists(directory): os.makedirs(directory)
        f = open(os.path.join(directory, fileName + '.csv'), 'w')
        for line in self.log:
            f.write(str(line)[1:-1].replace('None', '') + '\n')
        f.close()
        if clear: self.log = []
        print('Done. Saved to {}'.format(os.path.abspath(directory)))

    def reset(self):
    #Reset environment and return expanded state.
        self.state = np.expand_dims(self.env.reset(), axis=0)

    def close(self):
    #Close the environment.
        self.env.close()

    def step(self, action):
    #Run one step for given action and return data.
        observation, reward, done, info = self.env.step(action)
        observation = np.expand_dims(observation, axis=0)
        return observation, reward, done, info
    
    def run(self, num_episodes=100, render=False, checkpoint=False, cp_render=False, cp_interval=None, otype='AVI'):
    #Run num_episodes number of episodes and train the q model after each learn_every number of steps. The target
    #model is updated every update_every number of steps. If render is true then render each episode to monitor.
    #If checkpoint is true then save model weights and log and evaluate the model and convert and save the frames
    #every cp_interval number of of episodes. The evaluation is rendered if cp_render is true.
        printSTR = 'Episode: {}/{} | Score: {:.4f} | AVG 50: {:.4f} | Elapsed Time: {} mins'
        start_time = time()
        scores = []
        self.reset()
        for e in range(1, num_episodes + 1):
            score = 0
            while True:
                action = self.choose_action(training=True) #predict action
                next_state, reward, done, info = self.step(action) #perform action
                score += reward #cumulative score for episode
                reward = np.clip(reward, -1.0, 1.0).item() #clip reward to range [-1.0, 1.0]
                self.remember(action, next_state, reward, done) #store results
                self.steps += 1 #increment steps
                self.update_target() #update target network (update_every)
                self.learn() #fit q model (learn_every)
                if render: self.env.render()
                if done:
                    scores.append(score) #store scores for all epsisodes
                    self.reset()
                    break
            elapsed_time = round((time() - start_time)/60, 2)        
            print(printSTR.format(e, num_episodes, round(score, 4), np.average(scores[-50:]), elapsed_time))
            if checkpoint and (e % cp_interval) == 0:
                eval_score, frames = self.evaluate(render=cp_render)
                print('EVALUATION: {}'.format(round(eval_score, 4)))
                self.log.append([e, score, np.average(scores[-50:]), elapsed_time, eval_score])
                fileName = 'DDDQN_{}_{}_{}'.format(e, self.game, Now(separate=False))
                self.save('models', fileName)
                self.save_log('logs', fileName)
                convert_frames(frames, 'renders', fileName, otype=otype)
            elif checkpoint:
                self.log.append([e, score, np.average(scores[-50:]), elapsed_time, None])
        elapsed_time = round((time() - start_time)/60, 2)
        print('Finished training {} episodes in {} minutes.'.format(num_episodes, (time() - start_time)/60))
        
    def evaluate(self, render=False):
    #Run an episode and return the score and frames.
        frames = []
        score = 0
        self.reset()
        while True:
            action = self.choose_action(training=False) #get best action
            observation, reward, done, info = self.step(action) #perform action
            self.state = observation #update current state
            score += reward #cumulative score for episode
            if render: self.env.render()
            frames.append(self.env.render(mode='rgb_array'))
            if done:
                self.reset()
                break
        return score, frames
            
    def play_episode(self, render=False, render_and_save=False, otype='AVI'):
    #Run one episode. If render is true then render each episode to monitor.
    #If render_and_save is true then save frames and convert to GIF image or AVI movie.
    #The reward for the epsiode is returned.
        frames = []
        score = 0
        self.reset()
        while True:
            action = self.choose_action(training=False) #get best action
            observation, reward, done, info = self.step(action) #perform action
            self.state = observation #update current state
            score += reward #cumulative score for episode
            if render: self.env.render()
            if render_and_save: frames.append(self.env.render(mode='rgb_array'))
            if done:
                print('Finished! Score: {}'.format(score))
                self.reset()
                break
        if render_and_save:
            fileName = 'DDDQN_PLAY_{}_{}'.format(self.game, Now(separate=False))
            convert_frames(frames, 'renders', fileName, otype=otype)
        return score
        
if __name__ == '__main__':
    
    dddqn = DDDQNAgent('LunarLander-v2', 4, epsilon_decay=0.99999, batch_size=128)
    dddqn.q_eval.summary()
    dddqn.q_target.summary()
    
    #train model
    dddqn.run(num_episodes=4000, checkpoint=True, cp_render=True, cp_interval=100)
    
    #load model
    dddqn.load('models', 'DDDQN_3800_LunarLander_QEval.h5', 'DDDQN_3800_LunarLander_QTarget.h5')
    
    #play games
    dddqn.play_episode(render=True, render_and_save=False, otype='GIF')