#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 12:15:34 2019

@author: ramakrishnareddych
"""

import tensorflow as tf
import gym
import numpy as np

import matplotlib.pyplot as plt

from ipywidgets import widgets
from IPython.display import display


from matplotlib import animation
from JSAnimation.IPython_display import display_animation



learning_rate =0.01
gamma = 0.95
num_episodes=5
env = gym.make('CartPole-v0')
env = env.unwrapped
state_size = 4
action_size = env.action_space.n

max_episodes = 300

def display_frames_as_gif(frames, filename_gif = None):
    """
    Displays a list of frames as a gif, with controls
    """
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    if filename_gif: 
        anim.save(filename_gif, writer = 'imagemagick', fps=20)
    display(display_animation(anim, default_mode='loop'))
    
def discount_and_normalise_rewards(episode_rewards):
        discounted_episode_rewards = np.zeros_like(episode_rewards)
        cumulative = 0
        for t in reversed(range(len(episode_rewards))):
            cumulative = cumulative * gamma + episode_rewards[t]
            discounted_episode_rewards[t] = cumulative

        mean = np.mean(discounted_episode_rewards)
        std = np.std(discounted_episode_rewards)
        discounted_episode_rewards = (discounted_episode_rewards - mean) / (std)
        
        return discounted_episode_rewards

# Q learning
    
with tf.name_scope('inputs'):
        state = tf.placeholder(tf.float32, shape=(None, state_size), name="input_")
        actions = tf.placeholder(tf.float32, shape=(None, action_size), name="acctions")
        discounted_episode_rewards_ = tf.placeholder(tf.float32, [None, ], name="discounted_episode_rewards")
       
        
        mean_reward_ = tf.placeholder(tf.float32, name = "mean_reward")
    
    
        with tf.name_scope('fc1'):
            fc1 = tf.contrib.layers.fully_connected(inputs = input, num_outputs = 10,
                                                    activation_fn = tf.nn.relu,
                                                    weights_initializer = tf.contrib.layers.xavier_initializer())
        with tf.name_scope('fc2'):
            fc2 = tf.contrib.layers.fully_connected(inputs = fc1, num_outputs = action_size,
                                                    activation_fn = tf.nn.relu,
                                                    weights_initializer = tf.contrib.layers.xavier_initializer())
        with tf.name_scope('fc3'):
            fc3 = tf.contrib.layers.fully_connected(inputs = fc2, num_outputs = action_size,
                                                    activation_fn = None,
                                                    weights_initializer = tf.contrib.layers.xavier_initializer())
        
        with tf.name_scope("softmax"):
            action_distribution = tf.nn.softmax(fc3)
        diffs = discounted_episode_rewards_ - action_distribution
            
        with tf.name_scope("loss"):
            loss = tf.nn.l2_loss(action_distribution - 0.01*action_distribution)
        with tf.name_scope("train"):
            train_opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)
            
            
allRewards = []
total_rewards = 0
maximumRewardRecorded = 0
episode = 0
is_rendering = True
episode_states, episode_actions, episode_rewards = [],[],[]

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for episode in range(max_episodes):
        episode_rewards_sum = 0
        state = env.reset()
        if is_rendering:
            frames = []
        while True:
             if is_rendering:
                frame = env.render(mode = 'rgb_array')
                frames.append(frame)
       

                action_probability_distribution = sess.run(action_distribution,
                                                          feed_dict = {input: state.reshape([1,4])})
            
                action = np.random.choice(range(action_probability_distribution.shape[1]), p = action_probability_distribution.ravel())
            
                new_state, reward, done, info = env.step(action)
            
                episode_states.append(state)
            
            
                action_ = np.zeros(action_size)
                action_[action] = 1
            
                episode_actions.append(action_)
                episode_rewards.append(reward)
            
                if done:
                    episode_rewards_sum = np.sum(episode_rewards)
                
                    allRewards.append(episode_rewards_sum)
                
                    total_rewards = np.sum(allRewards)
                
                    mean_reward = np.divide(total_rewards, episode+1)
                
                    maximumRewardRecorded = np.argmax(allRewards)
                
                    print("==========================================")
                    print("Episode: ", episode)
                    print("Reward: ", episode_rewards_sum)
                    print("Mean Reward", mean_reward)
                    #print("total Reward", total_rewards)
                    #print("Max reward so far: ", maximumRewardRecorded)
                    plt.plot(allRewards)
            
                    plt.title("DQN Cart Pole")
                    plt.xlabel("episode")
                    plt.ylabel("rewards")
                
                    discounted_episode_rewards = discount_and_normalise_rewards(episode_rewards)
                
                    loss_, _ = sess.run([loss, train_opt], feed_dict = {input: np.vstack(np.array(episode_states)),
                                                                        actions: np.vstack(np.array(episode_actions)),
                                                                        discounted_episode_rewards_: discounted_episode_rewards})
                
                    episode_states, episode_actions, episode_rewards = [],[],[]
                
                    break
                
                state = new_state   
                
# Reinforce
            
with tf.name_scope('inputs'):
    input = tf.placeholder(tf.float32, shape=(None, state_size), name="input_")
    actions = tf.placeholder(tf.float32, shape=(None, action_size), name="acctions")
    discounted_episode_rewards_ = tf.placeholder(tf.float32, [None, ], name="discounted_episode_rewards")
    
    
    mean_reward_ = tf.placeholder(tf.float32, name = "mean_reward")
    
    
    with tf.name_scope('fc1'):
        fc1 = tf.contrib.layers.fully_connected(inputs = input, num_outputs = 10,
                                                activation_fn = tf.nn.relu,
                                                weights_initializer = tf.contrib.layers.xavier_initializer())
    with tf.name_scope('fc2'):
        fc2 = tf.contrib.layers.fully_connected(inputs = fc1, num_outputs = action_size,
                                                activation_fn = tf.nn.relu,
                                                weights_initializer = tf.contrib.layers.xavier_initializer())
    with tf.name_scope('fc3'):
        fc3 = tf.contrib.layers.fully_connected(inputs = fc2, num_outputs = action_size,
                                                activation_fn = None,
                                                weights_initializer = tf.contrib.layers.xavier_initializer())
    with tf.name_scope("softmax"):
        action_distribution = tf.nn.softmax(fc3)
        
    with tf.name_scope("loss"):
        neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits = fc3, labels = actions)
        loss = tf.reduce_mean(neg_log_prob * discounted_episode_rewards_)
        
    with tf.name_scope("train"):
        train_opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        
allRewards = []
total_rewards = 0
maximumRewardRecorded = 0
episode = 0
is_rendering = True
episode_states, episode_actions, episode_rewards = [],[],[]

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for episode in range(max_episodes):
        episode_rewards_sum = 0
        state = env.reset()
        if is_rendering:
            frames = []
        while True:
             if is_rendering:
                frame = env.render(mode = 'rgb_array')
                frames.append(frame)
       

                action_probability_distribution = sess.run(action_distribution,
                                                          feed_dict = {input: state.reshape([1,4])})
            
                action = np.random.choice(range(action_probability_distribution.shape[1]), p = action_probability_distribution.ravel())
            
                new_state, reward, done, info = env.step(action)
            
                episode_states.append(state)
            
            
                action_ = np.zeros(action_size)
                action_[action] = 1
            
                episode_actions.append(action_)
                episode_rewards.append(reward)
            
                if done:
                    episode_rewards_sum = np.sum(episode_rewards)
                
                    allRewards.append(episode_rewards_sum)
                
                    total_rewards = np.sum(allRewards)
                
                    mean_reward = np.divide(total_rewards, episode+1)
                
                    maximumRewardRecorded = np.argmax(allRewards)
                
                    print("==========================================")
                    print("Episode: ", episode)
                    print("Reward: ", episode_rewards_sum)
                    print("Mean Reward", mean_reward)
                    #print("Max reward so far: ", maximumRewardRecorded)
                    
                    plt.plot(allRewards)
            
                    plt.title("REINFORCE Cart Pole")
                    plt.xlabel("episode")
                    plt.ylabel("rewards")
                
                    discounted_episode_rewards = discount_and_normalise_rewards(episode_rewards)
                
                    loss_, _ = sess.run([loss, train_opt], feed_dict = {input: np.vstack(np.array(episode_states)),
                                                                        actions: np.vstack(np.array(episode_actions)),
                                                                        discounted_episode_rewards_: discounted_episode_rewards})
                
                    episode_states, episode_actions, episode_rewards = [],[],[]
                
                    break
                
                state = new_state

#Actor Critic                
#Critic
with tf.name_scope('inputs'):
        state = tf.placeholder(tf.float32, shape=(None, state_size), name="input_")
        actions = tf.placeholder(tf.float32, shape=(None, action_size), name="acctions")
        discounted_episode_rewards_ = tf.placeholder(tf.float32, [None, ], name="discounted_episode_rewards")
       
        
        mean_reward_ = tf.placeholder(tf.float32, name = "mean_reward")
    
    
        with tf.name_scope('fc1'):
            fc1 = tf.contrib.layers.fully_connected(inputs = input, num_outputs = 10,
                                                    activation_fn = tf.nn.relu,
                                                    weights_initializer = tf.contrib.layers.xavier_initializer())
        with tf.name_scope('fc2'):
            fc2 = tf.contrib.layers.fully_connected(inputs = fc1, num_outputs = action_size,
                                                    activation_fn = tf.nn.relu,
                                                    weights_initializer = tf.contrib.layers.xavier_initializer())
        with tf.name_scope('fc3'):
            fc3 = tf.contrib.layers.fully_connected(inputs = fc2, num_outputs = action_size,
                                                    activation_fn = None,
                                                    weights_initializer = tf.contrib.layers.xavier_initializer())
        
        with tf.name_scope("softmax"):
            action_distribution = tf.nn.softmax(fc3)
        diffs = discounted_episode_rewards_ - action_distribution
            
        with tf.name_scope("loss"):
            loss = tf.nn.l2_loss(action_distribution - 0.01*action_distribution)
        with tf.name_scope("train"):
            train_opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)
            
#Actor
with tf.name_scope('inputs'):
    input = tf.placeholder(tf.float32, shape=(None, state_size), name="input_")
    actions = tf.placeholder(tf.float32, shape=(None, action_size), name="acctions")
    #discounted_episode_rewards_ = tf.placeholder(tf.float32, [None, ], name="discounted_episode_rewards")
    
    
    advantage = tf.placeholder(tf.float32, name = "advantage")
    
    
    with tf.name_scope('fc1'):
        fc1 = tf.contrib.layers.fully_connected(inputs = input, num_outputs = 10,
                                                activation_fn = tf.nn.relu,
                                                weights_initializer = tf.contrib.layers.xavier_initializer())
    with tf.name_scope('fc2'):
        fc2 = tf.contrib.layers.fully_connected(inputs = fc1, num_outputs = action_size,
                                                activation_fn = tf.nn.relu,
                                                weights_initializer = tf.contrib.layers.xavier_initializer())
    with tf.name_scope('fc3'):
        fc3 = tf.contrib.layers.fully_connected(inputs = fc2, num_outputs = action_size,
                                                activation_fn = None,
                                                weights_initializer = tf.contrib.layers.xavier_initializer())
    with tf.name_scope("softmax"):
        action_distribution = tf.nn.softmax(fc3)
        
    with tf.name_scope("loss"):
        neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits = fc3, labels = actions)
        loss = tf.reduce_mean(neg_log_prob * discounted_episode_rewards_)
        
    with tf.name_scope("train"):
        train_opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)

            
allRewards = []
total_rewards = 0
maximumRewardRecorded = 0
episode = 0
is_rendering = True
episode_states, episode_actions, episode_rewards = [],[],[]

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for episode in range(max_episodes):
        episode_rewards_sum = 0
        state = env.reset()
        if is_rendering:
            frames = []
        while True:
             if is_rendering:
                frame = env.render(mode = 'rgb_array')
                frames.append(frame)
       

                action_probability_distribution = sess.run(action_distribution,
                                                          feed_dict = {input: state.reshape([1,4])})
            
                action = np.random.choice(range(action_probability_distribution.shape[1]), p = action_probability_distribution.ravel())
            
                new_state, reward, done, info = env.step(action)
            
                episode_states.append(state)
            
            
                action_ = np.zeros(action_size)
                action_[action] = 1
            
                episode_actions.append(action_)
                episode_rewards.append(reward)
            
                if done:
                    episode_rewards_sum = np.sum(episode_rewards)
                
                    allRewards.append(episode_rewards_sum)
                
                    total_rewards = np.sum(allRewards)
                
                    mean_reward = np.divide(total_rewards, episode+1)
                
                    maximumRewardRecorded = np.argmax(allRewards)
                
                    print("==========================================")
                    print("Episode: ", episode)
                    print("Reward: ", episode_rewards_sum)
                    print("Mean Reward", mean_reward)
                    #print("Max reward so far: ", maximumRewardRecorded)
                    
                    plt.plot(allRewards)
            
                    plt.title("Actor-Critic Cart Pole")
                    plt.xlabel("episode")
                    plt.ylabel("rewards")
                
                    discounted_episode_rewards = discount_and_normalise_rewards(episode_rewards)
                    

                    #discounted_episode_rewards = advantage - diffs

                    
                
                    loss_, _ = sess.run([loss, train_opt], feed_dict = {input: np.vstack(np.array(episode_states)),
                                                                        actions: np.vstack(np.array(episode_actions)),
                                                                        discounted_episode_rewards_: discounted_episode_rewards})
                
                    episode_states, episode_actions, episode_rewards = [],[],[]
                
                    break
                
                state = new_state
            
            
# reference code
#https://github.com/rlcode/reinforcement-learning/blob/master/2-cartpole/5-a3c/cartpole_a3c.py
#https://github.com/simoninithomas/Deep_reinforcement_learning_Course
            
