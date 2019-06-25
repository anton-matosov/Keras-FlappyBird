# -*- coding: utf-8 -*-

import argparse
import os
import random
from collections import deque

import numpy as np
import skimage
import tensorflow as tf
from tensorflow.keras.layers import Activation, Convolution2D, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

import game.wrapped_flappy_bird as game

GAME = 'bird'  # the name of the game being played for log files
ACTIONS = 2  # number of valid actions
GAMMA = 0.99  # decay rate of past observations
OBSERVATION = 5000.  # timesteps to observe before training
EXPLORE = 50000.  # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001  # final value of epsilon
INITIAL_EPSILON = 0.1  # starting value of epsilon
REPLAY_MEMORY = 50000  # number of previous transitions to remember
BATCH = 32  # size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-4
TIME_PERIOD = 1000

IMG_ROWS, IMG_COLS = 80, 80  # Convert image into Black and white
IMG_CHANNELS = 4  # We stack 4 frames

basedir = 'saved_networks'
model_name = 'model'


def build_model():
    model = Sequential()
    model.add(Convolution2D(
        32, 8, 8,
        input_shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS))
    )  # 80*80*4

    model.add(Activation('relu'))
    model.add(Convolution2D(64, 4, 4))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 2, 2))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(2))

    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse', optimizer=adam)

    return model


def trainNetwork(model, args):
    if args['mode'] == 'run':
        OBSERVE = 999999999  # We keep observe, never train
        epsilon = FINAL_EPSILON
    elif args['mode'] == 'train':  # We go to training mode
        OBSERVE = OBSERVATION
        epsilon = INITIAL_EPSILON
    else:
        raise ValueError('Please key in run or train.')

    # open up a game state to communicate with emulator
    game_state = game.GameState()

    # store the previous observations in replay memory
    D = deque()

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, _, terminal = game_state.frame_step(do_nothing)

    x_t = skimage.color.rgb2gray(x_t)
    x_t = skimage.transform.resize(x_t, (80, 80))
    x_t = skimage.exposure.rescale_intensity(x_t, out_range=(0, 255))

    x_t /= 255

    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    # In Keras, need to reshape
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  # 1*80*80*4

    t0 = 0
    print("{} Now we load weight {}".format('-'*30, '-'*30))
    checkpoint = tf.train.get_checkpoint_state(basedir)
    if checkpoint and checkpoint.model_checkpoint_path:
        model_path = checkpoint.model_checkpoint_path
        name = model_path.rsplit(os.path.sep)[1]
        t0 = int(name.rsplit('-')[1])
        model.load_weights(model_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find any old network weights.")

    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse', optimizer=adam)

    t = 0
    while True:
        loss = 0
        Q_sa = 0
        action_index = 0
        r_t = 0
        a_t = np.zeros([ACTIONS])
        # choose an action epsilon greedy
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                print("{} Random Action {}".format('-'*30, '-'*30))
                action_index = random.randrange(ACTIONS)
                a_t[action_index] = 1
            else:
                # input a stack of 4 images, get the prediction
                q = model.predict(s_t)
                max_Q = np.argmax(q)
                action_index = max_Q
                a_t[max_Q] = 1

        # We reduced the epsilon gradually
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run the selected action and observed next state and reward
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)

        x_t1 = skimage.color.rgb2gray(x_t1_colored)
        x_t1 = skimage.transform.resize(x_t1, (80, 80))
        x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))

        x_t1 /= 255

        x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1)  # 1x80x80x1
        s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)

        # store the transition in D
        D.append((s_t, action_index, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # only train if done observing
        if t > OBSERVE:
            # sample a minibatch to train on
            minibatch = random.sample(D, BATCH)

            # Now we do the experience replay
            state_t, action_t, reward_t, state_t1, terminal = zip(*minibatch)
            state_t = np.concatenate(state_t)
            state_t1 = np.concatenate(state_t1)
            targets = model.predict(state_t)
            Q_sa = model.predict(state_t1)
            targets[range(BATCH), action_t] = (
                reward_t + GAMMA * np.max(Q_sa, axis=1)*np.invert(terminal))

            loss += model.train_on_batch(state_t, targets)

        s_t = s_t1
        t += 1

        # save progress every TIME_PERIOD iterations
        if t % TIME_PERIOD == 0 and args['mode'] == 'train':
            print("{} Now we save model {}".format('-'*30, '-'*30))
            model_path = os.path.join(
                basedir, "{}-{}".format(model_name, t+t0))
            model.save_weights(model_path)

        # print info
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print(
            "TIMESTEP: {}, STATE: {}, EPSILON: {:.5f}, ACTION: {}, REWARD: {}, Q_MAX: {:.2f}, Loss: {:.5f}".format(
                t+t0, state, epsilon, action_index, r_t, np.max(Q_sa), loss))

    print("{} Episode finished! {}".format('-'*30, '-'*30))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', help='train / run', required=True)
    args = vars(parser.parse_args())
    model = build_model()
    trainNetwork(model, args)


if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)
    main()
