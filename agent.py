import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class Agent:
    def __init__(self, state_size):
        self.state_size = state_size
        self.action_size = 3
        self.memory = deque(maxlen=1000)
        self.inventory = []

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        self.model = self._model()

    def _model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation="relu"))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(8, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        options = self.model.predict(state, verbose=0)
        return np.argmax(options[0])

    def expReplay(self, batch_size):
        mini_batch = list(self.memory)[-batch_size:]

        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])

            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target

            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay