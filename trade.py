import math
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import random
import tensorflow as tf
from tqdm import tqdm
from collections import deque

class AI_Trader():
    def __init__(self, state_size, action_space=3, model_name="AITrader"):
        self.state_size = state_size
        self.action_space = action_space
        self.memory = deque(maxlen=2000)
        self.model_name = model_name

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_final = 0.01
        self.epsilon_decay = 0.995

        self.model = self.model_builder()
    
    def model_builder(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(units=32, activation='relu', input_dim=self.state_size))
        model.add(tf.keras.layers.Dense(units=64, activation='relu'))
        model.add(tf.keras.layers.Dense(units=128, activation='relu'))
        model.add(tf.keras.layers.Dense(units=self.action_space, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

        return model
    
    def trade(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_space)
        actions = self.model.predict(state)
        return np.argmax(actions[0])
    
    def batch_train(self, batch_size):
        batch = []
        for i in range(len(self.memory) - batch_size + 1, len(self.memory)):
            batch.append(self.memory[i])
        
        for state, action, reward, next_state, done in batch:
            reward = reward
            if not done:
                reward = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target = self.model.predict(state)
            target[0][action] = reward

            self.model.fit(state, target, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_final:
            self.epsilon *= self.epsilon_decay

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def stocks_price_format(n):
    if isinstance(n, pd.Series):
        value = n.iloc[0]
    else:
        value = n
    return ("- R$" if value < 0 else "R$") + "{0:.2f}".format(abs(value))

        
def dataset_loader(stock_name):   
    dataset = yf.download(stock_name)
    close = dataset['Close']
    return close
    
def state_creator(data, timestep, window_size):
    starting_id = timestep - window_size + 1
        
    if starting_id >= 0:
        windowed_data = data[starting_id:timestep+1].values
    else:
        windowed_data = - starting_id * [data.iloc[0]] + list(data.iloc[0:timestep+1].values)
        
    state = []
    for i in range(window_size - 1):
        state.append(sigmoid(windowed_data[i+1] - windowed_data[i]))
    return np.array([state]), windowed_data
    

# Carregar dados
stock_name = "AAPL"
data = dataset_loader(stock_name)

# Criar estado inicial
s = state_creator(data, 0, 5)
window_size = 10

# Exemplo de uso trader
episodes = 10
batch_size = 32
data_samples = len(data)

trader = AI_Trader(window_size)

print(trader.model.summary())

for episode in range(1, episodes + 1):
    print(f'Episode: {episode} de {episodes}')
    state, _ = state_creator(data, 0, window_size + 1)
    total_profit = 0
    trader.inventory = []
    for t in tqdm(range(data_samples)):
        action = trader.trade(state)
        next_state, _ = state_creator(data, t+1, window_size + 1)
        reward = 0
        
        if action == 1:
            trader.inventory.append(data.iloc[t])
            print(f'AI Trader comprou: {stocks_price_format(data.iloc[t])}')
        elif action == 2 and len(trader.inventory) > 0:
            buy_price = trader.inventory.pop(0)
            current_price = data.iloc[t].item()
            reward = max(current_price - buy_price.item(), 0)
            total_profit += current_price - buy_price.item()
            print(f'AI Trader vendeu: {stocks_price_format(current_price)} Lucro de: {stocks_price_format(current_price - buy_price.item())}')
        
        if t == data_samples - 1:
            done = True
        else:
            done = False
        
        trader.memory.append((state, action, reward, next_state, done))
        state = next_state
        
        if done:
            print("########################")
            print(f'Lucro total estimado: {(total_profit)}')
            print("########################")
        
        if len(trader.memory) > batch_size:
            trader.batch_train(batch_size)
    
    if episode % 10 == 0:
        trader.model.save("ai_trader_{}.h5".format(episode))
