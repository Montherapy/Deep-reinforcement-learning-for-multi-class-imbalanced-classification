from agent import Agent
from dataset import Dataset
from network import QNetwork
from memory import Memory
from option import config

if __name__ == '__main__':

    dataset = Dataset(config)
    q_network = QNetwork(config)
    memory = Memory()
    agent = Agent(q_network, dataset, memory, config)
    agent.train()