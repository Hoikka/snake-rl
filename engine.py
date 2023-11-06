import torch

def train_one_episode(agent, env, loss_fn, epsilon):
    agent.train(True)


def evaluate_agent_performance(agent, env, n_games, epsilon):
    agent.eval()