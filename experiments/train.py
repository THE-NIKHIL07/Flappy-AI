import torch
from env import FlappyBirdEnv
from agent import Agent

def train(resume_best_model=True):
    max_episodes = 10000
    epsilon_start = 0.01
    epsilon_final = 0.01
    epsilon_decay = 0.995
    epsilon = epsilon_start
    best_score = 12
    tau = 0.01

    env = FlappyBirdEnv(render_mode=None, frame_skip=1, headless=True)
    agent = Agent(env.action_space.n)

    if resume_best_model:
        try:
            agent.net.load_state_dict(torch.load("checkpoint_ep300.pth", map_location=agent.device))
            agent.target_net.load_state_dict(agent.net.state_dict())
            print("Resumed training from best_model.pth")
        except FileNotFoundError:
            print("No best_model.pth found. Starting from scratch.")

    for ep in range(max_episodes):
        if ep == 1000:
            env.frame_skip = 2
            print("Switched frame_skip to 2 for faster training.")

        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state, epsilon)
            next_state, reward, done, _, info = env.step(action)
            agent.append(state, action, reward, next_state, done)
            agent.train_step()
            state = next_state
            total_reward += reward

        agent.soft_update(tau)

        if info["score"] > best_score:
            best_score = info["score"]
            torch.save(agent.net.state_dict(), "best_model_new.pth")

        if ep % 100 == 0 and ep != 0:
            torch.save(agent.net.state_dict(), f"checkpoint_ep{ep}.pth")

        epsilon = max(epsilon_final, epsilon * epsilon_decay)

        print(f"Episode: {ep} | Score: {info['score']} | Best: {best_score} | Epsilon: {epsilon:.3f}")

if __name__ == "__main__":
    train()