import torch
from experiments.env import FlappyBirdEnv
from agent import Agent
import pygame

def play(best_model_path="best_model_new.pth", episodes=10):
    
    env = FlappyBirdEnv(render_mode="human", frame_skip=1)

    
    env._init_display()

  
    agent = Agent(env.action_space.n)
    agent.net.load_state_dict(torch.load(best_model_path, map_location=agent.device))
    agent.net.eval()

    best_score = 0

    for ep in range(episodes):
        state, _ = env.reset()
        done = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    break

            action = agent.select_action(state, epsilon=0.05)
            state, reward, done, _, info = env.step(action)

            if info["score"] > best_score:
                best_score = info["score"]

           
            env.render(best_score=best_score)

        print(f"Episode {ep+1}: Score = {info['score']} | Best = {best_score}")

    env.close()
    pygame.quit()


if __name__ == "__main__":
    play(episodes=5)