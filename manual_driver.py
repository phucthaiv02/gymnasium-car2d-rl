import gymnasium as gym
import pygame

pygame.init()
screen = pygame.display.set_mode((400, 300))

# Key mapping
ACTION_KEYS = {
    pygame.K_UP: 3,      # Accelerate
    pygame.K_DOWN: 4,    # Brake
    pygame.K_LEFT: 2,    # Steer left
    pygame.K_RIGHT: 1    # Steer right
}


def get_action_from_keys(keys):
    # Default to no action
    action = 0

    if keys[pygame.K_UP]:
        action = ACTION_KEYS[pygame.K_UP]  # Accelerate
    elif keys[pygame.K_DOWN]:
        action = ACTION_KEYS[pygame.K_DOWN]  # Brake
    elif keys[pygame.K_LEFT]:
        action = ACTION_KEYS[pygame.K_LEFT]  # Steer left
    elif keys[pygame.K_RIGHT]:
        action = ACTION_KEYS[pygame.K_RIGHT]  # Steer right

    return action


env = gym.make("CarRacing-v2", render_mode="human", max_episode_steps=10000,
               lap_complete_percent=0.95, domain_randomize=False, continuous=False)

state, info = env.reset()
done = False
rewards = 0
while not done:
    keys = pygame.key.get_pressed()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    action = get_action_from_keys(keys)

    next_state, reward, done, truncated, info = env.step(action)
    rewards += reward

    if truncated or done:
        print('Final reward:', rewards)
        break
    env.render()

env.close()
pygame.quit()
