# reward_function.py
import numpy as np

def compute_reward(lidar, speed, accel, steer, weather_onehot):
    """
    lidar: normalized (0 = very close, 1 = far)
    speed: normalized 0–1
    accel, steer: actions in [-1,1]
    weather_onehot: [sun, fog, rain, snow, night]
    """

    sun, fog, rain, snow, night = weather_onehot

    bad_weather = fog + rain + snow

    # ------------------------------------
    # Base reward: moving forward safely
    # ------------------------------------
    reward = 2.0 * speed                        # strong positive incentive

    # ------------------------------------
    # Weather-aware control
    # ------------------------------------
    if bad_weather > 0.1:
        # Penalize fast speeds
        reward -= 8.0 * speed

        # Penalize acceleration in bad weather
        reward -= 4.0 * max(accel, 0)

        # Penalize steering aggressively
        reward -= 3.0 * abs(steer)

        # Reward slowing down
        if speed < 0.2:
            reward += 4.0

    # ------------------------------------
    # Proximity safety (lidar)
    # ------------------------------------
    # lidar normalized: 0 = danger, 1 = safe
    if lidar < 0.2:
        reward -= 20.0 * (0.2 - lidar)    # large penalty
    elif lidar > 0.8:
        reward += 2.0                     # reward safe margin

    # Smooth driving
    reward -= 0.5 * abs(accel)
    reward -= 0.5 * (steer ** 2)

    return float(reward)
