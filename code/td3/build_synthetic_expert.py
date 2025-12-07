# build_synthetic_expert.py
import numpy as np

def generate_expert_actions(states_raw):
    """
    states_raw contains:
        scene, lidar, speed, yaw_rate, weather_logits
    We use:
        lidar_norm = states_raw[:, 768]
        speed_norm = states_raw[:, 769]
        weather_logits = states_raw[:, 771:776]
    """

    lidar = states_raw[:, 768]
    speed = states_raw[:, 769]
    weather_logits = states_raw[:, 771:]

    # Convert weather logits → 1-hot
    weather_idx = np.argmax(weather_logits, axis=1)
    weather_onehot = np.eye(5)[weather_idx]

    actions = []

    for i in range(len(states_raw)):
        w = weather_onehot[i]
        sun, fog, rain, snow, night = w

        # ------- Base expert logic -------
        # Target speed depends on weather
        if fog or rain or snow:
            target_speed = 0.1
        elif night:
            target_speed = 0.3
        else:
            target_speed = 0.5

        # Acceleration expert
        accel = 3 * (target_speed - speed[i])
        accel = np.clip(accel, -1, 1)

        # Steering expert:
        # small random steering for realism
        steer = np.clip(np.random.normal(0, 0.05), -1, 1)

        # Strong correction: if lidar dangerously low
        if lidar[i] < 0.1:
            accel = -1.0   # full brake
            steer = 0.0

        actions.append([accel, steer])

    actions = np.array(actions)
    return actions, weather_onehot
