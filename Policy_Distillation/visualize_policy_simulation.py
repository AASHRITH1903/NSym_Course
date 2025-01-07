import gymnasium



# Simulate using trained policy
frames = []

env  = gymnasium.make("Acrobot-v1", render_mode="rgb_array")
s, info = env.reset()

action_func = get_action_func(policy)

for _ in range(200):
    a = action_func(s)

    s, reward, terminated, truncated, info = env.step(a)
    
    frames.append(env.render())

    if terminated or truncated:
        s, info = env.reset()

env.close()



# make GIF using frames
from matplotlib import animation

fig = plt.figure(figsize=(6, 6))
plt.axis('off') 

def update_frame(num):
    plt.imshow(frames[num])
    return plt

ani = animation.FuncAnimation(fig, update_frame, frames=len(frames), interval=50)

ani.save("dt_acrobot_animation.gif", writer="pillow", fps=10)