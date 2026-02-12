import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Configuration ---
WIDTH = 100
HEIGHT = 50
NUM_PEDS = 30  # Number of pedestrians per direction
FRAMES = 300  # Length of video
FPS = 30

# Setup the figure
fig, ax = plt.subplots(figsize=(10, 5), facecolor='black')
ax.set_facecolor('black')
ax.set_xlim(0, WIDTH)
ax.set_ylim(0, HEIGHT)
ax.set_xticks([])
ax.set_yticks([])

# Remove borders for a clean input video
for spine in ax.spines.values():
    spine.set_visible(False)

# --- Initialize Pedestrians ---
# Lane 1: Top (Moving Left) - y range [30, 45]
# Lane 2: Bottom (Moving Right) - y range [5, 20]
# The middle [20, 30] is the "danger zone" or separator,
# but we will leave gaps *within* the flows for the robot.

# Generate random positions
# Top Lane (Moving Left <-)
x_top = np.random.uniform(0, WIDTH, NUM_PEDS)
y_top = np.random.uniform(30, 45, NUM_PEDS)
s_top = np.random.uniform(0.2, 0.5, NUM_PEDS)  # speed

# Bottom Lane (Moving Right ->)
x_bot = np.random.uniform(0, WIDTH, NUM_PEDS)
y_bot = np.random.uniform(5, 20, NUM_PEDS)
s_bot = np.random.uniform(0.2, 0.5, NUM_PEDS)  # speed

# Visual elements (White dots representing pedestrians)
scat_top = ax.scatter(x_top, y_top, c='white', s=50)
scat_bot = ax.scatter(x_bot, y_bot, c='white', s=50)

# Optional: Draw a faint line separating lanes (visual aid only)
ax.axhline(y=25, color='gray', linestyle='--', alpha=0.3)


def update(frame):
    global x_top, x_bot

    # Move Top Lane Left
    x_top -= s_top
    # Wrap around (infinite flow)
    x_top = np.where(x_top < 0, WIDTH, x_top)

    # Move Bottom Lane Right
    x_bot += s_bot
    # Wrap around (infinite flow)
    x_bot = np.where(x_bot > WIDTH, 0, x_bot)

    # Update scatter plots
    scat_top.set_offsets(np.c_[x_top, y_top])
    scat_bot.set_offsets(np.c_[x_bot, y_bot])

    return scat_top, scat_bot


# Create Animation
ani = animation.FuncAnimation(fig, update, frames=FRAMES, interval=1000 / FPS, blit=True)

# Save the video
print("Generating video... please wait.")
try:
    # Try saving as MP4 (Best for input)
    ani.save('pedestrian_flow.mp4', writer='ffmpeg', fps=FPS, dpi=100)
    print("Success! Saved as 'pedestrian_flow.mp4'")
except Exception as e:
    print(f"FFMpeg not found ({e}). Saving as GIF instead...")
    ani.save('pedestrian_flow.gif', writer='pillow', fps=FPS)
    print("Success! Saved as 'pedestrian_flow.gif'")

plt.close()