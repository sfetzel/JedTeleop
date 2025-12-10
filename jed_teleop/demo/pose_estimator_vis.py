import numpy as np
import matplotlib.pyplot as plt

from ..hand_pose_estimator import HandPoseEstimator

from scipy.spatial.transform import Rotation as R
import matplotlib.animation as animation

estimator = HandPoseEstimator(2)
#estimator = MockEstimator()
estimator.start()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_aspect('equal') 

# Draw centered axes
val = [1,0,0]
labels = ['x', 'y', 'z']
colors = ['r', 'g', 'b']
for v in range(3):
    x = [val[v-0], -val[v-0]]
    y = [val[v-1], -val[v-1]]
    z = [val[v-2], -val[v-2]]
    ax.plot(x,y,z,'k-', linewidth=1)
    ax.text(val[v-0], val[v-1], val[v-2], labels[v], color=colors[v], fontsize=20)

xs = [0,0]
ys = [0,0]
zs = [0,0]

# Create a blank line that will be updated in the animate function.
line_x, = ax.plot3D(xs, ys, zs, color='#ff0000', marker="<", markevery=[1])
line_y, = ax.plot3D(xs, ys, zs, color='#00ff00', marker="<", markevery=[1])
line_z, = ax.plot3D(xs, ys, zs, color='#0000ff', marker="<", markevery=[1])

point = np.array([0.5, 0.5, 0.5])
rotation = np.array([0.0,0,0])
s = 0.2
cs = np.array([[-s,0,0], [s,0,0], [0,-s,0], [0,s,0], [0,0,-s], [0,0,s]])

def update_points(frame):
    global point, rotation
    result = estimator.get_deltas()
    if not result is None:
        rotation += result[3:6]
        point += result[:3]
        
        r = R.from_euler('xyz', rotation, degrees=False)
        directions = r.apply(cs) + point

        xs = directions[:, 0]
        ys = directions[:, 1]
        zs = directions[:, 2]

        line_x.set_data_3d(directions[0:2, 0], directions[0:2, 1], directions[0:2, 2])
        line_y.set_data_3d(directions[2:4, 0], directions[2:4, 1], directions[2:4, 2])
        line_z.set_data_3d(directions[4:, 0], directions[4:, 1], directions[4:, 2])
        plt.draw()
    #timer = threading.Timer(2.0, update_points)
    #timer.start()

#update_points()

# Hide everything else
# Hide axes ticks
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
# make the panes transparent
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# Hide box axes
ax._axis3don = False

# Expand to remove white space
ax.set_xlim(np.array([-1,1])*.57)
ax.set_ylim(np.array([-1,1])*.57)
ax.set_zlim(np.array([-1,1])*.57)

plt.tight_layout()

ani = animation.FuncAnimation(fig=fig, func=update_points, frames=400, interval=125)

plt.show()
plt.close()

estimator.stop()
print("Stopped estimator")
