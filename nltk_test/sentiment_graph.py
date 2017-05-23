import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import time

style.use("ggplot")

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def animate(i):
    pull_data = open('twitter-feeds.txt','r').read()
    lines = pull_data.split('\n')
    x_arr = []
    y_arr = []
    x = 0
    y = 0    for line in lines:
        x+=1
        if "pos" in line:
            y+=1
        elif "neg" in line:
            y-=1

        x_arr.append(x)
        y_arr.append(y)

    ax1.clear()
    ax1.plot(x_arr, y_arr)

ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()
