import matplotlib.pyplot as plt
import numpy as np

NUM_COLORS = 50

cm = plt.get_cmap('gist_rainbow')
fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(NUM_COLORS):
    lines = ax.plot(np.arange(10)*(i+1))
    lines[0].set_color(cm(i//3*3.0/NUM_COLORS))
    print(cm(i//3*3.0/NUM_COLORS))
    lines[0].set_linewidth(i%3 + 1)

fig.savefig("/mnt/c/Users/spide/Desktop/test.jpg")