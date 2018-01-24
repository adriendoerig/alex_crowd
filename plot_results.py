import numpy as np
import matplotlib.pyplot as plt

# load results
LOGDIR = 'alexcrowd_batch_norm_5_logdir\\version_0_hidden_512_resize_2.0'
vernier = np.squeeze(np.load(LOGDIR+'/vernier_percent_correct.npy'))
crowded = np.squeeze(np.load(LOGDIR+'/crowded_percent_correct.npy'))
uncrowded = np.squeeze(np.load(LOGDIR+'/uncrowded_percent_correct.npy'))

# cosmetics
layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8', 'prob']


####### PLOT RESULTS #######

N = len(layers)
ind = np.arange(N)  # the x locations for the groups
width = 0.25        # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, vernier, width, color=(146./255, 181./255, 88./255))
rects2 = ax.bar(ind + width, crowded, width, color=(220./255, 76./255, 70./255))
rects3 = ax.bar(ind + 2*width, uncrowded, width, color=(79./255, 132./255, 196./255))

# add some text for labels, title and axes ticks, and save figure
ax.set_ylabel('Percent correct')
ax.set_title('Vernier decoding from alexnet layers')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(layers)
ax.plot([-0.3,N], [50, 50], 'r--') # chance level cashed line
ax.legend((rects1[0], rects2[0], rects3[0]), ('vernier', '1 square', '7 squares'))
plt.savefig(LOGDIR+'/plot.png')