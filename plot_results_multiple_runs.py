import numpy as np
import matplotlib.pyplot as plt

# load results
LOGDIR = 'alexcrowd_batch_norm_666_logdir'
n_runs = 6
n_hidden = 512
resize_factor = 1.0

vernier = np.zeros(shape=(n_runs, 9))
crowded = np.zeros_like(vernier)
uncrowded = np.zeros_like(vernier)

for run in range(n_runs):
    folder = LOGDIR + '/version_' + str(run) + '_hidden_' + str(n_hidden )+ '_resize_' + str(resize_factor)
    vernier[run, :] = np.squeeze(np.load(folder+'/vernier_percent_correct.npy'))
    crowded[run, :] = np.squeeze(np.load(folder+'/crowded_percent_correct.npy'))
    uncrowded[run, :] = np.squeeze(np.load(folder+'/uncrowded_percent_correct.npy'))


vernier_avg = np.mean(vernier, axis=0)
vernier_std = np.std(vernier, axis=0)
crowded_avg = np.mean(crowded, axis=0)
crowded_std = np.std(crowded, axis=0)
uncrowded_avg = np.mean(uncrowded, axis=0)
uncrowded_std = np.std(uncrowded, axis=0)


# cosmetics
layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8', 'prob']

# PLOT RESULTS #

N = len(layers)
ind = np.arange(N)  # the x locations for the groups
width = 0.25        # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, vernier_avg, width, yerr=vernier_std, color=(3./255, 57./255, 108./255))
rects2 = ax.bar(ind + width, crowded_avg, width, yerr=crowded_std, color=(0./255, 91./255, 150./255))
rects3 = ax.bar(ind + 2*width, uncrowded_avg, width, yerr=uncrowded_std, color=(100./255, 151./255, 177./255))
# rects1 = ax.bar(ind, vernier_avg, width, yerr=vernier_std, color=(146./255, 181./255, 88./255))
# rects2 = ax.bar(ind + width, crowded_avg, width, yerr=crowded_std, color=(220./255, 76./255, 70./255))
# rects3 = ax.bar(ind + 2*width, uncrowded_avg, width, yerr=uncrowded_std, color=(79./255, 132./255, 196./255))

# add some text for labels, title and axes ticks, and save figure
ax.set_ylabel('Percent correct')
ax.set_title('Vernier decoding from alexnet layers')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(layers)
ax.plot([-0.3,N], [50, 50], 'k--')  # chance level cashed line
ax.legend((rects1[0], rects2[0], rects3[0]), ('vernier', '1 square', '7 squares'))
plt.savefig(LOGDIR+'/plot.png')