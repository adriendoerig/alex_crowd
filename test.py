import numpy as np
import matplotlib.pyplot as plt
import random

n_runs = 2

for run in range(n_runs):
    for STIM in ['vernier', 'crowded', 'uncrowded']:
        if 'vernier' in STIM:
            for TRAINING in [True, False]:
                print(STIM+'___'+str(TRAINING))
        else:
            TRAINING = False
            print(STIM + '___' + str(TRAINING))

#
# n_runs = 20
# percent_correct_runs1 = np.zeros(shape=(n_runs, 9))
# percent_correct_runs2 = np.zeros(shape=(n_runs, 9))
# percent_correct_runs3 = np.zeros(shape=(n_runs, 9))
#
# for run in range(n_runs):
#     percent_correct1 = 55*np.ones(shape=(9))+[0, 1, 0, 0, 1, 2, 3, 0, -1]
#     percent_correct_runs1[run, :] = percent_correct1
#     percent_correct2 = 60 * np.ones(shape=(9))*random.randint(0, 1)
#     percent_correct_runs2[run, :] = percent_correct2
#     percent_correct3 = 75 * np.ones(shape=(9))*random.randint(0, 1)
#     percent_correct_runs3[run, :] = percent_correct3
#
# print(np.mean(percent_correct_runs1, axis=0))
# print(np.std(percent_correct_runs1, axis=0))
# print(np.mean(percent_correct_runs2, axis=0))
# print(np.std(percent_correct_runs2, axis=0))
#
# ####### PLOT RESULTS #######
#
# # cosmetics
# layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8', 'prob']
#
# N = len(layers)
# ind = np.arange(N)  # the x locations for the groups
# width = 0.25        # the width of the bars
#
# fig, ax = plt.subplots()
# rects1 = ax.bar(ind, np.mean(percent_correct_runs1, axis=0), width, color=(146./255, 181./255, 88./255))
# rects2 = ax.bar(ind + width, np.mean(percent_correct_runs2, axis=0), width, color=(220./255, 76./255, 70./255))
# rects3 = ax.bar(ind + 2*width, np.mean(percent_correct_runs3, axis=0), width, color=(79./255, 132./255, 196./255))
#
# # add some text for labels, title and axes ticks, and save figure
# ax.set_ylabel('Percent correct')
# ax.set_title('Vernier decoding from alexnet layers')
# ax.set_xticks(ind + width / 2)
# ax.set_xticklabels(layers)
# ax.plot([-0.3,N], [50, 50], 'r--')  # chance level cashed line
# ax.legend((rects1[0], rects2[0], rects3[0]), ('vernier', '1 square', '7 squares'))
# plt.show()