import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Configure matplotlib with Latex font
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

dense = [21.47, 7.27, 2.52]
smyrf1x16 = [19.28, 12.18, 6.69]
smyrf1x32 = [19.25, 11.99, 6.56]
smyrf1x64 = [19.28, 11.95, 6.55]
smyrf1x128 = [19.19, 11.62, 6.33]
smyrf2x16 = [19.20, 11.41, 6.11]
smyrf2x32 = [19.12, 11.10, 5.93]
smyrf2x64 = [19.01, 11.03, 5.90]
smyrf2x128 = [18.28, 10.41, 5.57]
smyrf4x16 = [17.87, 10.02, 5.16]
smyrf4x32 = [17.17, 9.54, 4.93]
smyrf4x64 = [17.08, 9.45, 4.88]
smyrf4x128 = [15.62, 8.60, 4.44]

x_axis = [1024, 2048, 4096]
all = [dense, smyrf1x16, smyrf1x32, smyrf1x64, smyrf1x128,
       smyrf2x16, smyrf2x32, smyrf2x64, smyrf2x128,
       smyrf4x16, smyrf4x32, smyrf4x64, smyrf4x128]

legends = ['dense', 'smyrf1x16', 'smyrf1x32', 'smyrf1x64', 'smyrf1x128',
           'smyrf2x16', 'smyrf2x32', 'smyrf2x64', 'smyrf2x128',
           'smyrf4x16', 'smyrf4x32', 'smyrf4x64', 'smyrf4x128']

fig = plt.figure()
fig.suptitle('Elapsed time per iteration on BERT (base) for different SMYRF configurations. \n Batch size=1 for all experiments.')
plt.xlabel('Sequence length (number of queries/keys)')
plt.ylabel('sec/iter')

for y_axis in all:
    plt.plot(x_axis, 1 / np.array(y_axis), marker='o')

plt.legend(legends)
fig.savefig('../visuals/speed_batch1.png')
plt.show()
