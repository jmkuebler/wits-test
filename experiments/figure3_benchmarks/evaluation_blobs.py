from matplotlib.font_manager import FontProperties
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')

font = {
    # 'family' : 'normal',
    # 'weight' : 'bold',
    'size': 11
}

plt.rc('font', **font)
plt.rc('lines', linewidth=2)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

fontP = FontProperties()
fontP.set_size('xx-small')
from pathlib import Path
import numpy as np
linewidth = 2
sample_per_mode = np.array([10,20,30,40,50])

#plot kfda witness this
npzfile = np.load('blobs_kfda_witness.npy')
power_kfda_witness = npzfile['arr_1']
plt.plot(sample_per_mode*9, power_kfda_witness[:5], label='kfda-witness', ls='solid', color='tab:red', marker='x')



results = []
for n in sample_per_mode:
    DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.joinpath("data")
    data_dir = Path(DEFAULT_DATA_DIR)
    filename = "results_Blobs_10x100_H1{}.npy".format(n)
    path = data_dir.joinpath(filename)
    # np.save('./Results_Blob_'+str(n)+'_H1_MMD-D.npy',Results)
    results.append(np.load(path))
# print(results)
results = np.mean(results, axis=2)
plt.plot(sample_per_mode*9, results[:, 1], label='MMD-D-witness', ls='solid', color='tab:blue', marker='*', linewidth=linewidth)
plt.plot(sample_per_mode*9, results[:, 0],  label='MMD-D', ls='dashdot', color='tab:blue', marker='D', linewidth=linewidth)



results_baselines = []
for n in sample_per_mode:
    DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.joinpath("data")
    data_dir = Path(DEFAULT_DATA_DIR)
    filename = "Results_Blob_{}_H1_Baselines.npy".format(n)
    path = data_dir.joinpath(filename)
    # np.save('./Results_Blob_'+str(n)+'_H1_MMD-D.npy',Results)
    results_baselines.append(np.load(path))
# print(results)
results_baselines = np.mean(results_baselines, axis=2)
methods = ["MMD-O", "C2ST_L", "C2ST_S", "ME", "SCF"]
markers = ['x', '+', '*', 'D', '.']
colors = ['tab:green', 'tab:cyan', 'tab:purple', 'gold', 'tab:olive']
for i in range(len(methods)):
    plt.plot(sample_per_mode*9, results_baselines[:, i], label=methods[i], ls='dotted', marker=markers[i], color=colors[i], linewidth=linewidth)

# plt.legend()
plt.xlabel("Samplesize")
plt.ylabel("Rejection Rate")
plt.savefig('blobs_including_baselines.pdf', bbox_inches="tight")
