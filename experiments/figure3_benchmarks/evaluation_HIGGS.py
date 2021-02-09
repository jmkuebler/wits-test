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

linewidth = 2

from pathlib import Path
import numpy as np
ns = [1000, 2000, 3000, 5000]

#plot kfda witness this
npzfile = np.load('kfda_higgs_M500_100iterations.npy')
power_kfda_witness = npzfile['arr_1']
plt.plot(ns, power_kfda_witness, label='kfda-witness', ls='solid', color='tab:red', marker='x', linewidth=linewidth)




results = []
for n in ns:
    DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.joinpath("data")
    data_dir = Path(DEFAULT_DATA_DIR)
    filename = "Results_100times10_HIGGS_n{}_H1_MMD-D.npy".format(n)
    path = data_dir.joinpath(filename)
    # np.save('./Results_Blob_'+str(n)+'_H1_MMD-D.npy',Results)
    results.append(np.load(path))
# print(results)
results = np.mean(results, axis=2)
plt.plot(ns, results[:, 1], label='MMD-D-witness', ls='solid', color='tab:blue', marker='*', linewidth=linewidth)
plt.plot(ns, results[:, 0],  label='MMD-D', ls='dashdot', color='tab:blue', marker='D', linewidth=linewidth)

results_baselines = []
for n in ns:
    DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.joinpath("data")
    data_dir = Path(DEFAULT_DATA_DIR)
    filename = "Results_HIGGS_{}_H1_Baselines.npy".format(n)
    path = data_dir.joinpath(filename)
    # np.save('./Results_Blob_'+str(n)+'_H1_MMD-D.npy',Results)
    results_baselines.append(np.load(path))
# print(results)
results_baselines = np.mean(results_baselines, axis=2)
methods = ["MMD-O", "C2ST_L", "C2ST_S", "ME", "SCF"]
markers = ['x', '+', '*', 'D', '.']
colors = ['tab:green', 'tab:cyan', 'tab:purple', 'gold', 'tab:olive']
for i in range(len(methods)):
    plt.plot(ns, results_baselines[:, i], label=methods[i], ls='dotted', marker=markers[i], color=colors[i], linewidth=linewidth)

plt.legend(ncol=4, bbox_to_anchor=(-0.1,1.2), loc="upper left")
plt.xlabel("Samplesize")
plt.ylabel("Rejection Rate")
plt.savefig('Deep_mmd_vs_witness_HIGGS.pdf', bbox_inches="tight")
