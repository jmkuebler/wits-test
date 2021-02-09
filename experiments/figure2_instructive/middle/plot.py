import numpy as np
import matplotlib.pyplot as plt

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

npzfile = np.load('fig2_middle.npy')
ratios = npzfile['arr_0']
power_kfda_witness = npzfile['arr_1']
power_mmd = npzfile['arr_2']
power_mmd_witness = npzfile['arr_3']


plt.plot(ratios, power_mmd_witness, label='opt-mmd-witness', color='tab:blue', marker='*')
plt.plot(ratios, power_mmd, label='opt-mmd-boot', color='tab:blue', ls='dashdot', marker='D')
plt.plot(ratios, power_kfda_witness, label='kfda-witness', color='tab:red', marker='x')

plt.legend()
plt.xlabel(r"Splitting Ratio $r = n_{tr} / (n_{tr} +n_{te})$")
plt.ylabel("Rejection Rate")


# plt.show()
plt.savefig('fig2_middle.pdf', bbox_inches="tight")