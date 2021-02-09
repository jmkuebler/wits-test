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

npzfile = np.load('fig2_right.npy')
n = npzfile['arr_0']
power_kfda_witness = npzfile['arr_1']
power_mmd = npzfile['arr_2']
power_mmd_witness = npzfile['arr_3']


plt.plot(n, power_mmd, label='opt-mmd-boot', color='tab:blue', ls='dashdot', marker='D')
plt.plot(n, power_kfda_witness, label='kfda-witness', color='tab:red', marker='x')
plt.plot(n, power_mmd_witness, label='opt-mmd-witness', color='tab:blue', marker='*')

plt.legend()
plt.xlabel("Samplesize")
plt.ylabel("Rejection Rate")

# plt.title('FDA witness with grid search optimized kernel and regularization.')

# plt.title('test power')


# plt.show()
plt.savefig('fig2_right.pdf', bbox_inches="tight")
# with open('type-II.npy', 'wb') as f:
#     np.savez(f, n, power_witness, power_mmd)