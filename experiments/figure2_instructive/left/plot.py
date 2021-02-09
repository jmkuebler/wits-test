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

npzfile = np.load('type-IIe-2.npy')
ratios = npzfile['arr_0']
power_mmd_witness = npzfile['arr_1']
power_mmd = npzfile['arr_2']
power_mmd_opt = npzfile['arr_3']
power_kfda_witness = npzfile['arr_4']
power_kfda_boot = npzfile['arr_5']


plt.plot(ratios, power_mmd_witness, label='mmd-witness', color='tab:blue', marker='*')
plt.plot(ratios, [power_mmd[0]]*len(ratios), label='mmd-boot', color='tab:blue', ls='dotted')
plt.plot(ratios, power_mmd_opt, label='opt-mmd-boot', color='tab:blue', ls='dashdot', marker='D')
plt.plot(ratios, power_kfda_witness, label='kfda-witness', color='tab:red', marker='x')
plt.plot(ratios, [power_kfda_boot[0]]*len(ratios), label='kfda-boot', color='tab:red', ls='dashed')

plt.legend()
plt.xlabel(r"Splitting Ratio $r = n_{tr} / (n_{tr} +n_{te})$")
plt.ylabel("Rejection Rate")
plt.ylim(0,1)

# plt.title('FDA witness with grid search optimized kernel and regularization.')

# plt.title('test power')


# plt.show()
plt.savefig('left_exp_reg-e-2.pdf', bbox_inches="tight")
# with open('type-II.npy', 'wb') as f:
#     np.savez(f, n, power_witness, power_mmd)