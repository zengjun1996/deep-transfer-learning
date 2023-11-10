import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# loss_b = pd.read_csv('CDL_B_loss.csv').iloc[:, 2]
# loss_c = pd.read_csv('CDL_C_loss.csv').iloc[:, 2]
# loss_d = pd.read_csv('CDL_D_loss.csv').iloc[:, 2]
loss_e = pd.read_csv('CDL_E_loss.csv').iloc[:, 2]
loss_d_e = pd.read_csv('CDL_D_to_CDL_E_1_8loss.csv').iloc[:, 2]

# plt.plot(loss_b[:], label='CDL-B')
# plt.plot(loss_c[:], label='CDL-C')
# plt.plot(loss_d[:], label='CDL-D')
plt.plot(loss_e[:], label='CDL-E (with CDL-A)')
plt.plot(loss_d_e[:], label='CDL-E (with CDL-D)')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Compression rate=1/8')
plt.legend()
plt.show()
