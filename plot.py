import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from arguments import get_args
import os

if __name__ == "__main__":

    args = get_args()
    data_len = args.demo_length
    eval_file_path = args.save_dir + args.env_name + '/seed_' + str(args.seed) + '/eval_success_rates.npy'
    eval_file_1_path = args.save_dir + args.env_name + '/seed_' + str(args.seed) + '/eval_success_rates_vanillaHER.npy'
    eval_file_2_path = args.save_dir + args.env_name + '/seed_' + str(args.seed) + '/eval_success_rates_MEP.npy'

    if not os.path.isfile(eval_file_path):
        print("Result file do not exist!")
    else:
        data = np.load(eval_file_path)[:data_len]
        data1 = np.load(eval_file_1_path)[:data_len]
        data2 = np.load(eval_file_2_path)[:data_len]
        print(data)

        # data_epoch = []
        # for i in range(len(data)):
        #     if i%args.n_cycles == 0:
        #         data_epoch.append(data[i].copy())
        # data_epoch = np.array(data_epoch)
        # data_epoch = data_epoch[:20]
        # print(data_epoch)
        # print(data_epoch.shape)
        # print(data.shape)
        x = np.linspace(0, len(data), len(data))
        x1 = np.linspace(0, len(data1), len(data1))
        x2 = np.linspace(0, len(data2), len(data2))
        #x_fix = np.linspace(0, data_len, data_len)

        mpl.style.use('ggplot')
        fig = plt.figure(1)
        fig.patch.set_facecolor('white')
        plt.xlabel('Epochs', fontsize=16)
        plt.ylabel('Test Success Rate', fontsize=16)
        plt.title(args.env_name, fontsize=20)

        #plt.plot(x_fix, data2, color='lightgreen', linewidth=2, label='EggFull, HER+EBP')
        #plt.plot(x_fix, data1, color='indianred', linewidth=2, label='EggFull, vanilla HER')
        plt.plot(x1, data1, color='red', linewidth=2, label='HER')
        plt.plot(x2, data2, color='green', linewidth=2, label='HER+MEP')
        plt.plot(x, data, color='blue', linewidth=2, label='HER+my_MEP')


        plt.legend(loc='lower right')

        plt.show()
