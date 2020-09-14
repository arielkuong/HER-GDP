import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from arguments import get_args
import os

if __name__ == "__main__":

    args = get_args()
    data_len = args.demo_length
    eval_file_path = args.save_dir + args.env_name + '/seed_' + str(args.seed) + '/eval_success_rates.npy'

    if not os.path.isfile(eval_file_path):
        print("Result file do not exist!")
    else:
        data = np.load(eval_file_path)[:data_len]
        print(data)

        x = np.linspace(0, len(data), len(data))

        mpl.style.use('ggplot')
        fig = plt.figure(1)
        fig.patch.set_facecolor('white')
        plt.xlabel('Epochs', fontsize=16)
        plt.ylabel('Test Success Rate', fontsize=16)
        plt.title(args.env_name, fontsize=20)

        plt.plot(x, data, color='blue', linewidth=2, label='HER+GDP')
        plt.legend(loc='lower right')

        plt.show()
