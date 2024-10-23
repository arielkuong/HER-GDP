import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from arguments import get_args
import os

# seed = [123, 439, 224, 759, 536]

if __name__ == "__main__":

    args = get_args()
    eval_file_path_1_mean = 'HER_only_scripts/' + args.save_dir + args.env_name + '/eval_success_rates_her_5seeds_mean.npy'
    eval_file_path_1_high = 'HER_only_scripts/' + args.save_dir + args.env_name + '/eval_success_rates_her_5seeds_high.npy'
    eval_file_path_1_low = 'HER_only_scripts/' + args.save_dir + args.env_name + '/eval_success_rates_her_5seeds_low.npy'
    eval_file_path_2_mean = 'HER_GDP_scripts/' + args.save_dir + args.env_name + '/eval_success_rates_gdp_5seeds_mean.npy'
    eval_file_path_2_high = 'HER_GDP_scripts/' + args.save_dir + args.env_name + '/eval_success_rates_gdp_5seeds_high.npy'
    eval_file_path_2_low = 'HER_GDP_scripts/' + args.save_dir + args.env_name + '/eval_success_rates_gdp_5seeds_low.npy'



    data_len = 40
    data1_mean = np.load(eval_file_path_1_mean)[:data_len]
    data1_high = np.load(eval_file_path_1_high)[:data_len]
    data1_low = np.load(eval_file_path_1_low)[:data_len]
    data2_mean = np.load(eval_file_path_2_mean)[:data_len]
    data2_high = np.load(eval_file_path_2_high)[:data_len]
    data2_low = np.load(eval_file_path_2_low)[:data_len]

    x = np.linspace(0, data_len, data_len)

    mpl.style.use('ggplot')
    fig = plt.figure(1)
    fig.patch.set_facecolor('white')
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Test Success Rate', fontsize=16)
    plt.title(args.env_name, fontsize=20)

    plt.plot(x, data1_mean, color='blue', linewidth=2, label='vanilla HER')
    plt.fill_between(x, data1_low, data1_high, color='green', alpha=0.1)
    plt.plot(x, data2_mean, color='red', linewidth=2, label='HER+GDP')
    plt.fill_between(x, data2_low, data2_high, color='blue', alpha=0.1)
    plt.legend(loc='lower right')

    plt.show()
