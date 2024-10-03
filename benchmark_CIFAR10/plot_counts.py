import tikzplotlib
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from pathlib import Path
import numpy as np
import pickle

hfont = {'fontname':'sans serif'}
plt.rcParams["font.family"] = "Times New Roman"
matplotlib.rc('xtick', labelsize=19)
matplotlib.rc('ytick', labelsize=19)
matplotlib.rc('axes', titlesize=19)
# matplotlib.rc('legend', fontsize=20)
# matplotlib.rc('font', size=20)
# matplotlib.rc('figure', titlesize=20)
# matplotlib.rc('text', usetex=True)
def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)


def main():
    levels = 256
    
    pkl_repo = './benchmark_CIFAR10/results_pkl/'
    
    in_file = pkl_repo + 'resnet8pareto_conf_80_70_Pc8_Pm8_seed1.pkl'
    file = open(in_file, 'rb')
    front_conf8 = pickle.load(file)
    file.close()
    
    in_file = pkl_repo + 'resnet14pareto_conf_80_70_Pc8_Pm8_seed1.pkl'
    file = open(in_file, 'rb')
    front_conf14 = pickle.load(file)
    file.close()
    
    in_file = pkl_repo + 'resnet20pareto_conf_80_70_Pc8_Pm8_seed1.pkl'
    file = open(in_file, 'rb')
    front_conf20 = pickle.load(file)
    file.close()
    
    in_file = pkl_repo + 'resnet32pareto_conf_120_70_Pc8_Pm8_seed1.pkl'
    file = open(in_file, 'rb')
    front_conf32 = pickle.load(file)
    file.close()

    in_file = pkl_repo + 'resnet50pareto_conf_120_70_Pc8_Pm8_seed1.pkl'
    file = open(in_file, 'rb')
    front_conf50 = pickle.load(file)
    file.close()

    in_file = pkl_repo + 'resnet56pareto_conf_120_70_Pc8_Pm8_seed1.pkl'
    file = open(in_file, 'rb')
    front_conf56 = pickle.load(file)
    file.close()
    
    count_conf8 = np.zeros(levels)
    count_conf14 = np.zeros(levels)
    count_conf20 = np.zeros(levels)
    count_conf32 = np.zeros(levels)
    count_conf50 = np.zeros(levels)
    count_conf56 = np.zeros(levels)
    conf_index = np.arange(0, levels)
    # conf_index = np.arange(0, 64)

    count_conf88 =  np.zeros(64)
    count_conf144 = np.zeros(64)
    count_conf200 = np.zeros(64)
    count_conf322 = np.zeros(64)
    count_conf500 = np.zeros(64)
    count_conf566 = np.zeros(64)

    tot = 0
    for conf in front_conf8:
        for lay in conf:
            count_conf8[lay] += 1
            tot += 1
    # count_conf8 = count_conf8 / tot * 100
    
    tot = 0
    for conf in front_conf14:
        for lay in conf:
            count_conf14[lay] += 1
            tot += 1
    # count_conf14 = count_conf14 / tot * 100

    tot = 0
    for conf in front_conf20:
        for lay in conf:
            count_conf20[lay] += 1
            tot += 1
    # count_conf20 = count_conf20 / tot * 100

    tot = 0
    for conf in front_conf32:
        for lay in conf:
            count_conf32[lay] += 1
            tot += 1
    # count_conf32 = count_conf32 / tot * 100
    
    tot = 0
    for conf in front_conf50:
        for lay in conf:
            count_conf50[lay] += 1
            tot += 1
    # count_conf50 = count_conf50 / tot * 100
    
    tot = 0
    for conf in front_conf56:
        for lay in conf:
            count_conf56[lay] += 1
            tot += 1
    # count_conf56 = count_conf56 / tot * 100
    # cluster count_conf8 data in chunks of N

    for i in range(0,64):
        count_conf88[i] = count_conf8[2*i] + count_conf8[2*i+1] + count_conf8[2*i+2] + count_conf8[2*i+3]
        count_conf144[i] = count_conf14[2*i] + count_conf14[2*i+1] + count_conf14[2*i+2] + count_conf14[2*i+3]
        count_conf200[i] = count_conf20[2*i] + count_conf20[2*i+1] + count_conf20[2*i+2] + count_conf20[2*i+3]
        count_conf322[i] = count_conf32[2*i] + count_conf32[2*i+1] + count_conf32[2*i+2] + count_conf32[2*i+3]
        count_conf500[i] = count_conf50[2*i] + count_conf50[2*i+1] + count_conf50[2*i+2] + count_conf50[2*i+3]
        count_conf566[i] = count_conf56[2*i] + count_conf56[2*i+1] + count_conf56[2*i+2] + count_conf56[2*i+3]



    # fig = figure(figsize=(13, 8))
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.8

    # ax.plot(conf_index, count_conf8 , marker="o", label="ResNet 8", color='blue')
    # ax.plot(conf_index, count_conf14, marker="+", label="ResNet 14", color='orange')
    # ax.plot(conf_index, count_conf20, marker=">", label="ResNet 20", color='gray')
    # ax.plot(conf_index, count_conf32, marker="*", label="ResNet 32", color='red')
    # ax.plot(conf_index, count_conf50, marker=">", label="ResNet 50", color='green')
    # ax.plot(conf_index, count_conf56, marker="o", label="ResNet 56", color='black')

    # ax.bar(conf_index, count_conf8+count_conf14+count_conf20+count_conf32+count_conf50+count_conf56, width=width, label="ResNet 8", color='blue')
    # ax.bar(conf_index, count_conf88+count_conf144+count_conf200+count_conf322+count_conf500+count_conf566, width=width, label="ResNet 8", color='blue')
    # ax.bar(conf_index, count_conf8, width=width, label="ResNet 8", color='orange')
    # ax.bar(conf_index + width, count_conf14, width=width, label="ResNet 14", color='gray')
    # ax.bar(conf_index + 2 * width, count_conf20, width=width, label="ResNet 20", color='red')
    # ax.bar(conf_index + 3 * width, count_conf32, width=width, label="ResNet 32", color='black')
    # ax.bar(conf_index + 4 * width, count_conf50, width=width, label="ResNet 50", color='green')
    # ax.bar(conf_index + 5 * width, count_conf56, width=width, label="ResNet 56", color='blue')

    ax.bar(conf_index, count_conf8, width=width, label="ResNet-8", color='orange' )
    bottom = count_conf8
    ax.bar(conf_index, count_conf14, width=width, label="ResNet-14", color='gray' , bottom=bottom)
    bottom += count_conf14
    ax.bar(conf_index, count_conf20, width=width, label="ResNet-20", color='red'  , bottom=bottom)
    bottom += count_conf20
    ax.bar(conf_index, count_conf32, width=width, label="ResNet-32", color='black', bottom=bottom)
    bottom += count_conf32
    ax.bar(conf_index, count_conf50, width=width, label="ResNet-50", color='green', bottom=bottom)
    bottom += count_conf50
    ax.bar(conf_index, count_conf56, width=width, label="ResNet-56", color='blue' , bottom=bottom)


    # ax.bar(conf_index, count_conf88, width=width, label="ResNet 8", color='orange')
    # ax.bar(conf_index + width, count_conf144, width=width, label="ResNet 14", color='gray')
    # ax.bar(conf_index + 2 * width, count_conf200, width=width, label="ResNet 20", color='red')
    # ax.bar(conf_index + 3 * width, count_conf322, width=width, label="ResNet 32", color='black')
    # ax.bar(conf_index + 4 * width, count_conf500, width=width, label="ResNet 50", color='green')
    # ax.bar(conf_index + 5 * width, count_conf566, width=width, label="ResNet 56", color='blue')
    # ax.xticks(fontsize=14)
    # ax.yticks(fontsize=14)
    ax.set_yscale('log')
    # ax.set_title("Approximate multiplier configuration occurrences")
    ax.set_xlabel("Approximation level", fontsize=19)
    ax.set_ylabel("Occurrence", fontsize=19)

    ax.set_ylim(0.8, 500)
    ax.set_xlim(-1, 257)
    ax.legend(ncols=6, loc="upper center", fontsize=10)
    fig.tight_layout()
    Path('./benchmark_CIFAR10/results_fig/').mkdir(parents=True, exist_ok=True)
    ax.grid(True, axis='y')
    plt.savefig('./benchmark_CIFAR10/results_fig/all_config_count.pdf', format='pdf', dpi=1200)
    
    tikzplotlib_fix_ncols(fig)
    tikzplotlib.save('./benchmark_CIFAR10/results_fig/all_config_count.tex')
    plt.show()
    
    higher = np.argmax(count_conf8)
    print(higher, count_conf8[higher])
    
    high8 = []
    high14 = []
    high20 = []
    high32 = []
    high50 = []
    high56 = []
    
    for i in range(5):
        higher = np.argmax(count_conf8)
        high8.append([higher, round(count_conf8[higher], 2)])
        count_conf8[higher] = 0
        higher = np.argmax(count_conf14)
        high14.append([higher, round(count_conf14[higher], 2)])
        count_conf14[higher] = 0
        higher = np.argmax(count_conf20)
        high20.append([higher, round(count_conf20[higher], 2)])
        count_conf20[higher] = 0
        higher = np.argmax(count_conf32)
        high32.append([higher, round(count_conf32[higher], 2)])
        count_conf32[higher] = 0
        higher = np.argmax(count_conf50)
        high50.append([higher, round(count_conf50[higher], 2)])
        count_conf50[higher] = 0
        higher = np.argmax(count_conf56)
        high56.append([higher, round(count_conf56[higher], 2)])
        count_conf56[higher] = 0
    print(f'resnet8 config count {high8}')
    print(f'resnet14 config count {high14}')
    print(f'resnet20 config count {high20}')
    print(f'resnet32 config count {high32}')
    print(f'resnet50 config count {high50}')
    print(f'ResNet56 config count {high56}')


if __name__ == "__main__":
    main()

