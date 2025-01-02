import pickle
import numpy as np
import matplotlib.pyplot as plt
import json
import copy
import seaborn as sns

sns.set(style="whitegrid")
def init_plotting():
    # plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams["figure.figsize"] = [20*1.4, 16]
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['font.size'] = 37
    plt.rcParams['axes.labelsize'] = 2.3 * plt.rcParams['font.size']
    plt.rcParams['axes.titlesize'] = 2.3 * plt.rcParams['font.size']
    # plt.rcParams['legend.fontsize'] = 2.3 * plt.rcParams['font.size']
    plt.rcParams['xtick.labelsize'] = 2.3 * plt.rcParams['font.size']
    plt.rcParams['ytick.labelsize'] = 2.3 * plt.rcParams['font.size']
    plt.rcParams['lines.linewidth'] =12
    plt.rcParams.update({'figure.autolayout': True})

init_plotting()

def exp_smooth(vals, gamma=.99):
    res = [vals[0]]
    tv = res[0]
    for v in vals[1:]:
        tv = gamma * tv  + (1-gamma) * v
        res.append(tv)
    return res

datasets = ['breastcancer']
titles = {
    'synthetic': 'Navigation',
    'fico': 'Fico',
    'compas': 'Compas',
    'led': 'LED',
    'diag': 'Troubleshooting',
    'breastcancer': 'Breast cancer'
}
criteria = ['EC2', 'IG', 'random', 'all', 'dpp']
num_rands = 5
explorations = {
    'IG': ['TS', 'UCB'],
    'EC2': ['TS', 'UCB'],
    'random': ['TS'],
    'all': ['TS'],
    'dpp': ['TS']
}
c1, c2, c3, c4, c5 = '#d7191c', '#2b83ba', '#4dac26', '#ed9722', '#edd222', 
cs = [c1, c2, c3, c4, c5]
colors = {
    ('IG','TS'):c1,
    ('IG','UCB'):c2,
    ('EC2','TS'):c3,
    ('EC2','UCB'): '#022e22',
    ('random','TS'): c4,
    ('dpp','TS'): '#eb34eb',
    ('all','TS'): c5
}

labels = {
    "EC2": r'W-$EC^2$',
    "IG" : "W-IG",
    "UCB": "BUCB",
    "TS": "TS",
    "Greedy": "eps-greedy",
    "random": "Random",
    "all": "All",
    "dpp":"DPP"
}

for dataset in datasets:
    cost_all_opt = pickle.load(open("TS"+"_weighted_sampled"+"odm_cost_dics_"+"IG"+"_"+dataset+".pkl", "rb" ))[0]
    num_hypos_list = list(cost_all_opt.keys())
    
    for criterion in criteria:
        for num_hypos in num_hypos_list:
            
            for expl in explorations[criterion]:
                
                
                cost_all = pickle.load(open(expl+"_weighted_sampled"+"odm_cost_dics_"+criterion+"_"+dataset+".pkl", "rb" ))[0]
                
                to_plot_array = np.array(cost_all[num_hypos][0]).reshape(num_rands,-1)/10
                
                to_plot_array_copy = copy.deepcopy(to_plot_array)
                to_plot_array = np.mean(to_plot_array, axis=0)
                
                if criterion in {'random','dpp','all'}:
                    plt.plot(exp_smooth(to_plot_array), label=labels[criterion], color=colors[(criterion,expl)])
                else:
                    plt.plot(exp_smooth(to_plot_array), label=labels[criterion]+"-"+labels[expl], color=colors[(criterion,expl)])
                


    plt.xlabel('Time step')
    plt.ylabel('Cost')

    plt.title(titles[dataset])

    if dataset in {'synthetic','breastcancer','diag'}:
        plt.legend(fontsize=46, ncols=1,loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(f'ODM_Results/weighted/{dataset}/Regret_sampledtheta_num_hypo_{num_hypos}_{dataset}_weighted.png', format='png', bbox_inches='tight')
    plt.close()

#plot utility
for dataset in datasets:
    prediction_progress_opt = pickle.load(open("TS"+"_weighted_sampled"+"odm_cost_dics_"+"IG"+"_"+dataset+".pkl", "rb" ))[2]
    num_hypos_list = list(prediction_progress_opt.keys())
    
    for criterion in criteria:
        for num_hypos in num_hypos_list:
            
            for expl in explorations[criterion]:
                prediction_all = pickle.load(open(expl+"_weighted_sampled"+"odm_cost_dics_"+criterion+"_"+dataset+".pkl", "rb" ))[2]
                temp = []
                preds = prediction_all[num_hypos][0]
                for (y, y_hat) in preds:
                    if y_hat == -1:
                        temp.append(0)
                    elif y_hat == y:
                        temp.append(2)
                    else:
                        temp.append(-1)

                to_plot_array = np.array(temp).reshape(num_rands,-1)
                to_plot_array_copy = copy.deepcopy(to_plot_array)
                to_plot_array = np.mean(to_plot_array, axis=0)
                if criterion in {'random','dpp','all'}:
                    plt.plot(exp_smooth(to_plot_array), label=labels[criterion], color=colors[(criterion,expl)])
                else:
                    plt.plot(exp_smooth(to_plot_array), label=labels[criterion]+"-"+labels[expl], color=colors[(criterion,expl)])

    plt.xlabel('Time step')
    plt.ylabel('Utility')

    plt.title(titles[dataset])

    if dataset in {'synthetic','breastcancer','diag'}:
        plt.legend(fontsize=46, ncols=2)
    plt.savefig(f'ODM_Results/weighted/{dataset}/utility_sampledtheta_num_hypo_{num_hypos}_{dataset}_weighted.png', format='png', bbox_inches='tight')
    plt.close()