import argparse, sys
import random
import copy

import numpy as np
from sklearn.metrics import accuracy_score
from algs import decision_tree_learning, sample_hypotheses
from skmultiflow.trees import ExtremelyFastDecisionTreeClassifier
import matplotlib.pyplot as plt
from hypo_enumeration import enumerate_hypotheses_for_all_decision_regions

from utils import calculate_cost_of_observation, calculate_map_estimate_theta, calculate_performance, calculate_theta_mle, calculate_total_accuracy, create_correct_hypo, create_dataset_for_efdt_vfdt, create_expected_cost_for_test_outcomes, estimate_priors_and_theta
import pickle

#parse arguments
parser=argparse.ArgumentParser()

parser.add_argument('--rand', default=130)
parser.add_argument('--dataset', default='fico')
parser.add_argument('--minhypo', default=50) #min number of enumerated hypo per y
parser.add_argument('--maxhypo', default=100) #max number of enumerated hypo per y
parser.add_argument('--hypostep', default=50)
parser.add_argument('--thresholds', default=1)
parser.add_argument('--criterion', default='IG')
parser.add_argument('--numrands', default=5)
parser.add_argument('--exploration', default='UCB')
parser.add_argument('--opt', default='y')

args=parser.parse_args()



def main():
    opt = args.opt == 'y'
    print('opt:', opt)
    if not opt:
        #agent
        num_rands = int(args.numrands)
        random_states = list(range(int(args.rand), int(args.rand)+num_rands))
        dataset = args.dataset
        
        exploration = args.exploration
        ucb = exploration == "UCB"
        
        print('ucb:', ucb)
        criterion = args.criterion
        
        hypotheses_mle = None
        if int(args.thresholds)>1:
            thresholds = list(np.linspace(0.1,0.9,int(args.thresholds)))
        else:
            thresholds = [0.5]
        min_num_hypotheses = int(args.minhypo)
        max_num_hypotheses = int(args.maxhypo)
        hypotheses_step = int(args.hypostep)

        cost_progress = {}
        num_tests_progress = {}
        prediction_progress = {}

        for num_sampled_hypos in range(min_num_hypotheses, max_num_hypotheses, hypotheses_step):
            cost_in_progress = [[]]
            num_tests_in_progress = [[]]
            pred_in_progress = [[]]
            for rand_state in random_states:
                thetas_mle = calculate_theta_mle(rand_state, dataset)
                exp_cost_per_outcome = create_expected_cost_for_test_outcomes(thetas_mle[0].shape[1], dataset)
                print('feature costs:')
                print(exp_cost_per_outcome)
                np.random.seed(rand_state)
                random.seed(rand_state)
                print('random state = '+ str(rand_state))
                params, thetas, priors, test_csv, data_csv, theta_used_freq = estimate_priors_and_theta(dataset, rand_state=rand_state, num_thresholds=len(thresholds)) 
                print("priors")
                print(priors)
                print(sum(priors))
                if len(cost_in_progress)==1:
                    cost_in_progress = cost_in_progress * len(test_csv)
                    num_tests_in_progress = num_tests_in_progress * len(test_csv)
                    pred_in_progress = pred_in_progress * len(test_csv)
                
                # hypothses, decision_regions = sample_hypotheses(N=num_sampled_hypos, thetas=thetas, priors=priors, random_state=rand_state, total_samples=num_sampled_hypos, theta_used_freq=theta_used_freq)
                
                hypotheses_mle, frontiers = enumerate_hypotheses_for_all_decision_regions(copy.deepcopy(thetas), copy.deepcopy(thetas), 0.002,set(),num_sampled_hypos,None)
                print('len mle', len(hypotheses_mle))
                hypotheses_mle = list(hypotheses_mle)
                hypothses = copy.deepcopy(hypotheses_mle)

                print('number of sampled hypo is:', len(hypothses))

                print('Experimenting with ' + criterion)
                max_steps = test_csv.shape[1]-1
                print('max steps = '+ str(max_steps))
                print('number of data points')
                print(len(test_csv))
                for i in range(len(test_csv)):
                    if i%1 == 0:
                        print(i)
                    doc = test_csv.iloc[i].to_dict()
                    h_true = create_correct_hypo(
                        test_csv.iloc[i].values[:-1],
                        test_csv.iloc[i].values[-1],
                        copy.deepcopy(thetas),
                        copy.deepcopy(thetas),thresholds[0])
                    if not (h_true in set(hypothses)):
                        hypothses.append(h_true)

                    # print(test_csv.iloc[i].values[:-1])
                    data = None
                    if criterion == 'dpp':
                        data = copy.deepcopy(test_csv.values[:i+1, :-1])
                    obs, y, y_hat = decision_tree_learning(exp_cost_per_outcome,
                                                           hypothses,
                                                           thresholds,
                                                           params,
                                                           doc,
                                                           copy.deepcopy(thetas),
                                                           max_steps, 
                                                           priors,
                                                           criterion,
                                                           theta_used_freq, 
                                                           ucb=ucb, 
                                                           t=i+1, 
                                                           data=data, 
                                                           rand_state=rand_state)
                    # cost = None
                    # print(y,y_hat)
                    # print(priors.shape)
                    # if y != y_hat:
                    #     cost = test_csv.shape[1]*3
                    # else:
                    #     cost = len(obs.items())
                    cost = calculate_cost_of_observation(
                        exp_cost_per_outcome, obs, y, copy.deepcopy(thetas_mle))
                    # print(exp_cost_per_outcome)
                    # print('cost:',len(obs))
                    # print(y, y_hat)
                    cost_in_progress[i].append(cost)
                    num_tests_in_progress[i].append(len(obs.items()))
                    pred_in_progress[i].append((y,y_hat))
                    
                    thetas = []
                    thetas.append(np.random.beta(params[0][:,:,0], params[0][:,:,1]))
                    
                    # if not (criterion == 'all' or criterion == 'dpp'):
                    # hypothses, decision_regions = sample_hypotheses(N=num_sampled_hypos, thetas=thetas, priors=priors, random_state=rand_state, total_samples=num_sampled_hypos, theta_used_freq=theta_used_freq)
                    hypotheses_mle, frontiers = enumerate_hypotheses_for_all_decision_regions(copy.deepcopy(thetas), copy.deepcopy(thetas), 0.002,set(),num_sampled_hypos,None)
                    hypotheses_mle = list(hypotheses_mle)
                    hypothses = copy.deepcopy(hypotheses_mle)
                    # print(len(hypothses))

            
            cost_progress[num_sampled_hypos] = cost_in_progress
            num_tests_progress[num_sampled_hypos] = num_tests_in_progress
            prediction_progress[num_sampled_hypos] = pred_in_progress

        to_save = [cost_progress, num_tests_progress, prediction_progress]
        f = open(exploration+"_weighted_sampled"+"odm_cost_dics_"+criterion+"_"+dataset+".pkl", "wb")
        pickle.dump(to_save,f)
        f.close()
    
    if opt:
        #opt_agent
        num_rands = int(args.numrands)
        random_states = list(range(int(args.rand), int(args.rand)+num_rands))
        dataset = args.dataset
        exploration = args.exploration
        ucb = exploration == "UCB"
        print('ucb:', ucb)
        criterion = args.criterion
        if int(args.thresholds)>1:
            thresholds = list(np.linspace(0.1,0.9,int(args.thresholds)))
        else:
            thresholds = [0.5]
        min_num_hypotheses = int(args.minhypo)
        max_num_hypotheses = int(args.maxhypo)
        hypotheses_step = int(args.hypostep)

        cost_progress_opt = {}
        prediction_progress_opt = {}

        for num_sampled_hypos in range(min_num_hypotheses, max_num_hypotheses, hypotheses_step):
            cost_in_progress = [[]]
            pred_in_progress = [[]]
            for rand_state in random_states:
                np.random.seed(rand_state)
                random.seed(rand_state)
                print('random state for opt = '+ str(rand_state))
                params, thetas, priors, test_csv, data_csv, theta_used_freq = estimate_priors_and_theta(dataset, rand_state=rand_state, num_thresholds=len(thresholds)) 
                print("priors")
                print(priors)
                print(sum(priors))
                thetas_mle = calculate_theta_mle(rand_state, dataset)
                exp_cost_per_outcome = create_expected_cost_for_test_outcomes(thetas_mle[0].shape[1], dataset)
                print('feature costs:')
                print(exp_cost_per_outcome) #ilavvvvm
                print(len(thetas_mle))
                
                if len(cost_in_progress)==1:
                    cost_in_progress = cost_in_progress * len(test_csv)
                    pred_in_progress = pred_in_progress * len(test_csv)
                # hypothses, decision_regions = sample_hypotheses(N=num_sampled_hypos, thetas=theta_mle, priors=priors, random_state=rand_state, total_samples=num_sampled_hypos, theta_used_freq=theta_used_freq)
                hypotheses_mle, frontiers = enumerate_hypotheses_for_all_decision_regions(copy.deepcopy(thetas_mle), copy.deepcopy(thetas_mle), 0.002,set(),num_sampled_hypos,None)
                print('len mle', len(hypotheses_mle))
                hypotheses_mle = list(hypotheses_mle)
                hypothses = copy.deepcopy(hypotheses_mle)
                print('sampled')
                print('Experimenting with ' + criterion)
                max_steps = test_csv.shape[1]-1
                print('max steps = '+ str(max_steps))
                print('number of data points')
                print(len(test_csv))
                for i in range(len(test_csv)):
                    if i%1000 == 0:
                        print(i)
                    doc = test_csv.iloc[i].to_dict()
                    h_true = create_correct_hypo(
                        test_csv.iloc[i].values[:-1],
                        test_csv.iloc[i].values[-1],
                        copy.deepcopy(thetas_mle),
                        copy.deepcopy(thetas))
                    if not (h_true in set(hypothses)):
                        hypothses.append(h_true)
                    obs, y, y_hat = decision_tree_learning(exp_cost_per_outcome,
                                                           hypothses,
                                                           thresholds,
                                                           params,
                                                           doc,
                                                           copy.deepcopy(thetas_mle),
                                                           max_steps, 
                                                           priors, 
                                                           criterion, 
                                                           theta_used_freq, 
                                                           ucb=False, 
                                                           t=i+1)
                    # cost = None
                    # print(y,y_hat)
                    # if y != y_hat:
                    #     cost = test_csv.shape[1]*3
                    # else:
                    #     cost = len(obs.items())
                    cost = calculate_cost_of_observation(
                        exp_cost_per_outcome, obs, y, copy.deepcopy(thetas_mle))
                    # print('cost')
                    # print(cost, y_hat)
                    cost_in_progress[i].append(cost)
                    pred_in_progress[i].append((y,y_hat))

                    # hypothses, decision_regions = sample_hypotheses(N=num_sampled_hypos, thetas=theta_mle, priors=priors, random_state=rand_state, total_samples=num_sampled_hypos, theta_used_freq=theta_used_freq)
                    hypothses = copy.deepcopy(hypotheses_mle)

            
            cost_progress_opt[num_sampled_hypos] = cost_in_progress
            prediction_progress_opt[num_sampled_hypos] = pred_in_progress

        to_save = [cost_progress_opt, prediction_progress_opt]
        f = open(exploration+"_weighted_"+"odm_opt_cost_dics_"+criterion+"_"+dataset+".pkl", "wb")
        pickle.dump(to_save,f)
        f.close()

if __name__=="__main__":
    main()
