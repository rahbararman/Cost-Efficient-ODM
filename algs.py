import copy
import random
import networkx as nx
from scipy.stats import beta
from dppy.finite_dpps import FiniteDPP
import numpy as np
from discretize import find_best_threshold_EC2, find_best_threshold_IG, find_best_threshold_US
from hypo_enumeration import enumerate_hypotheses_for_all_decision_regions

from utils import Hypothesis, calculate_expected_cut, calculate_expected_theta, calculate_expected_total_cost_for_tests, calculate_h_probs_with_observations, calculate_p_feature_xA, calculate_p_y_xA, calculate_p_y_xA_IG_neg, compute_initial_h_probs, estimate_priors_and_theta, find_inconsistent_hypotheses
epsilon = 0.1
min_epsilon = 0.01
decay_rate = 1.0

def EC2(exp_cost_per_outcome, thresholds,h_probs, document, hypotheses, thetas, priors, observations,ucb, t, params, G=None, epsilon=0.0):

    '''
    Return the next feature to be queried and the current graph
    Parameters:
        G: the graph
        h_probs: a dictionary containing h_indices as keys and p(h|x_A) as values
        document: a dictionary containing feature names as keys and features as values
        hypotheses: ndarray of Hypothesis objects
        decision_regions: two dimentional list. first dimension is decision region and the second is the list of hyptheses in that region
        thetas: the condictional probabilites. a list of m*n ndarrays where m is the number of decision regions and n is the number of features
        priors: prior probabilities of decision regions (ys)
        observations: a dictionary containing queried features as keys and  (thr_ind,value) as values.
    note: hypothesis names are the respective decimal number of the binary realization
    '''
    #building the graph if it is none
    if G is None:
        G = nx.Graph()
        for i in range(len(hypotheses)):
            for j in range(i+1, len(hypotheses)):
                if hypotheses[i].decision_region != hypotheses[j].decision_region:
                    G.add_edge(hypotheses[i].value, hypotheses[j].value, weight=h_probs[hypotheses[i].value]*h_probs[hypotheses[j].value])
                
    #select the feature
    best_feature = None
    max_cut = float('-inf')
    best_thr_ind = 0
    rand_number = random.uniform(0,1)
    exp_total_costs = calculate_expected_total_cost_for_tests(priors, exp_cost_per_outcome, thetas)
    if rand_number < epsilon:
        best_feature = np.random.choice(list(document.keys()))
        best_thr_ind = find_best_threshold_EC2(thetas, observations, best_feature, priors, G, hypotheses, thresholds, ucb=ucb, t=t, params=params)
        return best_feature, thresholds[best_thr_ind], best_thr_ind, G
    for feature in document.keys():
        # print('feature', feature)
        p_y_xA = calculate_p_y_xA(thetas, priors, observations, ucb=ucb, t=t, params=params)
        thr_ind = find_best_threshold_EC2(thetas, observations, feature, priors, G, hypotheses, thresholds, ucb=ucb, t=t, params=params)
        p_feature_xA =  calculate_p_feature_xA(feature, thetas, p_y_xA, (thr_ind,1), ucb=ucb, t=t, params=params)#P(x=1|x_A)
        p_not_feature_xA =  calculate_p_feature_xA(feature, thetas, p_y_xA, (thr_ind,0), ucb=ucb, t=t, params=params)#P(x=0|x_A)
        expected_cut = calculate_expected_cut(feature, p_feature_xA, p_not_feature_xA, G, hypotheses)
        expected_cut = expected_cut/exp_total_costs[feature]
        if (expected_cut > max_cut):
            max_cut = expected_cut
            best_feature = feature
            best_thr_ind = thr_ind
    return best_feature, thresholds[best_thr_ind], best_thr_ind, G

def random_feature_selection(thresholds, document):
    best_feature = np.random.choice(list(document.keys()))
    best_thr_ind = np.random.choice(len(thresholds))
    return best_feature, thresholds[best_thr_ind], best_thr_ind

def IG(exp_cost_per_outcome, thresholds, thetas, priors, observations, document, ucb, t, params, epsilon=0.0):
    exp_total_costs = calculate_expected_total_cost_for_tests(priors, exp_cost_per_outcome, thetas)
    if ucb:
        #step1: compute entropy(y|x_A)
        p_y_xA = calculate_p_y_xA_IG_neg(thetas, priors, observations, ucb, t, params)
        temp = p_y_xA * np.log2(p_y_xA)
        entropy_y_xA = -sum(temp)
        
        
        #step2: for all features x compute IG(x)
        best_feature = None
        best_feature = list(document.keys())[0]
        max_IG = float('-inf')
        best_thr_ind = 0
        rand_number = random.uniform(0,1)
        if rand_number <= epsilon:
            best_feature = np.random.choice(list(document.keys()))
            best_thr_ind = np.random.choice(len(thresholds))
            return best_feature, thresholds[best_thr_ind], best_thr_ind
        for feature in document.keys():
            #a. compute entropy(y|x_A,feature=1)
            # thr_ind = find_best_threshold_IG(thetas, observations, feature, priors, thresholds, p_y_xA, entropy_y_xA, ucb, t, params)
            thr_ind = 0
            new_observations = {feature:(thr_ind,1)}
            new_observations.update(observations)
            p_y_xA_feature = calculate_p_y_xA(thetas, priors, new_observations, ucb, t, params)
            temp = p_y_xA_feature * np.log2(p_y_xA_feature)
            entropy_y_xA_feature = -sum(temp)
            #b. compute entropy(y|x_A,feature=0)
            new_observations = {feature:(thr_ind,0)}
            new_observations.update(observations)
            p_y_xA_not_feature = calculate_p_y_xA(thetas, priors, new_observations, ucb, t, params)
            temp = p_y_xA_not_feature * np.log2(p_y_xA_not_feature)
            entropy_y_xA_not_feature = -sum(temp)
            #c. compute expected IG(feature)
            p_feature_xA =  calculate_p_feature_xA(feature, thetas, p_y_xA, (thr_ind, 1), ucb, t, params)#P(x=1|x_A)
            p_not_feature_xA =  calculate_p_feature_xA(feature, thetas, p_y_xA, (thr_ind, 0), ucb, t, params)#P(x=0|x_A)
            
            expected_IG = p_feature_xA*(entropy_y_xA-entropy_y_xA_feature)+p_not_feature_xA*(entropy_y_xA-entropy_y_xA_not_feature)
            expected_IG = expected_IG/exp_total_costs[feature]
            if (expected_IG > max_IG):
                max_IG = expected_IG
                best_feature = feature
                best_thr_ind = thr_ind
        return best_feature, thresholds[best_thr_ind], best_thr_ind
    else:
        #step1: compute entropy(y|x_A)
        p_y_xA = calculate_p_y_xA(thetas, priors, observations,ucb, t, params)
        temp = p_y_xA * np.log2(p_y_xA)
        entropy_y_xA = -sum(temp)
        
        
        #step2: for all features x compute IG(x)
        best_feature = None
        
        max_IG = float('-inf')
        best_thr_ind = 0
        rand_number = random.uniform(0,1)
        if rand_number <= epsilon:
            best_feature = np.random.choice(list(document.keys()))
            best_thr_ind = np.random.choice(len(thresholds))
            return best_feature, thresholds[best_thr_ind], best_thr_ind
        for feature in document.keys():
            # print('feature', feature)
            #a. compute entropy(y|x_A,feature=1)
            thr_ind = find_best_threshold_IG(thetas, observations, feature, priors, thresholds, p_y_xA, entropy_y_xA, ucb, t, params)
            new_observations = {feature:(thr_ind,1)}
            new_observations.update(observations)
            p_y_xA_feature = calculate_p_y_xA(thetas, priors, new_observations, ucb, t, params)
            temp = p_y_xA_feature * np.log2(p_y_xA_feature)
            entropy_y_xA_feature = -sum(temp)
            #b. compute entropy(y|x_A,feature=0)
            new_observations = {feature:(thr_ind,0)}
            new_observations.update(observations)
            p_y_xA_not_feature = calculate_p_y_xA(thetas, priors, new_observations,ucb, t, params)
            temp = p_y_xA_not_feature * np.log2(p_y_xA_not_feature)
            entropy_y_xA_not_feature = -sum(temp)
            #c. compute expected IG(feature)
            p_feature_xA =  calculate_p_feature_xA(feature, thetas, p_y_xA, (thr_ind, 1), ucb, t, params)#P(x=1|x_A)
            p_not_feature_xA =  calculate_p_feature_xA(feature, thetas, p_y_xA, (thr_ind, 0), ucb, t, params)#P(x=0|x_A)

            expected_IG = p_feature_xA*(entropy_y_xA-entropy_y_xA_feature)+p_not_feature_xA*(entropy_y_xA-entropy_y_xA_not_feature)
            expected_IG = expected_IG/exp_total_costs[feature]
            if (expected_IG > max_IG):
                max_IG = expected_IG
                best_feature = feature
                best_thr_ind = thr_ind
        return best_feature, thresholds[best_thr_ind], best_thr_ind

def all_features(document):
    return 

def US(theta_used_freq,thresholds, thetas, priors, observations, document, h_probs, hypothses):
    #step1: compute entropy(H|x_A)
    p_h_xA = np.array(list(h_probs.values()))
    temp = p_h_xA * np.log2(p_h_xA)
    entropy_h_xA = -sum(temp)
    
    #step2: for all features x compute US(x)
    best_feature = None
    max_US = float('-inf')
    best_thr_ind = 0
    for feature in document.keys():
        thr_ind = find_best_threshold_US(thetas, observations, feature, priors, thresholds, hypothses, theta_used_freq, entropy_h_xA)
        #a. compute entropy(h|x_A, feature=1)
        new_observations = {feature:(thr_ind,1)}
        new_observations.update(observations)
        h_probs = {}
        p_y_xA = calculate_p_y_xA(thetas, priors, new_observations)
        for h in hypothses:
            p_h_y = 1
            for feature_v, value in enumerate(h.value):
                expected_theta_feature_v = np.array([calculate_expected_theta(thetas, theta_used_freq, y_i, feature_v) for y_i in range(len(priors))])
                if int(value)==1:
                    p_h_y = p_h_y * expected_theta_feature_v
                else:
                    p_h_y = p_h_y * (1-expected_theta_feature_v)

            p_xA_y = 1
            for feature_v, (thr_ind,value) in new_observations.items():
                if int(value)==1:
                    p_xA_y = p_xA_y * thetas[thr_ind][:,int(feature_v)] 
                else:
                    p_xA_y = p_xA_y * (1-thetas[thr_ind][:,int(feature_v)])

            p_h_xA_y = p_h_y/p_xA_y

            p_xA = sum(priors*p_xA_y)
            p_y_xA = p_xA_y*priors/p_xA

            p_h_xA = sum(p_y_xA*p_h_xA_y)
            h_probs[h.value] = p_h_xA
            
        p_h_xA_feature = np.array(list(h_probs.values()))
        temp = p_h_xA_feature * np.log2(p_h_xA_feature)
        entropy_h_xA_feature = -sum(temp)
        
        
        #a. compute entropy(h|x_A, feature=0)
        new_observations = {feature:(thr_ind,0)}
        new_observations.update(observations)
        h_probs = {}
        p_y_xA = calculate_p_y_xA(thetas, priors, new_observations)
        for h in hypothses:
            p_h_y = 1
            for feature_v, value in enumerate(h.value):
                expected_theta_feature_v = np.array([calculate_expected_theta(thetas, theta_used_freq, y_i, feature_v) for y_i in range(len(priors))])
                if int(value)==1:
                    p_h_y = p_h_y * expected_theta_feature_v 
                else:
                    p_h_y = p_h_y * (1-expected_theta_feature_v)

            p_xA_y = 1
            for feature_v, (thr_ind,value) in new_observations.items():
                if int(value)==1:
                    p_xA_y = p_xA_y * thetas[thr_ind][:,int(feature_v)] 
                else:
                    p_xA_y = p_xA_y * (1-thetas[thr_ind][:,int(feature_v)])

            p_h_xA_y = p_h_y/p_xA_y

            p_xA = sum(priors*p_xA_y)
            p_y_xA = p_xA_y*priors/p_xA

            p_h_xA = sum(p_y_xA*p_h_xA_y)
            h_probs[h.value] = p_h_xA
        p_h_xA_not_feature = np.array(list(h_probs.values()))
        temp = p_h_xA_not_feature * np.log2(p_h_xA_not_feature)
        entropy_h_xA_not_feature = -sum(temp)
        
        
        
        #c. compute expected US(feature)
        p_y_xA = calculate_p_y_xA(thetas, priors, observations)
        p_feature_xA =  calculate_p_feature_xA(feature, thetas, p_y_xA, (thr_ind,1))#P(x=1|x_A)
        p_not_feature_xA =  calculate_p_feature_xA(feature, thetas, p_y_xA, (thr_ind,0))#P(x=0|x_A)
        
        expected_US = p_feature_xA*(entropy_h_xA-entropy_h_xA_feature)+p_not_feature_xA*(entropy_h_xA-entropy_h_xA_not_feature)
        if expected_US > max_US:
            max_US = expected_US
            best_feature = feature
            best_thr_ind = thr_ind
            
    return best_feature, thresholds[best_thr_ind], best_thr_ind



def decision_tree_learning(exp_cost_per_outcome, hypothses, thresholds,params, document, thetas, max_steps, priors, criterion, theta_used_freq, ucb, t, data=None, rand_state=None):
    '''
    Receives a document and builds a decision tree with the EC2 algorithm.
    Parameters:
        criterion: the method to choose next feature to be queried
        document: the document to be classified. A dictionary containing feature names as keys and features as values
        thetas: the condictional probabilites. a list of m*n ndarrays where m is the number of decision regions and n is the number of features
        max_steps: the maximum number of features to be queried
        priors: prior probabilities of decision regions (ys)
        hypothses: ndarray of Hypothesis objects
        decision regions
    '''
    #To use with enumeration
    #In each EC2(or IG/US) step:
        #step1: generate hypotheses for all decision regions (method call)
        #step2: calculate p(h|observations)
        #step3: EC2 for next step and observe test outcome
        #step4: update p(y|observation) for all y
        #step5: update log probs for all hypotheses
        #step6: Filter inconsistent hypotheses

    num_features = len(document.keys())
    num_labels = thetas[0].shape[0]
    

    
    # h_probs = compute_initial_h_probs(thetas, priors, hypothses, ucb=ucb, t=t, params=params) #using the naive bayes assumption and summing over all class labels
    observations = {}
    p_y_xA = calculate_p_y_xA(copy.deepcopy(thetas), priors, observations, ucb=ucb, t=t, params=params)
    h_probs = calculate_h_probs_with_observations(hypothses, p_y_xA)
    G = None
    document_label = document.pop('label', None)
    feature_names = list(document.keys())
    if criterion == 'dpp':
        L_cols = data.T.dot(data)
        dpp_cols = FiniteDPP(kernel_type='likelihood', **{'L': L_cols})
        dpp_features = dpp_cols.sample_exact(random_state=rand_state)
    for step in range(max_steps):
        # print('step:',step)
        if ('EC2' in criterion):
            #calculate h_probs using log_probs
            
            if (criterion == "EC2_epsgreedy"):
                feature_to_be_queried, thr, thr_ind, G = EC2(thresholds, h_probs,document,hypothses, thetas, priors, observations,ucb,t,params, G, epsilon)
            else:
                feature_to_be_queried, thr, thr_ind, G = EC2(exp_cost_per_outcome,
                                                             thresholds, 
                                                             h_probs,
                                                             document,
                                                             hypothses, 
                                                             copy.deepcopy(thetas), 
                                                             priors, 
                                                             observations,
                                                             ucb, 
                                                             t, 
                                                             params, 
                                                             G, 
                                                             0.0)
            #query the next feature.
            
            feature_value = document[feature_to_be_queried]
            feature_value = float(feature_value)
            if feature_value > thr:
                feature_value = 1
            else:
                feature_value = 0
            feature_value = int(float(feature_value))
            observations[feature_to_be_queried] = (thr_ind,(int(float(feature_value))))
            del document[feature_to_be_queried]
            
            #remove inconsistent hypotheses
            inconsistent_hypotheses = find_inconsistent_hypotheses(feature_to_be_queried, hypothses,feature_value)
            for inconsistenthypo in inconsistent_hypotheses:
                hypothses = [h for h in hypothses if h.value!=inconsistenthypo.value]
            
            #update p(h|x_A)
            #just update the log_probs for all h: subtract logP(x_t|y)
            p_x_t_y = 0
            if not ucb:
                if feature_value == 1:
                    p_x_t_y = copy.deepcopy(thetas[0][:, int(feature_to_be_queried)])
                else:
                    p_x_t_y = 1 - copy.deepcopy(thetas[0][:, int(feature_to_be_queried)])
            else:
                if feature_value==1:
                    p_x_t_y = beta.ppf(1/t, params[0][:,feature_to_be_queried,0], params[0][:,feature_to_be_queried,1])
                    if 0 in p_x_t_y:
                        p_x_t_y = p_x_t_y + 0.0000000001
                else:
                    p_x_t_y = 1 - beta.ppf(1-1/t, params[0][:,feature_to_be_queried,0], params[0][:,feature_to_be_queried,1])
                    if 0 in p_x_t_y:
                        p_x_t_y = p_x_t_y + 0.0000000001
            
            for h in hypothses:
                ys = h.log_prob_sampled.keys()
                for y_prime in ys:
                    # print('p_xt_y')
                    # print(p_x_t_y.shape)
                    # print(np.log(p_x_t_y[y_prime]))
                    h.log_prob_sampled[y_prime] = h.log_prob_sampled[y_prime] - np.log(p_x_t_y[y_prime])

            p_y_xA = calculate_p_y_xA(copy.deepcopy(thetas), priors, observations, ucb=ucb, t=t, params=params)
            h_probs = calculate_h_probs_with_observations(hypothses, p_y_xA)

            #update the graph
            G = nx.Graph()
            for i in range(len(hypothses)):
                for j in range(i+1, len(hypothses)):
                    if hypothses[i].decision_region != hypothses[j].decision_region:
                        G.add_edge(hypothses[i].value, hypothses[j].value, weight=h_probs[hypothses[i].value]*h_probs[hypothses[j].value])
            
            if len(G.edges) == 0:
                if len(hypothses) == 0:
                    print('here')
                    final_decision = -1
                else:
                    final_decision = hypothses[0].decision_region
                break
        if ("IG" in criterion):
            if (criterion == "IG_epsgreedy"):
                feature_to_be_queried, thr, thr_ind = IG(thresholds, thetas, priors, observations, document, 1.0)
            else:
                feature_to_be_queried, thr, thr_ind = IG(exp_cost_per_outcome,
                                                         thresholds, 
                                                         copy.deepcopy(thetas), 
                                                         priors, 
                                                         observations, 
                                                         document, 
                                                         ucb, 
                                                         t, 
                                                         params, 
                                                         0.0)
            feature_value = document[feature_to_be_queried]
            if feature_value > thr:
                feature_value = 1
            else:
                feature_value = 0
            feature_value = int(float(feature_value))
            inconsistent_hypotheses = find_inconsistent_hypotheses(feature_to_be_queried, hypothses,feature_value)
            for inconsistenthypo in inconsistent_hypotheses:
                hypothses = [h for h in hypothses if h.value!=inconsistenthypo.value]
            one_region = True
            if len(hypothses) == 0:
                final_decision = -1
                break
            re = hypothses[0].decision_region
            for hypo in hypothses:
                if hypo.decision_region != re:
                    one_region = False
                    break  
            observations[feature_to_be_queried] = (thr_ind,(int(float(feature_value))))
            del document[feature_to_be_queried]
            if one_region:
                final_decision = hypothses[0].decision_region
                break
        if (criterion == 'US'):
            feature_to_be_queried, thr, thr_ind = US(theta_used_freq, thresholds, thetas, priors, observations, document, h_probs, hypothses)
            feature_value = document[feature_to_be_queried]
            if feature_value > thr:
                feature_value = 1
            else:
                feature_value = 0
            feature_value = int(float(feature_value))
            inconsistent_hypotheses = find_inconsistent_hypotheses(feature_to_be_queried, hypothses,feature_value)
            for inconsistenthypo in inconsistent_hypotheses:
                hypothses = [h for h in hypothses if h.value!=inconsistenthypo.value]
            one_region = True
            if len(hypothses) == 0:
                break
            re = hypothses[0].decision_region
            for hypo in hypothses:
                if hypo.decision_region != re:
                    one_region = False
                    break  
            observations[feature_to_be_queried] = (thr_ind,(int(float(feature_value))))
            del document[feature_to_be_queried]
            if one_region:
                break
        
        if criterion == "random":
            feature_to_be_queried, thr, thr_ind = random_feature_selection(thresholds, document)
            feature_value = document[feature_to_be_queried]
            if feature_value > thr:
                feature_value = 1
            else:
                feature_value = 0
            feature_value = int(float(feature_value))
            inconsistent_hypotheses = find_inconsistent_hypotheses(feature_to_be_queried, hypothses,feature_value)
            for inconsistenthypo in inconsistent_hypotheses:
                hypothses = [h for h in hypothses if h.value!=inconsistenthypo.value]
            one_region = True
            if len(hypothses) == 0:
                final_decision = -1
                break
            re = hypothses[0].decision_region
            for hypo in hypothses:
                if hypo.decision_region != re:
                    one_region = False
                    break  
            observations[feature_to_be_queried] = (thr_ind,(int(float(feature_value))))
            del document[feature_to_be_queried]
            if one_region:
                final_decision = hypothses[0].decision_region
                break
        if criterion == 'all':
            feature_to_be_queried, thr, thr_ind = feature_names[step], thresholds[0], 0
            feature_value = document[feature_to_be_queried]
            if feature_value > thr:
                feature_value = 1
            else:
                feature_value = 0
            feature_value = int(float(feature_value))

            inconsistent_hypotheses = find_inconsistent_hypotheses(feature_to_be_queried, hypothses,feature_value)
            for inconsistenthypo in inconsistent_hypotheses:
                hypothses = [h for h in hypothses if h.value!=inconsistenthypo.value]
            if len(hypothses) == 0:
                final_decision = -1
            else:
                one_region = True
                re = hypothses[0].decision_region
                for hypo in hypothses:
                    if hypo.decision_region != re:
                        one_region = False
                        break  
                if one_region:
                    final_decision = hypothses[0].decision_region

            observations[feature_to_be_queried] = (thr_ind,(int(float(feature_value))))
            del document[feature_to_be_queried]
            
        
        if criterion == 'dpp':
            final_decision = None
            if step < len(dpp_features):
                feature_to_be_queried, thr, thr_ind = dpp_features[step], thresholds[0], 0
                feature_value = document[feature_to_be_queried]
                if feature_value > thr:
                    feature_value = 1
                else:
                    feature_value = 0
                feature_value = int(float(feature_value))
                inconsistent_hypotheses = find_inconsistent_hypotheses(feature_to_be_queried, hypothses,feature_value)
                for inconsistenthypo in inconsistent_hypotheses:
                    hypothses = [h for h in hypothses if h.value!=inconsistenthypo.value]
                if len(hypothses) == 0:
                    final_decision = -1
                else:
                    one_region = True
                    re = hypothses[0].decision_region
                    for hypo in hypothses:
                        if hypo.decision_region != re:
                            one_region = False
                            break  
                    if one_region:
                        final_decision = hypothses[0].decision_region

                observations[feature_to_be_queried] = (thr_ind,(int(float(feature_value))))
                del document[feature_to_be_queried]
            else:
                if final_decision is None:
                    final_decision = -1
                break
    
    
    #predict the label based on observations
    #y_hat = argmax_y p(y|observations)
    if ucb:
        p_ob_y = 1
        for feature, (thr_ind,value) in observations.items():
            if int(value)==1:
                q = beta.ppf(1-1/t, params[0][:,feature,0], params[0][:,feature,1])
                p_ob_y = p_ob_y * q 
            else:
                q = beta.ppf(1/t, params[0][:,feature,0], params[0][:,feature,1])
                p_ob_y = p_ob_y * (1-q)
        y_hat = np.argmax(priors*p_ob_y)  
    
    else:
        p_ob_y = 1
        for feature, (thr_ind,value) in observations.items():
            if int(value)==1:
                p_ob_y = p_ob_y * copy.deepcopy(thetas[thr_ind][:,int(feature)]) 
            else:
                p_ob_y = p_ob_y * (1-copy.deepcopy(thetas[thr_ind][:,int(feature)]))
        y_hat = np.argmax(priors*p_ob_y)  
    
    
    
    #observe the label
    y = int(document_label)
    y_hat = final_decision
    
    for feature, (thr_ind,value) in observations.items():
        theta_used_freq[y, feature, thr_ind] = theta_used_freq[y, feature, thr_ind] + 1
        if int(value)==1:
            params[thr_ind][int(y), int(feature), 0] += 1    
        else:
            params[thr_ind][int(y), int(feature), 1] += 1
        
    return observations, y, y_hat

def sample_hypotheses(N, thetas, priors, random_state, total_samples, theta_used_freq):
    #sampling hypotheses and generating decision regions
    #step1: sample y1,y2,...,yN from priors
    
    np.random.seed(random_state)
    num_features = thetas[0].shape[1]
    num_labels = thetas[0].shape[0]
    sampled_ys = []
    for n in range(N):
        y_n = np.random.choice(a = len(priors), p=priors)
        sampled_ys.append(y_n)
    #step2: sample h1,h2,...,hN from p(x|y)

    decision_regions = {}
    hypothses = []
    observed_hypothses = []
    for y_n in sampled_ys:
        #sample h
        while (True):
            sampled_h = ''
            for f in range(num_features):
                expected_theta_ij = calculate_expected_theta(thetas, theta_used_freq, y_n, f)
                generated_feature = np.random.choice(a=[0,1], p=[1-expected_theta_ij,expected_theta_ij])
                sampled_h = sampled_h + str(generated_feature)
            #determine region for sampled hypothesis
            #Decision region of h_i = argmax_j p(y_j|h_i) based on theta
            #1.compute p(y|sampled h) for all y
            p_h_y = 1
            for feature, value in enumerate(sampled_h):
                value = int(value)
                expected_theta_feature = np.array([calculate_expected_theta(thetas, theta_used_freq, y_i, feature) for y_i in range(num_labels)])
                if value==1:
                    p_h_y = p_h_y * expected_theta_feature
                else:
                    p_h_y = p_h_y * (1-expected_theta_feature)

            region = np.argmax(priors*p_h_y)
            if not (sampled_h in observed_hypothses):
                observed_hypothses.append(sampled_h)
                new_h = Hypothesis(sampled_h)
                hypothses.append(new_h)
                new_h.decision_region = region

                if not (region in decision_regions.keys()):
                    decision_regions[region] = set()
                    decision_regions[region].add(sampled_h)
                else:
                    decision_regions[region].add(sampled_h)
                break
    return hypothses, decision_regions





