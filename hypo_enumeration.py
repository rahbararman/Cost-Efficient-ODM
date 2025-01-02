import numpy as np
from utils import Hypothesis, binary, calculate_theta_mle
import copy
import time

def enumerate_hypotheses_for_all_decision_regions(thetas_mle, thetas_sampled, converge_thr, sampled_hypotheses_prev, max_hypo_per_region, frontiers=None):
    """Generates most likely hypotheses for all decision regions.

    Args:
        thetas_sampled: the sampled conditional probability table
        converge_thr: the parameter η
        sampled_hypotheses (optional): the previous samples of hypotheses (maybe a dictionary with decision regions as keys.). Defaults to None.
        frontiers (optional): the previous frontiers (maybe a dictionary with decision regions as keys.). Defaults to None.
    """
    num_classes = thetas_mle[0].shape[0]
    # print(sum(thetas_mle[0][0]))
    sampled_hypotheses = set()
    if frontiers is None:
        frontiers = {y: set() for y in range(num_classes)}
    give_ups = set()
    for y in range(num_classes):
        # print("sampling for region: ",y)
        if len(frontiers[y]) == 0:
            F_y_prev = None
        else:
            F_y_prev = frontiers[y]
        # print('mle')
        # print(thetas_mle)
        # print('sampled')
        # print(thetas_sampled)
        L_y, F_y = enumerate_hypotheses(
            y,
            copy.deepcopy(thetas_mle),
            copy.deepcopy(thetas_sampled),
            converge_thr,
            max_hypo_per_region,
            F_y_prev
        )
        
        frontiers[y] = F_y
        # for h in L_y.intersection(sampled_hypotheses):
        #     h.decision_region = -1
        give_ups = give_ups.union(L_y.intersection(sampled_hypotheses))

        sampled_hypotheses = sampled_hypotheses.union(L_y)
    for h in sampled_hypotheses:
        if h in give_ups:
            h.decision_region = -1
        
    
    sampled_hypotheses = sampled_hypotheses.union(sampled_hypotheses_prev)

    return sampled_hypotheses, frontiers

def enumerate_hypotheses(y, thetas_mle, thetas_sampled, converge_thr, max_hypo_per_region, frontier=None):
    """
    returns most likely hypotheses for a given decision region and the frontier
    Args:
        y: the decision region for which we want to generate most likely hypotheses
        thetas_mle: the (mle) conditional probability table
        thetas_sampled: the sampled conditional probability table
        converge_thr: the parameter η
        frontier: F_y
    """
    #Need to check if P[X_i=1|y]>=0.5 here ↓
    #keep a list of changed tests and change their probs (p_i and q_i) ↓
    #At the end flip the bits for changed tests.
    #step 1: sort in decreasing order (have to change some tests before this function to always have P[X_i=1|y]>=0.5)
    p_x_y = thetas_mle[0][y,:]
    flipped_tests = np.where(p_x_y<0.5)[0]
    p_x_y[flipped_tests] = 1 - p_x_y[flipped_tests]
    sorted_tests = np.argsort(p_x_y)[::-1]
    #step 2: calculate p_i and q_i
    p_i = np.log(p_x_y)[sorted_tests]
    q_i = np.log(1-p_x_y)[sorted_tests]

    #need to calculate log_probs for all decision regions with the sampled thetas!
    p_i_q_i_sampled_dict = {}
    for y_prime in range(thetas_mle[0].shape[0]):
        p_x_y_sampled = thetas_sampled[0][y_prime,:]
        p_x_y_sampled[flipped_tests] = 1 - p_x_y_sampled[flipped_tests]
        p_i_sampled = np.log(p_x_y_sampled)[sorted_tests]
        q_i_sampled = np.log(1-p_x_y_sampled)[sorted_tests]
        p_i_q_i_sampled_dict[y_prime] = {
            "p_i_sampled": p_i_sampled,
            "q_i_sampled": q_i_sampled
        }
    
    #step 3: populate F_y (if necessary)
    if frontier is None:
        frontier=set()
        h_1 = Hypothesis('1'*(len(p_i)))
        h_1.value = flip_test_values(h_1.value, flipped_tests)
        h_1.log_prob = sum(p_i)
        #for all y
        sampled_log_probs = {}
        for y_prime in range(thetas_mle[0].shape[0]):
            sampled_log_probs[y_prime] = sum(p_i_q_i_sampled_dict[y_prime]["p_i_sampled"])
        h_1.log_prob_sampled = sampled_log_probs
        h_1.decision_region = y
        frontier.add(h_1)
        L_y=set()
    sum_prob = 0
    for h in L_y:
        sum_prob += np.exp(h.log_prob)
    #step 4: generate hypotheses based on F_y
    while (sum_prob < 1 - converge_thr) and (len(L_y)<max_hypo_per_region):
        new_h = max(frontier, key=lambda h: h.log_prob)
        frontier.remove(new_h)
        L_y.add(new_h)
        sum_prob = 0
        for h in L_y:
            sum_prob += np.exp(h.log_prob)
        # print(sum_prob)
        #generate children for new_h
        # start_time = time.time()
        children = generate_children_for_hypo(new_h, sorted_tests, flipped_tests, p_i, q_i, p_i_q_i_sampled_dict, thetas_mle)
        # print("--- %s seconds:children ---" % (time.time() - start_time))
        for h_c in children:
            # start_time = time.time()
            # if h_c.value not in [h.value for h in frontier]:
            h_c.decision_region = y
            frontier.add(h_c)
            # print("--- %s seconds: check ---" % (time.time() - start_time))
    
    return L_y, frontier

def generate_children_for_hypo(h, sorted_tests, flipped_tests, p_i, q_i, p_i_q_i_sampled_dict, thetas_mle):
    # print(p_i_q_i_sampled_dict)
    children = []
    h_value_sorted = sort_string(flip_test_values(h.value, flipped_tests), sorted_tests)
    h_value_sorted_list = list(h_value_sorted)
    if h_value_sorted_list[-1] == '1':
        new_value = copy.deepcopy(h_value_sorted_list)
        new_value[-1] = '0'
        new_value = ''.join(new_value)
        new_value = restore_string(new_value, sorted_tests)
        new_log_prob = h.log_prob + q_i[-1] - p_i[-1]
        #for all y
        sampled_log_probs = {}
        for y_prime in range(thetas_mle[0].shape[0]):
            new_log_prob_sampled = h.log_prob_sampled[y_prime] +\
                  p_i_q_i_sampled_dict[y_prime]["q_i_sampled"][-1] -\
                      p_i_q_i_sampled_dict[y_prime]["p_i_sampled"][-1]
            sampled_log_probs[y_prime] = new_log_prob_sampled
        
        new_h = Hypothesis(flip_test_values(new_value, flipped_tests))
        new_h.log_prob = new_log_prob
        new_h.log_prob_sampled = sampled_log_probs
        children.append(new_h)
    
    right_most_index = h_value_sorted.rfind('10')
    if right_most_index != -1:
        new_value = copy.deepcopy(h_value_sorted_list)
        new_value[right_most_index] = '0'
        new_value[right_most_index + 1] = '1'
        new_value = ''.join(new_value)
        new_value = restore_string(new_value, sorted_tests)
        new_log_prob = h.log_prob + q_i[right_most_index] -\
              p_i[right_most_index] + p_i[right_most_index+1] -\
              q_i[right_most_index+1]
        #for all y
        sampled_log_probs = {}
        for y_prime in range(thetas_mle[0].shape[0]):
            new_log_prob_sampled = h.log_prob_sampled[y_prime] +\
                p_i_q_i_sampled_dict[y_prime]["q_i_sampled"][right_most_index] -\
                p_i_q_i_sampled_dict[y_prime]["p_i_sampled"][right_most_index] +\
                    p_i_q_i_sampled_dict[y_prime]["p_i_sampled"][right_most_index+1] -\
                        p_i_q_i_sampled_dict[y_prime]["q_i_sampled"][right_most_index+1]
            sampled_log_probs[y_prime] = new_log_prob_sampled
        new_h = Hypothesis(flip_test_values(new_value, flipped_tests))
        new_h.log_prob = new_log_prob
        new_h.log_prob_sampled = sampled_log_probs
        children.append(new_h)

    return children

def flip_test_values(s, indices):
    value_list = list(s)
    for i in indices:
        value_list[i] = str(1-int(value_list[i]))
    return ''.join(value_list)

def restore_string(s, indices):
    res = list(s)
    for i in enumerate(indices):
        res[i[1]] = s[i[0]]
    return ''.join(res)

def sort_string(s, indices):
    sorted = np.array(list(s))[indices]
    return ''.join(sorted)

def test():
    thetas = [np.array([[1,0,0.5,1], [0,0,1,1], [1,1,0,1], [0,0,1,1], [0,1,0,1]])]
    thetas_sampled = [np.array([[0.1,0.3,0.4,0.7], [0.2,0.3,0.3,0.2], [0.4,0.3,0.1, 0.9], [0.4,0.3,0.1, 0.9], [0.4,0.3,0.1, 0.9]])]
    samples, frontiers = enumerate_hypotheses_for_all_decision_regions(thetas, thetas_sampled, 0.002,set(),1,None)
    # for h in samples:
    #     print(h.value)
    #     print(h.decision_region)
    #     print(np.exp(h.log_prob))
    #     print({y:np.exp(h.log_prob_sampled[y]) for y in h.log_prob_sampled.keys()})
    #     print('***********************')
    # L_y, frontier = enumerate_hypotheses(1, thetas, thetas_sampled, 0.002, None)
    for h in samples:
        print(h.value)
        print(h.decision_region)
        print(np.exp(h.log_prob))
        print({y:np.exp(h.log_prob_sampled[y]) for y in h.log_prob_sampled.keys()})
        print('***********************')


if __name__ == "__main__":
    test()