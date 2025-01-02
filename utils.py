import copy
import random
from matplotlib.pyplot import thetagrids
import numpy as np
from scipy.stats import beta
import networkx as nx
import pandas as pd
import pickle
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, minmax_scale

class Hypothesis:
    def __init__(self, value, is_active=True, decision_region=None):
        self.value = value
        self.is_active = is_active
        self.decision_region = decision_region

    def __eq__(self, other):
        return self.value == other.value
    
    def __hash__(self):
        return hash(self.value)


def calculate_h_probs_with_observations(hypothses, p_y_xA):
    #TODO: UCB? how to integrate with enumeration?
    # print(p_y_xA.shape)
    h_probs = {}
    for h in hypothses:
        log_probs = h.log_prob_sampled
        p_h_xA = np.sum([np.exp(log_probs[y])*p_y_xA[y] for y in log_probs.keys()])
        h_probs[h.value] = p_h_xA
    return h_probs


def binary(num, length):
    '''
        creates a name for each hypotheses
    '''
    return format(num, '#0{}b'.format(length + 2))[2:]



def compute_initial_h_probs(thetas, priors, hypothses, ucb, t, params):
    '''
    return a dictionary containing probabilities of each hypothesis (p(h)) with h's as keys
    parameters:
        thetas: the condictional probabilites. a list of m*n ndarrays where m is the number of decision regions and n is the number of features
        priors: prior probabilities of decision regions (ys)
        hypothses: ndarray containing all hypotheses (objects) 
    '''
    if not ucb:
        probs = {}
        for h in hypothses:
            p_h_y = 1
            for feature, value in enumerate(h.value):
                if int(float(value))==1:
                    p_h_y = p_h_y * thetas[0][:,feature] 
                else:
                    p_h_y = p_h_y * (1-thetas[0][:,feature])
            
            p_h = sum(priors*p_h_y)
            probs[h.value] = p_h
        return probs
    else:
        probs = {}
        for h in hypothses:
            p_h_y = 1
            for feature, value in enumerate(h.value):
                if int(float(value))==1:
                    q = beta.ppf(1-1/t, params[0][:,feature,0], params[0][:,feature,1])
                    p_h_y = p_h_y * q 
                else:
                    q = beta.ppf(1/t, params[0][:,feature,0], params[0][:,feature,1])
                    p_h_y = p_h_y * (1-q)
            
            p_h = sum(priors*p_h_y)
            probs[h.value] = p_h
        return probs



def find_inconsistent_hypotheses(feature, hypotheses, feature_value):
    #checked
    '''
    returns a list of hypothesis inconsistent with feature observation (feature values)
    parameters:
        feature: the observed feature
        hypotheses: the list of hypotheses (objects)
        feature_value: observed feature value
    '''
    inconsistent_hypotheses = []
    for h in hypotheses:
        if str(h.value)[int(feature)] != str(int(feature_value)):
            inconsistent_hypotheses.append(h)
#     print([h.value for h in inconsistent_hypotheses])
    return inconsistent_hypotheses

def calculate_p_y_xA_IG_neg(thetas, priors, observations, ucb, t, params):
    #checked
    '''
    returns a ndarray containing p(y|x_A) for all ys
    parameters:
        thetas: the condictional probabilites. a list of m*n ndarrays where m is the number of decision regions and n is the number of features
        priors: prior probabilities of decision regions (ys)
        observations: a dictionary containing queried features as keys and  (thr_ind,value) as values.
    '''
    if not ucb:
        if (len(observations.items())==0):
            return priors
        #calculate p_xA
        p_xA_y = 1
        for feature, (thr_ind,value) in observations.items():
            if int(value)==1:
                p_xA_y = p_xA_y * thetas[thr_ind][:,int(feature)] 
            else:
                p_xA_y = p_xA_y * (1-thetas[thr_ind][:,int(feature)])
                
        
        p_xA = sum(priors*p_xA_y)
        p_y_xA = p_xA_y*priors/p_xA
        
    #     print(p_y_xA)
        return p_y_xA
    else:
        if (len(observations.items())==0):
            return priors
        #calculate p_xA
        p_xA_y = 1
        for feature, (thr_ind,value) in observations.items():
            if int(value)==1:
                q = beta.ppf(1/t, params[0][:,feature,0], params[0][:,feature,1])
                if 0 in q:
                    q = q + 0.000001
                p_xA_y = p_xA_y * q 
            else:
                q = beta.ppf(1-1/t, params[0][:,feature,0], params[0][:,feature,1])
                if 1 in q:
                    q = q - 0.000001
                p_xA_y = p_xA_y * (1-q)
                
        p_xA_y_denom = 1
        for feature, (thr_ind,value) in observations.items():
            if int(value)==1:
                q = beta.ppf(1-1/t, params[0][:,feature,0], params[0][:,feature,1])
                if 0 in q:
                    q = q + 0.000001
                p_xA_y_denom = p_xA_y_denom * q 
            else:
                q = beta.ppf(1/t, params[0][:,feature,0], params[0][:,feature,1])
                if 1 in q:
                    q = q - 0.000001
                p_xA_y_denom = p_xA_y_denom * (1-q)
        p_xA = sum(priors*p_xA_y_denom)
        p_y_xA = p_xA_y*priors/p_xA
        
    #     print(p_y_xA)
        return p_y_xA

def calculate_p_y_xA(thetas, priors, observations, ucb, t, params):
    #checked
    '''
    returns a ndarray containing p(y|x_A) for all ys
    parameters:
        thetas: the condictional probabilites. a list of m*n ndarrays where m is the number of decision regions and n is the number of features
        priors: prior probabilities of decision regions (ys)
        observations: a dictionary containing queried features as keys and  (thr_ind,value) as values.
    '''
    if not ucb:
        if (len(observations.items())==0):
            return priors
        #calculate p_xA
        p_xA_y = 1
        for feature, (thr_ind,value) in observations.items():
            if int(value)==1:
                p_xA_y = p_xA_y * thetas[thr_ind][:,int(feature)]
            else:
                p_xA_y = p_xA_y * (1-thetas[thr_ind][:,int(feature)])
        
        if 0 in p_xA_y:
            p_xA_y = p_xA_y + 0.000001
                
        
        p_xA = sum(priors*p_xA_y)
        p_y_xA = p_xA_y*priors/p_xA
        
    #     print(p_y_xA)
        return p_y_xA
    else:
        if (len(observations.items())==0):
            return priors
        #calculate p_xA
        p_xA_y = 1
        for feature, (thr_ind,value) in observations.items():
            if int(value)==1:
                q = beta.ppf(1-1/t, params[0][:,feature,0], params[0][:,feature,1])
                if 0 in q:
                    q = q + 0.000001
                p_xA_y = p_xA_y * q 
            else:
                q = beta.ppf(1/t, params[0][:,feature,0], params[0][:,feature,1])
                if 1 in q:
                    q = q - 0.000001
                p_xA_y = p_xA_y * (1-q)
                
        p_xA_y_denom = 1
        for feature, (thr_ind,value) in observations.items():
            if int(value)==1:
                q = beta.ppf(1/t, params[0][:,feature,0], params[0][:,feature,1])
                if 0 in q:
                    q = q + 0.000001
                p_xA_y_denom = p_xA_y_denom * q 
            else:
                q = beta.ppf(1-1/t, params[0][:,feature,0], params[0][:,feature,1])
                if 1 in q:
                    q = q - 0.000001
                p_xA_y_denom = p_xA_y_denom * (1-q)
        p_xA = sum(priors*p_xA_y_denom)
        p_y_xA = p_xA_y*priors/p_xA
        
    #     print(p_y_xA)
        return p_y_xA


def calculate_p_feature_xA(feature, thetas, p_y_xA, feature_value, ucb, t, params):
    #checked
    '''
    parameters:
        thetas: the condictional probabilites. a list of m*n ndarrays where m is the number of decision regions and n is the number of features
        priors: prior probabilities of decision regions (ys)
        feature_value: (thr_ind, value) of the feature. 1 or 0
    '''
    if not ucb:
        if int(feature_value[1]) == 1:
            p_x_y = thetas[feature_value[0]][:, int(feature)]    
        else:
            p_x_y = 1 - thetas[feature_value[0]][:,int(feature)]
            
        p_feature_xA = sum(p_x_y*p_y_xA)
        
        return p_feature_xA
    else:
        if int(feature_value[1]) == 1:
            q = beta.ppf(1-1/t, params[0][:,feature,0], params[0][:,feature,1])
            p_x_y = q   
        else:
            q = beta.ppf(1/t, params[0][:,feature,0], params[0][:,feature,1])
            p_x_y = 1 - q
            
        p_feature_xA = sum(p_x_y*p_y_xA)
        
        return p_feature_xA

def calculate_expected_cut(feature,p_feature_xA, p_not_feature_xA, G, hypotheses):
    #checked
    #Need a proper way to find respective hypothesis
    '''
    parameters:
        hypotheses: the list of hypotheses (objects)
        p_feature_xA: P(x=1|x_A)
        p_not_feature_xA: P(x=0|x_A)
        G: the graph of hypotheses
    '''
    
    #step1: find the hypotheses inconsistent with feature
    hypotheses_feature = find_inconsistent_hypotheses(feature, hypotheses, 1)
    
    #step2: find the hypotheses inconsistent with not feature
    hypotheses_not_feature = find_inconsistent_hypotheses(feature, hypotheses, 0)
    
    #step3: Calculate the weights for each case
#     print(G.edges(data=True))
    edges_feature = G.edges(nbunch=[h.value for h in hypotheses_feature], data=True)
    sum_weights_feature = sum([w['weight'] for (u,v,w) in edges_feature])
    
    edges_not_feature = G.edges(nbunch=[h.value for h in hypotheses_not_feature], data=True)
    sum_weights_not_feature = sum([w['weight'] for (u,v,w) in edges_not_feature])
    
    #step4: Calculate the expectation
    expected_cut = p_feature_xA * sum_weights_feature + p_not_feature_xA *sum_weights_not_feature
    return expected_cut

def calculate_total_accuracy(thetas, thresholds, data, priors, theta_used_freq, metric='accuracy'):
    y_pred = []
    y_true = []
    
    for i in range(len(data)):
        # sampled_theta_ind = random.choice(range(len(thetas)))
        doc = data.iloc[i].to_dict()
        document_label = doc.pop('label', None)
        p_ob_y = 1
        for feature, value in doc.items():
            feature = int(float(feature))
            freqs_sum = np.sum(theta_used_freq, axis=0)
            thr_ind = np.argmax(freqs_sum[feature])
            if value > thresholds[thr_ind]:
                value = 1
            else:
                value = 0
            value = int(float(value))

            if value == 1:
                p_ob_y = p_ob_y * thetas[thr_ind][:,int(feature)]
            else:
                p_ob_y = p_ob_y * (1-thetas[thr_ind][:,int(feature)])
        y_pred.append(np.argmax(priors*p_ob_y))
        y_true.append(document_label)
    perf = 0.0
    if metric == 'accuracy':
        perf = accuracy_score(y_true, y_pred)
    if metric == 'fscore':
        perf = f1_score(y_true, y_pred, average='weighted')
    return perf


def estimate_priors_and_theta(dataset, rand_state, num_thresholds=1):

    if dataset == 'zoo':
        zoo_data = pd.read_csv('zoo.csv')
        labels = pd.DataFrame(zoo_data['class_type']-1)
        features = list(zoo_data.columns)

        features.remove('class_type')
        features.remove('animal_name')
        features.remove('legs')

        zoo_data = pd.concat([zoo_data[features], labels], axis=1)
        zoo_data.columns = list(range(len(zoo_data.columns)-1))+['label']
        X_train, X_test, y_train, y_test = train_test_split(zoo_data.iloc[:, :-1], zoo_data['label'],
                                                           test_size=0.8, random_state=rand_state)
    
    if dataset == 'led':
        data_csv = pd.read_csv('data_big.csv',header=None)
        data_csv.columns = [0, 1, 2, 3, 4, 5, 6, 'label']
        X_train, X_test, y_train, y_test = train_test_split(data_csv[[0, 1, 2, 3, 4, 5, 6]], data_csv['label'], test_size=0.5, random_state=rand_state)

    if dataset == 'heart':
        heart_data = pd.read_csv('heart.csv')
        labels = pd.DataFrame(heart_data['OVERALL_DIAGNOSIS'])
        features = list(heart_data.columns)
        features.remove('OVERALL_DIAGNOSIS')

        heart_data = pd.concat([heart_data[features], labels], axis=1)
        heart_data.columns = list(range(len(heart_data.columns)-1))+['label']
        X_train, X_test, y_train, y_test = train_test_split(heart_data.iloc[:,:-1], heart_data['label'], test_size=0.9, random_state=rand_state)
        
    if dataset == 'tumor':
        df = pd.read_csv('primary-tumor.data.csv',header=None)
        data = df.replace("?", np.NaN)
        data = data.dropna()
        data = data.apply(pd.to_numeric)
        data = data.apply(lambda x: x-1)
        data = data.drop([1, 3, 4],axis=1)
        data.columns = ['label'] + list(range(len(data.columns)-1)) 
        encoder = LabelEncoder()
        encoder.fit(data['label'])
        data['label'] = encoder.transform(data['label'])
        X_train, y_train = data.iloc[:,1:], data['label']
        X_test, y_test = data.iloc[:,1:], data['label']
    
    if dataset == 'synthetic':
        syn_data = pd.read_csv(f'synthetic_odt_data/{rand_state%10}.csv').drop('index', axis=1)
        labels = pd.DataFrame(syn_data['y'])
        features = list(syn_data.columns)
        features.remove('y')
        syn_data = pd.concat([syn_data[features], labels], axis=1)
        syn_data.columns = list(range(len(syn_data.columns)-1))+['label']
        X_train, X_test, y_train, y_test = train_test_split(syn_data.iloc[:, :-1], syn_data['label'],
                                                            test_size=0.1, random_state=rand_state)

    if dataset == 'diabetes':
        data = pd.read_csv('pima-indians-diabetes.csv', header=None)
        data.columns = list(range(len(data.columns)-1)) + ['label']
        np_from_data = data.to_numpy()
        np_from_data[:, :-1] = minmax_scale(np_from_data[:, :-1])
        data_new = pd.DataFrame(np_from_data, columns=data.columns)
        X_train, X_test, y_train, y_test = train_test_split(data_new.iloc[:,:-1], data_new['label'], test_size=0.8, random_state=rand_state)
    
    if dataset == 'breastcancer':
        data = pd.read_csv('breast-cancer.csv')
        data.drop('id', inplace=True, axis=1)
        data.columns = ['label'] + list(range(len(data.columns)-1)) 
        encoder = LabelEncoder()
        encoder.fit(data['label'])
        data['label'] = encoder.transform(data['label'])
        np_from_data = data.to_numpy()
        np_from_data[:, 1:] = minmax_scale(np_from_data[:, 1:])
        data_new = pd.DataFrame(np_from_data, columns=data.columns)
        X_train, X_test, y_train, y_test = train_test_split(data_new.iloc[:,1:], data_new['label'], test_size=0.8, random_state=rand_state)

    if dataset== 'fetal':
        data = pd.read_csv('fetal_health.csv')
        data.columns = list(range(len(data.columns)-1)) + ['label']
        encoder = LabelEncoder()
        encoder.fit(data['label'])
        data['label'] = encoder.transform(data['label'])
        np_from_data = data.to_numpy()
        np_from_data[:, :-1] = minmax_scale(np_from_data[:, :-1])
        data_new = pd.DataFrame(np_from_data, columns=data.columns)
        X_train, X_test, y_train, y_test = train_test_split(data_new.iloc[:,:-1], data_new['label'], test_size=0.05, random_state=rand_state)

    if dataset == 'fico':
        data = pd.read_csv("fico_binary.csv.train2.csv", delimiter=';')
        data.columns = list(range(len(data.columns)-1)) + ['label']
        np_from_data = data.to_numpy()
        data_new = pd.DataFrame(np_from_data, columns=data.columns)
        X_train, X_test, y_train, y_test = train_test_split(data_new.iloc[:,:-1], data_new['label'], test_size=0.9, random_state=rand_state)
        # data = pd.read_csv("fico_binary.csv.train2.csv", delimiter=';')
        # data.columns = list(range(len(data.columns)-1)) + ['label']
        # np_from_data = data.to_numpy()
        # data_new = pd.DataFrame(np_from_data, columns=data.columns)
        # X_test, y_test = data_new.iloc[:,:-1], data_new['label']

        # data = pd.read_csv("fico_binary.csv.test2.csv", delimiter=';')
        # data.columns = list(range(len(data.columns)-1)) + ['label']
        # np_from_data = data.to_numpy()
        # data_new = pd.DataFrame(np_from_data, columns=data.columns)
        # X_train, y_train = data_new.iloc[:,:-1], data_new['label']

    if dataset == 'compas':
        data = pd.read_csv("compas-binary.csv.train5.csv", delimiter=';')
        data.columns = list(range(len(data.columns)-1)) + ['label']
        np_from_data = data.to_numpy()
        data_new = pd.DataFrame(np_from_data, columns=data.columns)
        X_train, X_test, y_train, y_test = train_test_split(data_new.iloc[:,:-1], data_new['label'], test_size=0.9, random_state=rand_state)

        # X_test, y_test = data_new.iloc[:,:-1], data_new['label']

        # data = pd.read_csv("compas-binary.csv.test5.csv", delimiter=';')
        # data.columns = list(range(len(data.columns)-1)) + ['label']
        # np_from_data = data.to_numpy()
        # data_new = pd.DataFrame(np_from_data, columns=data.columns)
        # X_train, y_train = data_new.iloc[:,:-1], data_new['label']

    if dataset == 'diag':
        numpy_data = pickle.load(open("scenarios_processed.pkl","rb"))
        diag_data = pd.DataFrame(numpy_data)
        labels = pd.DataFrame(diag_data[940])
        features = list(diag_data.columns)
        features.remove(940)
        diag_data.columns = list(range(len(diag_data.columns)-1))+['label']
        ###################
        diag_data = diag_data[diag_data['label'].isin(list(range(15)))]
        for c in diag_data.columns:
            if sum(diag_data[c]) == 0:
                diag_data = diag_data.drop([c], axis=1)
        diag_data.columns = list(range(len(diag_data.columns)-1))+['label']
        diag_data = pd.concat([diag_data]*100, ignore_index=True).sample(frac=1, ignore_index=True)
        ###################
        X_test, y_test = diag_data.iloc[:, :-1], diag_data['label']
        X_train, y_train = copy.deepcopy(X_test), copy.deepcopy(y_test)

    data_csv = pd.concat([X_train,y_train], axis=1)
    test_csv = pd.concat([X_test,y_test], axis=1)
    
    num_features = data_csv.shape[1]-1
    num_classes = int(max(test_csv['label'])+1)
    # print('nummmmmm')
    # print(num_classes)
    
    params = []
    for i in range(num_thresholds):
        params.append(2*np.ones((num_classes, num_features, 2)))
    
    thetas = []
    for i in range(num_thresholds):
        thetas.append(np.random.beta(params[i][:,:,0], params[i][:,:,1]))
    
    # possible_ys = sorted(list(set(test_csv['label'].to_numpy())))
    priors = []
    # for l in possible_ys:
    for l in range(num_classes):
        data_y = test_csv[test_csv['label']==l]
        priors.append(len(data_y)/len(test_csv))
    
    theta_used_freq = np.ones((num_classes, num_features, num_thresholds))
        
    
    return params, thetas, np.array(priors), test_csv, data_csv, theta_used_freq

def calculate_expected_theta(thetas, theta_used_freq, label, feature):
    frequencies = theta_used_freq[label, feature,:]
    probs = frequencies/np.sum(frequencies)
    values = np.array([thetas[i][label, feature] for i in range(len(thetas))])
    return (values * probs).sum()


def create_dataset_for_efdt_vfdt(dataset, rand_state):

    if dataset == 'diabetes':
        data = pd.read_csv('pima-indians-diabetes.csv', header=None)
        data.columns = list(range(len(data.columns)-1)) + ['label']
        np_from_data = data.to_numpy()
        np_from_data[:, :-1] = minmax_scale(np_from_data[:, :-1])
        data_new = pd.DataFrame(np_from_data, columns=data.columns)
        X_train, X_test, y_train, y_test = train_test_split(data_new.iloc[:,:-1], data_new['label'], test_size=0.8, random_state=rand_state)

    if dataset == 'breastcancer':
        data = pd.read_csv('breast-cancer.csv')
        data.drop('id', inplace=True, axis=1)
        data.columns = ['label'] + list(range(len(data.columns)-1)) 
        encoder = LabelEncoder()
        encoder.fit(data['label'])
        data['label'] = encoder.transform(data['label'])
        np_from_data = data.to_numpy()
        np_from_data[:, 1:] = minmax_scale(np_from_data[:, 1:])
        data_new = pd.DataFrame(np_from_data, columns=data.columns)
        X_train, X_test, y_train, y_test = train_test_split(data_new.iloc[:,1:], data_new['label'], test_size=0.8, random_state=rand_state)

    if dataset== 'fetal':
        data = pd.read_csv('fetal_health.csv')
        data.columns = list(range(len(data.columns)-1)) + ['label']
        encoder = LabelEncoder()
        encoder.fit(data['label'])
        data['label'] = encoder.transform(data['label'])
        np_from_data = data.to_numpy()
        np_from_data[:, :-1] = minmax_scale(np_from_data[:, :-1])
        data_new = pd.DataFrame(np_from_data, columns=data.columns)
        X_train, X_test, y_train, y_test = train_test_split(data_new.iloc[:,:-1], data_new['label'], test_size=0.05, random_state=rand_state)

    if dataset == 'fico':
        data = pd.read_csv("fico_binary.csv.train2.csv", delimiter=';')
        data.columns = list(range(len(data.columns)-1)) + ['label']
        np_from_data = data.to_numpy()
        data_new = pd.DataFrame(np_from_data, columns=data.columns)
        X_train, X_test, y_train, y_test = train_test_split(data_new.iloc[:,:-1], data_new['label'], test_size=0.9, random_state=rand_state)
        # data = pd.read_csv("fico_binary.csv.train2.csv", delimiter=';')
        # data.columns = list(range(len(data.columns)-1)) + ['label']
        # np_from_data = data.to_numpy()
        # data_new = pd.DataFrame(np_from_data, columns=data.columns)
        # X_test, y_test = data_new.iloc[:,:-1], data_new['label']

        # data = pd.read_csv("fico_binary.csv.test2.csv", delimiter=';')
        # data.columns = list(range(len(data.columns)-1)) + ['label']
        # np_from_data = data.to_numpy()
        # data_new = pd.DataFrame(np_from_data, columns=data.columns)
        # X_train, y_train = data_new.iloc[:,:-1], data_new['label']

    if dataset == 'compas':
        data = pd.read_csv("compas-binary.csv.train5.csv", delimiter=';')
        data.columns = list(range(len(data.columns)-1)) + ['label']
        np_from_data = data.to_numpy()
        data_new = pd.DataFrame(np_from_data, columns=data.columns)
        X_train, X_test, y_train, y_test = train_test_split(data_new.iloc[:,:-1], data_new['label'], test_size=0.9, random_state=rand_state)

        # X_test, y_test = data_new.iloc[:,:-1], data_new['label']

        # data = pd.read_csv("compas-binary.csv.test5.csv", delimiter=';')
        # data.columns = list(range(len(data.columns)-1)) + ['label']
        # np_from_data = data.to_numpy()
        # data_new = pd.DataFrame(np_from_data, columns=data.columns)
        # X_train, y_train = data_new.iloc[:,:-1], data_new['label']

    
    return X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()



def calculate_performance(y_true, y_pred, metric='accuracy'):
    if metric=="accuracy":
        return accuracy_score(y_true, y_pred)
    if metric=="fscore":
        return f1_score(y_true, y_pred, average='weighted')

def calculate_theta_mle(rand_state, dataset, thresholds=None):
    


    if dataset == 'synthetic':
        syn_data = pd.read_csv(f'synthetic_odt_data/{rand_state%10}.csv').drop('index', axis=1)
        labels = pd.DataFrame(syn_data['y'])
        features = list(syn_data.columns)
        features.remove('y')
        syn_data = pd.concat([syn_data[features], labels], axis=1)
        syn_data.columns = list(range(len(syn_data.columns)-1))+['label']
        X_train, X_test, y_train, y_test = train_test_split(syn_data.iloc[:, :-1], syn_data['label'],
                                                            test_size=0.1, random_state=rand_state)

    if dataset == 'heart':
        heart_data = pd.read_csv('heart.csv')
        labels = pd.DataFrame(heart_data['OVERALL_DIAGNOSIS'])
        features = list(heart_data.columns)
        features.remove('OVERALL_DIAGNOSIS')

        heart_data = pd.concat([heart_data[features], labels], axis=1)
        heart_data.columns = list(range(len(heart_data.columns)-1))+['label']
        X_train, X_test, y_train, y_test = train_test_split(heart_data.iloc[:,:-1], heart_data['label'], test_size=0.9, random_state=rand_state)


    if dataset == 'synthetic':
        syn_data = pd.read_csv(f'synthetic_odt_data/{rand_state%10}.csv').drop('index', axis=1)
        labels = pd.DataFrame(syn_data['y'])
        features = list(syn_data.columns)
        features.remove('y')
        syn_data = pd.concat([syn_data[features], labels], axis=1)
        syn_data.columns = list(range(len(syn_data.columns)-1))+['label']
        X_train, X_test, y_train, y_test = train_test_split(syn_data.iloc[:, :-1], syn_data['label'],
                                                            test_size=0.1, random_state=rand_state)

    if dataset == 'led':
        data_csv = pd.read_csv('data_big.csv',header=None)
        data_csv.columns = [0, 1, 2, 3, 4, 5, 6, 'label']
        X_train, X_test, y_train, y_test = train_test_split(data_csv[[0, 1, 2, 3, 4, 5, 6]], data_csv['label'], test_size=0.5, random_state=rand_state)

    if dataset == 'diabetes':
        data = pd.read_csv('pima-indians-diabetes.csv', header=None)
        data.columns = list(range(len(data.columns)-1)) + ['label']
        np_from_data = data.to_numpy()
        np_from_data[:, :-1] = minmax_scale(np_from_data[:, :-1])
        data_new = pd.DataFrame(np_from_data, columns=data.columns)
        X_train, X_test, y_train, y_test = train_test_split(data_new.iloc[:,:-1], data_new['label'], test_size=0.8, random_state=rand_state)
    
    if dataset == 'breastcancer':
        data = pd.read_csv('breast-cancer.csv')
        data.drop('id', inplace=True, axis=1)
        data.columns = ['label'] + list(range(len(data.columns)-1)) 
        encoder = LabelEncoder()
        encoder.fit(data['label'])
        data['label'] = encoder.transform(data['label'])
        np_from_data = data.to_numpy()
        np_from_data[:, 1:] = minmax_scale(np_from_data[:, 1:])
        data_new = pd.DataFrame(np_from_data, columns=data.columns)
        X_train, X_test, y_train, y_test = train_test_split(data_new.iloc[:,1:], data_new['label'], test_size=0.8, random_state=rand_state)

    if dataset== 'fetal':
        data = pd.read_csv('fetal_health.csv')
        data.columns = list(range(len(data.columns)-1)) + ['label']
        encoder = LabelEncoder()
        encoder.fit(data['label'])
        data['label'] = encoder.transform(data['label'])
        np_from_data = data.to_numpy()
        np_from_data[:, :-1] = minmax_scale(np_from_data[:, :-1])
        data_new = pd.DataFrame(np_from_data, columns=data.columns)
        X_train, X_test, y_train, y_test = train_test_split(data_new.iloc[:,:-1], data_new['label'], test_size=0.05, random_state=rand_state)
    
    if dataset == 'tumor':
        df = pd.read_csv('primary-tumor.data.csv',header=None)
        data = df.replace("?", np.NaN)
        data = data.dropna()
        data = data.apply(pd.to_numeric)
        data = data.apply(lambda x: x-1)
        data = data.drop([1, 3, 4],axis=1)
        data.columns = ['label'] + list(range(len(data.columns)-1)) 
        encoder = LabelEncoder()
        encoder.fit(data['label'])
        data['label'] = encoder.transform(data['label'])
        X_train, y_train = data.iloc[:,1:], data['label']
        X_test, y_test = data.iloc[:,1:], data['label']

    if dataset == 'fico':
        data = pd.read_csv("fico_binary.csv.train2.csv", delimiter=';')
        data.columns = list(range(len(data.columns)-1)) + ['label']
        np_from_data = data.to_numpy()
        data_new = pd.DataFrame(np_from_data, columns=data.columns)
        X_train, X_test, y_train, y_test = train_test_split(data_new.iloc[:,:-1], data_new['label'], test_size=0.9, random_state=rand_state)
        # data = pd.read_csv("fico_binary.csv.train2.csv", delimiter=';')
        # data.columns = list(range(len(data.columns)-1)) + ['label']
        # np_from_data = data.to_numpy()
        # data_new = pd.DataFrame(np_from_data, columns=data.columns)
        # X_test, y_test = data_new.iloc[:,:-1], data_new['label']

        # data = pd.read_csv("fico_binary.csv.test2.csv", delimiter=';')
        # data.columns = list(range(len(data.columns)-1)) + ['label']
        # np_from_data = data.to_numpy()
        # data_new = pd.DataFrame(np_from_data, columns=data.columns)
        # X_train, y_train = data_new.iloc[:,:-1], data_new['label']

    if dataset == 'compas':
        data = pd.read_csv("compas-binary.csv.train5.csv", delimiter=';')
        data.columns = list(range(len(data.columns)-1)) + ['label']
        np_from_data = data.to_numpy()
        data_new = pd.DataFrame(np_from_data, columns=data.columns)
        X_train, X_test, y_train, y_test = train_test_split(data_new.iloc[:,:-1], data_new['label'], test_size=0.9, random_state=rand_state)
        # X_test, y_test = data_new.iloc[:,:-1], data_new['label']

        # data = pd.read_csv("compas-binary.csv.test5.csv", delimiter=';')
        # data.columns = list(range(len(data.columns)-1)) + ['label']
        # np_from_data = data.to_numpy()
        # data_new = pd.DataFrame(np_from_data, columns=data.columns)
        # X_train, y_train = data_new.iloc[:,:-1], data_new['label']


    if dataset == 'zoo':
        zoo_data = pd.read_csv('zoo.csv')
        labels = pd.DataFrame(zoo_data['class_type']-1)
        features = list(zoo_data.columns)

        features.remove('class_type')
        features.remove('animal_name')
        features.remove('legs')

        zoo_data = pd.concat([zoo_data[features], labels], axis=1)
        zoo_data.columns = list(range(len(zoo_data.columns)-1))+['label']
        X_train, X_test, y_train, y_test = train_test_split(zoo_data.iloc[:, :-1], zoo_data['label'],
                                                           test_size=0.8, random_state=rand_state)

    if dataset == 'heart':
        heart_data = pd.read_csv('heart.csv')
        labels = pd.DataFrame(heart_data['OVERALL_DIAGNOSIS'])
        features = list(heart_data.columns)
        features.remove('OVERALL_DIAGNOSIS')

        heart_data = pd.concat([heart_data[features], labels], axis=1)
        heart_data.columns = list(range(len(heart_data.columns)-1))+['label']
        X_train, X_test, y_train, y_test = train_test_split(heart_data.iloc[:, :-1], heart_data['label'],
                                                            test_size=0.9, random_state=rand_state)
    
    if dataset == 'diag':
        numpy_data = pickle.load(open("scenarios_processed.pkl","rb"))
        diag_data = pd.DataFrame(numpy_data)
        labels = pd.DataFrame(diag_data[940])
        features = list(diag_data.columns)
        features.remove(940)
        diag_data.columns = list(range(len(diag_data.columns)-1))+['label']
        ###################
        diag_data = diag_data[diag_data['label'].isin(list(range(15)))]
        for c in diag_data.columns:
            if sum(diag_data[c]) == 0:
                diag_data = diag_data.drop([c], axis=1)
        diag_data.columns = list(range(len(diag_data.columns)-1))+['label']
        diag_data = pd.concat([diag_data]*100, ignore_index=True).sample(frac=1, ignore_index=True)
        ###################
        X_test, y_test = diag_data.iloc[:, :-1], diag_data['label']
        X_train, y_train = copy.deepcopy(X_test), copy.deepcopy(y_test)


    
    data_csv = pd.concat([X_train,y_train], axis=1)
    test_csv = pd.concat([X_test,y_test], axis=1)

    num_samples = data_csv.shape[0]
    num_features = data_csv.shape[1]-1
    num_classes = int(max(test_csv['label'])+1)
    

    if thresholds:
        print(thresholds)
        thetas_mle = []
        for thr in thresholds:
            theta_mle = np.ones((num_classes, num_features))

            #Calculate initial theta based on data
            for y in range(num_classes):
                data_y = test_csv[test_csv['label']==y]
                if len(data_y) > 0:
                    for feature in range(num_features):
                        theta_mle[y, feature] = sum(data_y[feature]>thr)/len(data_y)
                        if not (dataset == 'diag'):
                            if theta_mle[y, feature] == 1:
                                theta_mle[y, feature] = theta_mle[y, feature] - 0.00001
                            if theta_mle[y, feature] == 0:
                                theta_mle[y, feature] = theta_mle[y, feature] + 0.00001
            thetas_mle.append(theta_mle)
        return thetas_mle

    else:
        print(num_features)
        print(num_classes)
        theta_mle = np.ones((num_classes, num_features))

        #Calculate initial theta based on data
        for y in range(num_classes):
            data_y = test_csv[test_csv['label']==y]
            if len(data_y) > 0:
                for feature in range(num_features):
                    # print(y)
                    # print(feature)
                    if dataset == 'breastcancer':
                        
                        theta_mle[y, feature] = sum(data_y[feature]>0.5)/len(data_y)
                    else:
                        theta_mle[y, feature] = sum(data_y[feature])/len(data_y)
                    if not (dataset == 'diag'):
                        if theta_mle[y, feature] == 1:
                            theta_mle[y, feature] = theta_mle[y, feature] - 0.00001
                        if theta_mle[y, feature] == 0:
                            theta_mle[y, feature] = theta_mle[y, feature] + 0.00001

        return [theta_mle]

def calculate_map_estimate_theta(params):
    alpha = params[:,:,0]
    beta = params[:,:,1]
    return (alpha-1)/(alpha+beta-2)

def create_expected_cost_for_test_outcomes(num_features, dataset):
    try:
        return pickle.load(open("expected_cost"+"_"+dataset+str(num_features)+".pkl", "rb" ))
    except:
        exp_cost = np.random.randint(low=1, high=10, size=(num_features, 2))
        pickle.dump(exp_cost,open("expected_cost"+"_"+dataset+str(num_features)+".pkl", "wb" ))
        return exp_cost

def calculate_cost_of_observation(exp_cost, observation, label, thetas_mle):
    cost = 0
    for test in observation.keys():
        
        cost += exp_cost[test,1] *\
              thetas_mle[0][label,test] + \
              exp_cost[test,0] *\
              (1-thetas_mle[0][label,test])
    
    return cost

def calculate_expected_total_cost_for_tests(priors, exp_cost_per_outcome, thetas):
    t = exp_cost_per_outcome[:,0].T*(1-thetas[0]) + exp_cost_per_outcome[:,1].T*thetas[0]
    return np.sum(t * priors.reshape((-1,1)), axis=0)

def create_correct_hypo(features, label, thetas_mle, thetas_sampled, threshold=None):
        
    if threshold is not None:
        for i in range(len(features)):
            features[i] = int(features[i]>=threshold)
        features = features.astype(int)
    value = ''.join(str(x) for x in features)
    # print('value:', value)
    h = Hypothesis(value)
    # h.decision_region = label

    sampled_log_probs = {}
    for y_prime in range(thetas_mle[0].shape[0]):
        p_x_y_sampled = copy.deepcopy(thetas_sampled[0][y_prime,:])
        p_i_sampled = np.log(p_x_y_sampled)
        q_i_sampled = np.log(1-p_x_y_sampled)
        
        t = 0
        for i,v in enumerate(value):
            if int(v) == 1:
                t+=p_i_sampled[i]
            else:
                t+=q_i_sampled[i]
        sampled_log_probs[y_prime] = t
    h.log_prob_sampled = sampled_log_probs
    
    max_log_prob = sampled_log_probs[0]
    assigned_label = 0
    for l in range(thetas_mle[0].shape[0]):
        if h.log_prob_sampled[l]>max_log_prob:
            assigned_label = l
            max_log_prob = h.log_prob_sampled[l]
    
    h.decision_region = assigned_label
    p_x_y = copy.deepcopy(thetas_mle[0][assigned_label,:])
    p_i = np.log(p_x_y)
    q_i = np.log(1-p_x_y)
    h.log_prob = 0
    for i,v in enumerate(value):
        if int(v) == 1:
            h.log_prob+=p_i[i]
        else:
            h.log_prob+=q_i[i]
    
    return h





def test():
    # theta_mle = calculate_theta_mle(101, 'diag')[0]
    # x  = np.sum(theta_mle>0.5, axis=1)
    # h1 = Hypothesis('111')
    # h1.decision_region = 0
    # h1.log_prob = np.log(0.1)
    # h1.log_prob_sampled = {
    #     0: np.log(0.4),
    #     1: np.log(0.5)
    # }

    # h2 = Hypothesis('110')
    # h2.decision_region = 1
    # h2.log_prob = np.log(0.3)
    # h2.log_prob_sampled = {
    #     0: np.log(0.6),
    #     1: np.log(0.7)
    # }
    # p_y_xA = np.array([0.1, 0.9])
    # print(calculate_h_probs_with_observations({h1, h2}, p_y_xA))
    # print(calculate_p_y_xA(np.array([[0.1,0.2],[0.2,0.4]]), np.array([0.2,0.8]),{},False,1, None))
    exp_cost = create_expected_cost_for_test_outcomes(4, 'test')
    print(exp_cost)
    # thetas = [np.array([[0.3,0.2,0.4],[0.1,0.4,0.1]])]
    # print(calculate_expected_total_cost_for_tests(np.array([0.6,0.4]),exp_cost,thetas))
    thetas = [np.array([[1,0,0.5,1], [0,0,1,1], [1,1,0,1], [0,0,1,1], [0,1,0,1]])]
    thetas_sampled = [np.array([[0.1,0.3,0.4,0.7], [0.2,0.3,0.3,0.2], [0.4,0.3,0.1, 0.9], [0.4,0.3,0.1, 0.9], [0.4,0.3,0.1, 0.9]])]
    print(calculate_cost_of_observation(exp_cost, {0:(0,1), 3:(0,0), 2:(0,1)}, 1, thetas))
    # h = create_correct_hypo(np.array([1,0,0,1]),0,thetas,thetas_sampled)
    # print(h.value)
    # print(h.decision_region)
    # print(np.exp(h.log_prob))
    # print({y:np.exp(h.log_prob_sampled[y]) for y in h.log_prob_sampled.keys()})

if __name__ == "__main__":
    test()