import numpy as np
from datasets import load_dataset
from collections import Counter
import random
import statistics
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier

from CFR import CFR
from sklearn.preprocessing import normalize

import warnings
warnings.filterwarnings("ignore")



def read_data_file(split):
    dataset = load_dataset("CEBaB/CEBaB")
    dataset = dataset.rename_column('review_majority', 'labels')
    # dataset = dataset.class_encode_column('labels')
    dataset_split = dataset[split]
    return [obs for obs in dataset_split]


def find_orig(data):
    id2index = {data[i]["id"]:i for i in range(len(data))}
    indices = np.array([i for i in range(len(data)) if data[i]["is_original"]])
    return indices


class ObsEncoder():

    def __init__(self, classes, class_labels_filters):
        self.clfs = []
        self.classes = classes
        self.class_label_filters = class_labels_filters
    
    def fit(self, X, Zs):
        for i in range(len(Zs)):
            eta_z = MLPClassifier([128], max_iter=1000, random_state=42) 
            f = np.in1d(Zs[i], class_labels_filters[i])
            eta_z.fit(X[f], Zs[i][f])
            self.clfs.append(eta_z)
    
    def score(self, X, Zs):
        scores = []
        for i in range(len(Zs)):
            f = np.in1d(Zs[i], class_labels_filters[i])
            score = self.clfs[i].score(X[f], Zs[i][f])
            scores.append((self.classes[i], score))
        return scores

    def predict(self, X):
        preds = []
        for i in range(len(self.classes)):
            pred = self.clfs[i].predict(X)
            preds.append(pred[:,np.newaxis])
        return np.concatenate(preds, axis=1)
    
    def get_approx_cf(self, X_orig, X_from, X_to):
        """
        X_orig: representations of original observations
        X_from: first member of edit pairs
        X_to: second member of edit pairs
        WARNING: X_orig.shape != X_to.shape 
        """
        preds = self.predict(X_orig)
        class_of_obs = ['']
        for i in range(len(self.class_label_filters)):
            tmp = []
            for j in range(len(self.class_label_filters[i])):
                tmp += [k + '_' + self.class_label_filters[i][j] for k in class_of_obs]
            class_of_obs = tmp
        class_of_obs = [i.strip('_') for i in class_of_obs]
        classes_of_obs = {k:[] for k in class_of_obs}
        preds_collapsed = ['_'.join(row) for row in preds]
        assert len(preds_collapsed) == X_orig.shape[0]
        for i in range(len(preds_collapsed)):
            classes_of_obs[preds_collapsed[i]].append(i)
        # Get the encoding of the second member of the edit pairs using our aspect predictor
        preds_to_collapsed = ['_'.join(row) for row in self.predict(X_to)]
        # Let's find observations in the original ones that have the same encoding
        # If there is no such original observation, then remove the edit pairs from the set
        indices_approx_cf = []
        mask = []
        for c in preds_to_collapsed:
            if classes_of_obs[c] == []:
                mask.append(False)
            else:
                mask.append(True)
                index = np.random.choice(classes_of_obs[c], 1)[0]
                indices_approx_cf.append(index)
        X_from_out = X_from[mask]
        X_to_out = X_orig[indices_approx_cf]
        return X_from_out, X_to_out, mask


def find_CFs(data, aspect, aspect_labels):
    id2index = {data[i]["id"]:i for i in range(len(data))}
    groups = {}
    data_filtered = []
    for d in data:
        if d["is_original"]:
            if (d[f'{aspect}_aspect_majority'] in aspect_labels):
                data_filtered.append(d)
            groups[d["original_id"]] = {
                "orig": d, # original observation
                "edits": [], # will store the edit group (including the original observation)
                "pairs": [] # will store the possible edit pairs (label of the aspect different and both in aspect_labels)
            }
        if d["edit_type"] == aspect and (d[f'{aspect}_aspect_majority'] in aspect_labels):
            data_filtered.append(d)
    # Now get all the groups of edits (including the original obs.)
    for d in data_filtered:
        groups[d["original_id"]]["edits"].append(d)
    # Nom let's create all the pairs of 2 observations within the groups
    # A good pair has 
    # - a different aspect majority
    for k,v in groups.items():
        for o1 in v["edits"]:
            for o2 in v["edits"]:
                if (not (o1["id"] == o2["id"])) and (o1[f'{aspect}_aspect_majority'] != o2[f'{aspect}_aspect_majority']):
                    v["pairs"].append((o1,o2))
    # get all pairs
    all_pairs = []
    for k,v in groups.items():
        all_pairs += v["pairs"]
    indices_orig = [id2index[i["id"]] for i,_ in all_pairs]
    indices_cf = [id2index[i["id"]] for _,i in all_pairs]
    return(indices_orig, indices_cf, all_pairs)

#---------------------------------------------------------------------------
# Treatment effect evaluation methods
# --------------------------------------------------------------------------

def ATE_exp_class(X, X_CF, Z, Z_CF, Y, Y_CF, from_, to_): 
    TEs_1 = (Y_CF.astype(int) - Y.astype(int))[Z == from_][Z_CF[Z == from_]==to_]
    TEs_2 = -(Y_CF.astype(int) - Y.astype(int))[Z == to_][Z_CF[Z == to_]==from_]
    return np.mean(np.concatenate([TEs_1, TEs_2]))

def ATE_clf_class(X, X_CF, Z, Z_CF, Y, Y_CF, eta_y, from_, to_): # CACE with classes as output
    TEs_1 = (eta_y.predict(X_CF).astype(int) - eta_y.predict(X).astype(int))[Z == from_][Z_CF[Z == from_]==to_]
    TEs_2 = -(eta_y.predict(X_CF).astype(int) - eta_y.predict(X).astype(int))[Z == to_][Z_CF[Z == to_]==from_]
    return np.mean(np.concatenate([TEs_1, TEs_2]))

def CACE(X, X_CF, Z, Z_CF, eta_y, from_, to_): # CaCE from Abraham et al., 2022 (CEBaB)
    """
    outputs probability distributions (1 per unordered pair {from_, to_})
    CACE(from_, to_) = -CACE(to_, from_)
    """
    TEs_1 = (eta_y.predict_proba(X_CF) - eta_y.predict_proba(X))[Z == from_][Z_CF[Z == from_]==to_]
    TEs_2 = -(eta_y.predict_proba(X_CF) - eta_y.predict_proba(X))[Z == to_][Z_CF[Z == to_]==from_]
    return np.concatenate([TEs_1, TEs_2], axis=0)



def error_eval(TEs_ref, TEs_compare, mode): # distances for ICaCE-Error 
    if mode == "l2":
        error = np.mean(np.linalg.norm(TEs_compare - TEs_ref, axis=1))
    if mode == "cosine":
        error = 1 - np.mean(np.sum(normalize(TEs_ref, axis=1, norm='l1')*normalize(TEs_compare, axis=1, norm='l1'), axis=1))
    if mode == 'normdiff':
        error = np.mean(np.abs(np.linalg.norm(TEs_compare, axis=1) - np.linalg.norm(TEs_ref, axis=1)))
    return(error)

def error_CACE(X, X_CF, X_CF_compare, Z, Z_CF, eta_y, z_labels_filter=None): # ICaCE-Error from Abraham et al., 2022 (CEBaB)
    """
    If X_CF_compare is None (i.e. no methods to compare with), it will compare the iCACE effects to a difference of 2 random vectors as in the CEBaB paper
    """
    labels_filtered = np.intersect1d(np.unique(np.concatenate([Z, Z_CF])), z_labels_filter) if z_labels_filter is not None else np.unique(np.concatenate([Z, Z_CF]))
    # unique_couples = lambda l1 : [(i,j) for i in l1 for j in l1 if i<j]
    couples =  [(i,j) for i in labels_filtered for j in labels_filtered if i<j]

    TEs_ref = np.concatenate([CACE(X, X_CF, Z, Z_CF, eta_y, z, z_cf) for z, z_cf in couples], axis=0)
    if X_CF_compare is not None:
        TEs_compare = np.concatenate([CACE(X, X_CF_compare, Z, Z_CF, eta_y, z, z_cf) for z, z_cf in couples], axis=0)
    else: # TEs compare is the difference of 2 random distributions
        rand1 = normalize(np.random.rand(TEs_ref.shape[0], TEs_ref.shape[1]), axis=1, norm='l1')
        rand2 = normalize(np.random.rand(TEs_ref.shape[0], TEs_ref.shape[1]), axis=1, norm='l1')
        TEs_compare = rand1 - rand2
    # print(TEs_ref[:10])
    # print(TEs_compare[:10])
    errors = {diff_mode:error_eval(TEs_ref, TEs_compare, diff_mode) for diff_mode in ['cosine', 'normdiff', 'l2']}
    return errors



if __name__ == '__main__':

    


    train_set = 'train_exclusive'

    output_dict = {}
    output_dict[train_set] = {}
    output_dict['validation'] = {}
    output_dict['test'] = {}
    concepts = ['food', 'service', 'ambiance', 'noise']
    # concepts = ['ambiance']

    base_dir = './datasets/CEBaB-v1.1/'
    with np.load(base_dir + f'D_{train_set}.npz') as original_data:
        X, Y_raw = original_data[f'{train_set}_X'], original_data[f'{train_set}_Y']
        X_val, Y_val_raw =  original_data['validation_X'], original_data['validation_Y']
        X_test, Y_test_raw = original_data['test_X'], original_data['test_Y']
        Zs = [original_data[f'{train_set}_Z_{c}'] for c in concepts] 
        Zs_val = [original_data[f'validation_Z_{c}'] for c in concepts] 
        Zs_test = [original_data[f'test_Z_{c}'] for c in concepts]
    
    def to_one_hot(A):
        A_1hot = np.zeros((A.size, A.max()+1))
        A_1hot[np.arange(A.size), A] = 1
        return A_1hot

    class_labels_filters = [['Positive', 'Negative', 'unknown']]*len(concepts)

    # Init a multi-aspect predictor
    OE = ObsEncoder(concepts, class_labels_filters)
    OE.fit(X, Zs)
    # scores = OE.score(X_test, Zs_test)
    # print(scores)

    # Load the source data (test split)
    data = read_data_file('test') # using Huggingface

    # Find the indices of the original data 
    indices_orig_data = find_orig(data)
    X_test_orig = X_test[indices_orig_data]

    # Find all the edit pairs ((X,X') and (X,X') are both in the set)
    # Edit pairs are calculated per aspect
    edit_pairs = {}
    for i in range(len(concepts)):
        aspect = concepts[i]
        indices_orig, indices_cf, pairs = find_CFs(data, aspect, ['Positive', 'Negative', 'unknown'])
        edit_pairs[aspect] = {
                "X": X_test[indices_orig],
                "X_CF": X_test[indices_cf],
                "Z": Zs_test[i][indices_orig],
                "Z_CF": Zs_test[i][indices_cf],
                "Y": Y_test_raw[indices_orig],
                "Y_CF": Y_test_raw[indices_cf],
                "pairs": pairs
        }

    

    CFRs = {}
    for i in range(len(concepts)):
        aspect = concepts[i]
        filter_original_from = [o["is_original"] for o,_ in edit_pairs[aspect]["pairs"]]
        CFR_ = CFR(projection='orth', rcond=1e-5, lcva_MSE_alpha=1e-4, lcva_MSE_lr=1e-2, random_state=42)
        CFR_.fit(X,Zs[i],X_val,Zs_val[i],labels=["Positive", "Negative"], verbose=False)


        # let's calculate all possible interventions on Z for every original observation
        full_interventions_X = []
        full_interventions_Z_CF = []
        full_interventions_Z = []
        
        X_tmp = edit_pairs[aspect]["X"][filter_original_from]
        Z_tmp = edit_pairs[aspect]["Z"][filter_original_from]
        for z_val in np.unique(Z_tmp):
            full_interventions_X.append(X_tmp[Z_tmp != z_val])
            full_interventions_Z_CF.append(np.full(Z_tmp[Z_tmp != z_val].shape, z_val))
            full_interventions_Z.append(Z_tmp[Z_tmp != z_val])
        full_interventions_X = np.concatenate(full_interventions_X, axis=0)
        full_interventions_Z_CF = np.concatenate(full_interventions_Z_CF)
        full_interventions_Z = np.concatenate(full_interventions_Z)


        CFRs[aspect] = {
            "X":edit_pairs[aspect]["X"][filter_original_from],
            "X_CF": CFR_.predict(edit_pairs[aspect]["X"][filter_original_from], edit_pairs[aspect]["Z_CF"][filter_original_from],no_concept_label="unknown"), 
            # "X_CF": CFR_.predict(edit_pairs[aspect]["X"][filter_original_from], edit_pairs[aspect]["Z_CF"][filter_original_from]),
            "X_CF_ref": edit_pairs[aspect]["X_CF"][filter_original_from],
            "Z": edit_pairs[aspect]["Z"][filter_original_from],
            "Z_CF": edit_pairs[aspect]["Z_CF"][filter_original_from],
            "Y": edit_pairs[aspect]["Y"][filter_original_from],
            "Y_CF": edit_pairs[aspect]["Y_CF"][filter_original_from],
            "full_X": full_interventions_X,
            "full_X_CF": CFR_.predict(full_interventions_X, full_interventions_Z_CF,no_concept_label="unknown"),
            "full_Z": full_interventions_Z,
            "full_Z_CF": full_interventions_Z_CF,
        }
    

    CFRs_3_labels = {}
    for i in range(len(concepts)):
        aspect = concepts[i]
        filter_original_from = [o["is_original"] for o,_ in edit_pairs[aspect]["pairs"]]
        CFR_ = CFR(projection='orth', rcond=1e-5, lcva_MSE_alpha=1e-4, lcva_MSE_lr=1e-2, random_state=42)
        CFR_.fit(X,Zs[i],X_val,Zs_val[i],labels=["Positive", "Negative", "unknown"], verbose=False)

        # let's calculate all possible interventions on Z for every original observation
        full_interventions_X = []
        full_interventions_Z_CF = []
        full_interventions_Z = []
        
        X_tmp = edit_pairs[aspect]["X"][filter_original_from]
        Z_tmp = edit_pairs[aspect]["Z"][filter_original_from]
        for z_val in np.unique(Z_tmp):
            full_interventions_X.append(X_tmp[Z_tmp != z_val])
            full_interventions_Z_CF.append(np.full(Z_tmp[Z_tmp != z_val].shape, z_val))
            full_interventions_Z.append(Z_tmp[Z_tmp != z_val])
        full_interventions_X = np.concatenate(full_interventions_X, axis=0)
        full_interventions_Z_CF = np.concatenate(full_interventions_Z_CF)
        full_interventions_Z = np.concatenate(full_interventions_Z)


        CFRs_3_labels[aspect] = {
            "X":edit_pairs[aspect]["X"][filter_original_from],
            "X_CF": CFR_.predict(edit_pairs[aspect]["X"][filter_original_from], edit_pairs[aspect]["Z_CF"][filter_original_from]),
            "X_CF_ref": edit_pairs[aspect]["X_CF"][filter_original_from],
            "Z": edit_pairs[aspect]["Z"][filter_original_from],
            "Z_CF": edit_pairs[aspect]["Z_CF"][filter_original_from],
            "Y": edit_pairs[aspect]["Y"][filter_original_from],
            "Y_CF": edit_pairs[aspect]["Y_CF"][filter_original_from],
            "full_X": full_interventions_X,
            "full_X_CF": CFR_.predict(full_interventions_X, full_interventions_Z_CF),
            "full_Z": full_interventions_Z,
            "full_Z_CF": full_interventions_Z_CF,
        }

    print("ECACE - Experimental CACE")
    print("Negative -> Positive:", ATE_exp_class(None, None, edit_pairs[aspect]["Z"], edit_pairs[aspect]["Z_CF"], edit_pairs[aspect]["Y"], edit_pairs[aspect]["Y_CF"], "Negative", "Positive"))
    print("Negative -> unknown :", ATE_exp_class(None, None, edit_pairs[aspect]["Z"], edit_pairs[aspect]["Z_CF"], edit_pairs[aspect]["Y"], edit_pairs[aspect]["Y_CF"], "Negative", "unknown"))
    print("Positive -> unknown :", ATE_exp_class(None, None, edit_pairs[aspect]["Z"], edit_pairs[aspect]["Z_CF"], edit_pairs[aspect]["Y"], edit_pairs[aspect]["Y_CF"], "Positive", "unknown"))
    print("\n")


    CACE_approx = {}
    CACE_CFR_binary = {}
    CACE_CFR_ternary = {}
    CACE_y = {} # CACE
    error_CACE_results = {}

    all_seeds = [i for i in range(42,52)] # 10 seeds

    # -----------------------------------------------------------------
    # Everything is deterministic above
    # -----------------------------------------------------------------

    for seed in all_seeds:

        print("SEED :", seed)
        CACE_approx[seed] = {}
        CACE_CFR_binary[seed] = {}
        CACE_CFR_ternary[seed] = {}
        CACE_y[seed] = {}
        error_CACE_results[seed] = {}

        for aspect in concepts:
            CACE_approx[seed][aspect] = {}
            CACE_CFR_binary[seed][aspect] = {}
            CACE_CFR_ternary[seed][aspect] = {}
            CACE_y[seed][aspect] = {}
    


        # Train an Y classifier
        def linear_classifier(args=None, random_state=42):
            lcf = SGDClassifier(
                loss='log_loss', validation_fraction=0.2, early_stopping=True, random_state=random_state, 
                penalty='l2', max_iter=5000, n_jobs=-1, learning_rate='optimal', warm_start=True, n_iter_no_change=1000
            )
            if args is not None:
                lcf.set_params(**args)
            return lcf
        eta_y = linear_classifier({'alpha':5e-5}, random_state=seed)
        eta_y.fit(X[Y_raw != 'no majority'], Y_raw[Y_raw != 'no majority'])


        # Let's sample approximate CFs
        approximate_cfs = {}
        for i in range(len(concepts)):
            aspect = concepts[i]
            filter_original_from = [o["is_original"] for o,_ in edit_pairs[aspect]["pairs"]]
            X_from, X_to, mask = OE.get_approx_cf(X_test_orig, edit_pairs[aspect]["X"][filter_original_from], edit_pairs[aspect]["X_CF"][filter_original_from])
            approximate_cfs[aspect] = {
                "X":X_from,
                "X_CF": X_to, 
                "X_CF_ref": edit_pairs[aspect]["X_CF"][filter_original_from][mask],
                "Z": edit_pairs[aspect]["Z"][filter_original_from][mask],
                "Z_CF": edit_pairs[aspect]["Z_CF"][filter_original_from][mask],
                "Y": edit_pairs[aspect]["Y"][filter_original_from][mask],
                "Y_CF": edit_pairs[aspect]["Y_CF"][filter_original_from][mask],
            }

        

        # CACE (class version)
        for aspect in concepts:
            print(aspect)
            print("CACE - Edit pairs in test set: ", edit_pairs[aspect]["X"].shape[0])
            print("Per type edit pairs:", Counter(
                    ['_'.join(row) for row in np.concatenate([edit_pairs[aspect]["Z"][:,np.newaxis], edit_pairs[aspect]["Z_CF"][:,np.newaxis]], axis=1)]
                ))
            CACE_y[seed][aspect]["Negative -> Positive"] = ATE_clf_class(edit_pairs[aspect]["X"],  edit_pairs[aspect]["X_CF"], edit_pairs[aspect]["Z"], edit_pairs[aspect]["Z_CF"], None, None, eta_y, "Negative", "Positive")
            CACE_y[seed][aspect]["Negative -> unknown"] = ATE_clf_class(edit_pairs[aspect]["X"],  edit_pairs[aspect]["X_CF"], edit_pairs[aspect]["Z"], edit_pairs[aspect]["Z_CF"], None, None, eta_y, "Negative", "unknown")
            CACE_y[seed][aspect]["Positive -> unknown"] = ATE_clf_class(edit_pairs[aspect]["X"],  edit_pairs[aspect]["X_CF"], edit_pairs[aspect]["Z"], edit_pairs[aspect]["Z_CF"], None, None, eta_y, "Positive", "unknown")
            [print(k, CACE_y[seed][aspect][k]) for k in CACE_y[seed][aspect].keys()]
            print("----------")

        # CACE (class version with approximate CFs)
        for aspect in concepts:
            print(aspect)
            print("CACE approx. - Approximate pairs in test set: ", approximate_cfs[aspect]["X"].shape[0])
            print("Per type edit pairs:", Counter(
                    ['_'.join(row) for row in np.concatenate([approximate_cfs[aspect]["Z"][:,np.newaxis], approximate_cfs[aspect]["Z_CF"][:,np.newaxis]], axis=1)]
                ))
            CACE_approx[seed][aspect]["Negative -> Positive"] = ATE_clf_class(approximate_cfs[aspect]["X"],  approximate_cfs[aspect]["X_CF"], approximate_cfs[aspect]["Z"], approximate_cfs[aspect]["Z_CF"], None, None, eta_y, "Negative", "Positive")
            CACE_approx[seed][aspect]["Negative -> unknown"] = ATE_clf_class(approximate_cfs[aspect]["X"],  approximate_cfs[aspect]["X_CF"], approximate_cfs[aspect]["Z"], approximate_cfs[aspect]["Z_CF"], None, None, eta_y, "Negative", "unknown")
            CACE_approx[seed][aspect]["Positive -> unknown"] = ATE_clf_class(approximate_cfs[aspect]["X"],  approximate_cfs[aspect]["X_CF"], approximate_cfs[aspect]["Z"], approximate_cfs[aspect]["Z_CF"], None, None, eta_y, "Positive", "unknown")
            [print(k, CACE_approx[seed][aspect][k]) for k in CACE_approx[seed][aspect].keys()]
            print("----------")

        # CACE (class version with CFRs)
        for aspect in concepts:
            print(aspect)
            print("CACE CFRs binary setting - Approximate pairs in test set: ", CFRs[aspect]["X"].shape[0])
            print("Per type edit pairs:", Counter(
                    ['_'.join(row) for row in np.concatenate([CFRs[aspect]["Z"][:,np.newaxis], CFRs[aspect]["Z_CF"][:,np.newaxis]], axis=1)]
                ))
            CACE_CFR_binary[seed][aspect]["Negative -> Positive"] = ATE_clf_class(CFRs[aspect]["X"],  CFRs[aspect]["X_CF"], CFRs[aspect]["Z"], CFRs[aspect]["Z_CF"], None, None, eta_y, "Negative", "Positive")
            CACE_CFR_binary[seed][aspect]["Negative -> unknown"] = ATE_clf_class(CFRs[aspect]["X"],  CFRs[aspect]["X_CF"], CFRs[aspect]["Z"], CFRs[aspect]["Z_CF"], None, None, eta_y, "Negative", "unknown")
            CACE_CFR_binary[seed][aspect]["Positive -> unknown"] = ATE_clf_class(CFRs[aspect]["X"],  CFRs[aspect]["X_CF"], CFRs[aspect]["Z"], CFRs[aspect]["Z_CF"], None, None, eta_y, "Positive", "unknown")

            # # NEW
            # CACE_CFR_binary[seed][aspect]["Negative -> Positive"] = ATE_clf_class(CFRs[aspect]["full_X"],  CFRs[aspect]["full_X_CF"], CFRs[aspect]["full_Z"], CFRs[aspect]["full_Z_CF"], None, None, eta_y, "Negative", "Positive")
            # CACE_CFR_binary[seed][aspect]["Negative -> unknown"] = ATE_clf_class(CFRs[aspect]["full_X"],  CFRs[aspect]["full_X_CF"], CFRs[aspect]["full_Z"], CFRs[aspect]["full_Z_CF"], None, None, eta_y, "Negative", "unknown")
            # CACE_CFR_binary[seed][aspect]["Positive -> unknown"] = ATE_clf_class(CFRs[aspect]["full_X"],  CFRs[aspect]["full_X_CF"], CFRs[aspect]["full_Z"], CFRs[aspect]["full_Z_CF"], None, None, eta_y, "Positive", "unknown")

            [print(k, CACE_CFR_binary[seed][aspect][k]) for k in CACE_CFR_binary[seed][aspect].keys()]
            print("----------")

        # CACE (class version with CFRs 3 labels)
        for aspect in concepts:
            print(aspect)
            print("CACE CFRs ternary setting - Approximate pairs in test set: ", CFRs_3_labels[aspect]["X"].shape[0])
            print("Per type edit pairs:", Counter(
                    ['_'.join(row) for row in np.concatenate([CFRs_3_labels[aspect]["Z"][:,np.newaxis], CFRs_3_labels[aspect]["Z_CF"][:,np.newaxis]], axis=1)]
                ))
            CACE_CFR_ternary[seed][aspect]["Negative -> Positive"] = ATE_clf_class(CFRs_3_labels[aspect]["X"],  CFRs_3_labels[aspect]["X_CF"], CFRs_3_labels[aspect]["Z"], CFRs_3_labels[aspect]["Z_CF"], None, None, eta_y, "Negative", "Positive")
            CACE_CFR_ternary[seed][aspect]["Negative -> unknown"] = ATE_clf_class(CFRs_3_labels[aspect]["X"],  CFRs_3_labels[aspect]["X_CF"], CFRs_3_labels[aspect]["Z"], CFRs_3_labels[aspect]["Z_CF"], None, None, eta_y, "Negative", "unknown")
            CACE_CFR_ternary[seed][aspect]["Positive -> unknown"] = ATE_clf_class(CFRs_3_labels[aspect]["X"],  CFRs_3_labels[aspect]["X_CF"], CFRs_3_labels[aspect]["Z"], CFRs_3_labels[aspect]["Z_CF"], None, None, eta_y, "Positive", "unknown")

            # # New
            # CACE_CFR_ternary[seed][aspect]["Negative -> Positive"] = ATE_clf_class(CFRs_3_labels[aspect]["full_X"],  CFRs_3_labels[aspect]["full_X_CF"], CFRs_3_labels[aspect]["full_Z"], CFRs_3_labels[aspect]["full_Z_CF"], None, None, eta_y, "Negative", "Positive")
            # CACE_CFR_ternary[seed][aspect]["Negative -> unknown"] = ATE_clf_class(CFRs_3_labels[aspect]["full_X"],  CFRs_3_labels[aspect]["full_X_CF"], CFRs_3_labels[aspect]["full_Z"], CFRs_3_labels[aspect]["full_Z_CF"], None, None, eta_y, "Negative", "unknown")
            # CACE_CFR_ternary[seed][aspect]["Positive -> unknown"] = ATE_clf_class(CFRs_3_labels[aspect]["full_X"],  CFRs_3_labels[aspect]["full_X_CF"], CFRs_3_labels[aspect]["full_Z"], CFRs_3_labels[aspect]["full_Z_CF"], None, None, eta_y, "Positive", "unknown")


            [print(k, CACE_CFR_ternary[seed][aspect][k]) for k in CACE_CFR_ternary[seed][aspect].keys()]
            print("----------")

        print("----- ERROR CACE -----")
        
        for aspect in concepts:
            print(aspect)
            error_CACE_results[seed][aspect] = {
                "Error CACE (random):": error_CACE(edit_pairs[aspect]["X"], edit_pairs[aspect]["X_CF"], None, edit_pairs[aspect]["Z"], edit_pairs[aspect]["Z_CF"], eta_y, z_labels_filter=['Positive', 'Negative', 'unknown']),
                "Error CACE (approx):": error_CACE(approximate_cfs[aspect]["X"], approximate_cfs[aspect]["X_CF_ref"], approximate_cfs[aspect]["X_CF"], approximate_cfs[aspect]["Z"], approximate_cfs[aspect]["Z_CF"], eta_y, z_labels_filter=['Positive', 'Negative', 'unknown']),
                "Error CACE (CFRs):": error_CACE(CFRs[aspect]["X"], CFRs[aspect]["X_CF_ref"], CFRs[aspect]["X_CF"], CFRs[aspect]["Z"], CFRs[aspect]["Z_CF"], eta_y, z_labels_filter=['Positive', 'Negative', 'unknown']),
                "Error CACE (CFRs 3 labels):": error_CACE(CFRs_3_labels[aspect]["X"], CFRs_3_labels[aspect]["X_CF_ref"], CFRs_3_labels[aspect]["X_CF"], CFRs_3_labels[aspect]["Z"], CFRs_3_labels[aspect]["Z_CF"], eta_y, z_labels_filter=['Positive', 'Negative', 'unknown'])
            }
            for k in ["Error CACE (random):", "Error CACE (approx):", "Error CACE (CFRs):", "Error CACE (CFRs 3 labels):"]:
                print(k, error_CACE_results[seed][aspect][k])
        print("-------------\nAggregated results:")
        for k in ["Error CACE (random):", "Error CACE (approx):", "Error CACE (CFRs):", "Error CACE (CFRs 3 labels):"]:
            print(k, "cosine:", np.mean([error_CACE_results[seed][i][k]["cosine"] for i in concepts]),"normdiff:", np.mean([error_CACE_results[seed][i][k]["normdiff"] for i in concepts]),"l2:", np.mean([error_CACE_results[seed][i][k]["l2"] for i in concepts]))
    
    
    print("---------------------------")
    print("Final results:")
    print("CACE")
    for aspect in concepts:
        for k in ["Negative -> Positive", "Negative -> unknown", "Positive -> unknown"]:
            res = [CACE_y[seed][aspect][k] for seed in all_seeds]
            print(aspect, k, statistics.mean(res), f"({statistics.stdev(res)})")
        print("\n")
    print("CACE approximate CFs")
    for aspect in concepts:
        for k in ["Negative -> Positive", "Negative -> unknown", "Positive -> unknown"]:
            res = [CACE_approx[seed][aspect][k] for seed in all_seeds]
            print(aspect, k, statistics.mean(res), f"({statistics.stdev(res)})")
        print("\n")
    print("CACE CFRs binary setting")
    for aspect in concepts:
        for k in ["Negative -> Positive", "Negative -> unknown", "Positive -> unknown"]:
            res = [CACE_CFR_binary[seed][aspect][k] for seed in all_seeds]
            print(aspect, k, statistics.mean(res), f"({statistics.stdev(res)})")
        print("\n")
    print("CACE CFRs ternary setting")
    for aspect in concepts:
        for k in ["Negative -> Positive", "Negative -> unknown", "Positive -> unknown"]:
            res = [CACE_CFR_ternary[seed][aspect][k] for seed in all_seeds]
            print(aspect, k, statistics.mean(res), f"({statistics.stdev(res)})")
        print("\n")
    
    print("Errors")
    for k in ["Error CACE (random):", "Error CACE (approx):", "Error CACE (CFRs):", "Error CACE (CFRs 3 labels):"]:
        for dist in ["cosine", "normdiff", "l2"]:   
            res = [error_CACE_results[seed][aspect][k][dist] for seed in all_seeds for aspect in concepts]
            print(k, dist, statistics.mean(res), f"({statistics.stdev(res)})")
        print("\n")



    
    quit()




