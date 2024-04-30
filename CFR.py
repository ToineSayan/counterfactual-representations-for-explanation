from concept_erasure import LeaceEraser, LeaceFitter
import torch
import numpy as np
import scipy

from sklearn.neural_network import  MLPRegressor
from sklearn.mixture import GaussianMixture



class Erasure:
    
    def __init__(self, projection='orth', rcond=1e-5):
        """
        Parameters: 
        projection (str): 'orth' for an orthogonal projection, 'leace' for an oblique projection using the LEACE projector
        rcond (str): linalg rcond coef
        """
        self.rcond = rcond
        self.projection = projection
        self.P = None
        self.SigXZ = None
        self.E_SigXZ = None
        self.E_P = None
    
    def fit(self, X, Z_1hot):
        self.SigXZ = LeaceFitter.fit(
            torch.Tensor(X),
            torch.Tensor(Z_1hot)
        ).sigma_xz.numpy()
        self.E_SigXZ = scipy.linalg.orth(self.SigXZ, rcond=self.rcond).T

        if self.projection == 'leace':
            self.P = LeaceEraser.fit(
                torch.Tensor(X),
                torch.Tensor(Z_1hot)                    
        ).P.numpy()
        else:
            # I - A (A.T A)^-1 A.T
            self.P = np.eye(self.SigXZ.shape[0]) - self.E_SigXZ.T @ np.linalg.inv(self.E_SigXZ @ self.E_SigXZ.T) @ self.E_SigXZ
        self.E_P = scipy.linalg.orth(self.P, rcond=self.rcond).T
    
    def erase_concept(self, X):
        return X @ self.P.T
    
    def to_vec(self, coordinates, subspace):
        E_subspace = self.E_P if subspace == 'E_P' else self.E_SigXZ
        if len(coordinates.shape) < 2:
            return coordinates[:,np.newaxis] @ E_subspace
        else:
            return coordinates @ E_subspace
    
    def get_coordinates(self, X, subspace):
        E_subspace = self.E_P if subspace == 'E_P' else self.E_SigXZ
        if self.projection == 'leace':
            return (X - X @ self.P.T) @ E_subspace.T.squeeze()
        else:
            return X @ E_subspace.T.squeeze()


class LinearConceptValueAssignmentWithMSEloss:
    
    def __init__(self, random_state, rcond=1e-5):
        self.predictors = {}
        self.noises = {}
        self.LinearErasure = Erasure() # P and Sigma_XZ in it
        self.random_state = random_state
        self.alpha = 1e-4 # orth
        self.lr = 1e-2

    def fit_linear_classifier(self, X, Z, linreg_args = {}, validate=False, X_validation=None, Z_validation=None):
        shuffle = True
        validation_fraction = 0.3
        if validate:
            validation_fraction = X_validation.shape[0] / (X.shape[0] + X_validation.shape[0]) 
            X = np.concatenate((X,X_validation), axis=0)
            Z = np.concatenate((Z,Z_validation))
            shuffle = False
        
        X_ = self.LinearErasure.erase_concept(X)
        coord_SigXZ = self.LinearErasure.get_coordinates(X, subspace='E_SigXZ')
        for z in np.unique(Z): 
            linear_regression_mse = MLPRegressor(
                hidden_layer_sizes=(), # linear
                activation="identity",
                max_iter=5000,
                solver='adam',
                early_stopping=True,
                validation_fraction=validation_fraction,
                n_iter_no_change=100,
                alpha=self.alpha,
                learning_rate_init=self.lr,
                random_state=self.random_state,
                warm_start=True,
                shuffle=shuffle
            )
            linear_regression_mse.set_params(**linreg_args)
            linear_regression_mse.fit(X_[Z==z], coord_SigXZ[Z==z])
            self.predictors[z] = linear_regression_mse
        return True

    def fit_gaussian_dispersion(self, X, Z):
        X_ = self.LinearErasure.erase_concept(X)
        coord_SigXZ = self.LinearErasure.get_coordinates(X, subspace='E_SigXZ')
        # Calculate the noise variance per concept value
        for z in np.unique(Z):
            nu_X = self.predictors[z].predict(X_[Z==z])
            noise = coord_SigXZ[Z==z] - nu_X
            if noise.ndim == 1:
                noise = noise.reshape(-1, 1)
            gmm = GaussianMixture(n_components=1, covariance_type='full', max_iter=300)
            gmm.fit(noise)
            self.noises[z] = gmm
        return True

    def fit(self, X, Z, validate=False, X_validation=None, Z_validation=None):
        self.fit_linear_classifier(X, Z, validate=validate, X_validation=X_validation, Z_validation=Z_validation)
        self.fit_gaussian_dispersion(X, Z)
        return True
    
    def predict_nu(self, X, Z_assigned):
        X_ = self.LinearErasure.erase_concept(X)
        coord_SigXZ_predicted = np.empty((X.shape[0], self.LinearErasure.E_SigXZ.shape[0]))
        for z in np.unique(Z_assigned):
            indices = Z_assigned == z
            predictions = self.predictors[z].predict(X_[indices])
            if len(predictions.shape) < 2: # predictions must be a matrix with 2 dimensions
                predictions = predictions[:,np.newaxis]
            coord_SigXZ_predicted[indices] = predictions
        X_SigXZ  = self.LinearErasure.to_vec(coord_SigXZ_predicted, subspace='E_SigXZ') 
        return X_ + X_SigXZ


    def predict(self, X, Z_assigned, no_sampling = False):
        X_n_nu = self.predict_nu(X, Z_assigned)
        # Let's sample noise for each observation
        if not no_sampling:
            X_noises_sampled = np.empty(X.shape)
            for z in np.unique(Z_assigned):
                indices = Z_assigned == z
                n_noises = np.sum(indices)
                noises = self.noises[z].sample(n_noises)
                X_noises_sampled[indices] = self.LinearErasure.to_vec(noises[0],subspace='E_SigXZ')
            return X_n_nu + X_noises_sampled
        else:
            return X_n_nu
        
    def score(self, X, Z):
        X_nu = self.predict(X, Z, no_sampling=True) 
        return np.mean(np.linalg.norm(X-X_nu, axis=1))

    def sample_counterfactuals(self, X, z_value, no_sampling=False):
        color = 'blue'
        Z_assigned = np.full((X.shape[0],), z_value)
        X_to_plot = self.predict(X, Z_assigned, no_sampling)
        X_to_plot = self.LinearErasure.get_coordinates(X_to_plot, subspace='E_SigXZ')
        if len(X_to_plot.shape) > 1 and X_to_plot.shape[1] == 2:
            _ = pyplot.scatter(X_to_plot[:,0], X_to_plot[:,1], s=0.1, color=color)
        elif len(X_to_plot.shape) == 1:
            _ = pyplot.hist(X_to_plot[:,], bins=100, color=color) 


# class CounterfactualClassifier:
    
#     def __init__(self, clf, cf_generator, random_state):
#         self.clf = clf # a classifier 
#         self.cf_generator = cf_generator # a countrefactual generator
    

#     def cf_predict(self, X, Z_assigned, num_cf=20, no_sampling=False):
#         """
#         Predicts values using the self.clf classifier from CFRs generated 
#         from observations and values of the manipulated attribute to be assigned.

#         Parameters
#         ----------
#         X: ndarray of shape (n_samples, n_features)
#             Original data representations.

#         Z_assigned: ndarray of shape (n_samples,)
#             Counterfactual values of the manipulated attribute to assign.

#         num_cf: int (default: 20)
#             Number of counterfactuals to sample when CFRs are considered stochastic.
        
#         no_sampling: bool (default: False)
#             if True CFRs are deterministic, else CFRs are considered stochastic.

#         Returns
#         -------
#         y: ndarray of shape (n_samples,)
#             Predictions 
#         """
#         y_probs_avg = self.cf_predict_proba(X, Z_assigned, num_cf, no_sampling)
#         y = np.argmax(y_probs_avg, axis=1)
#         return y


#     def cf_predict_proba(self, X, Z_assigned, num_cf=20, no_sampling=False):
#         """
#         Predicts probabilities over Y-values using the local classifier from CFRs generated 
#         from observations and values of the manipulated attribute to be assigned.

#         Parameters
#         ----------
#         X: ndarray of shape (n_samples, n_features)
#             Original data representations.

#         Z_assigned: ndarray of shape (n_samples,)
#             Counterfactual values of the manipulated attribute to assign.

#         num_cf: int (default: 20)
#             Number of counterfactuals to sample when CFRs are considered stochastic.
        
#         no_sampling: bool (default: False)
#             if True CFRs are deterministic, else CFRs are considered stochastic.

#         Returns
#         -------
#         y: ndarray of shape (n_samples, n_classes)
#             Probability distributions over Y-values 
#         """
#         y_probs_avg = np.zeros((X.shape[0], self.clf.classes_.shape[0]))
#         if no_sampling:
#             num_cf = 1 
#         for _ in range(num_cf):
#             X_sampled= self.cf_generator.predict(X, Z_assigned, no_sampling=no_sampling)
#             y_probs = self.clf.predict_proba(X_sampled)
#             y_probs_avg += y_probs
#         return (1/num_cf)*y_probs_avg
    
#     def orig_score(self, X, Y):
#         return self.clf.score(X, Y)
    
#     def orig_predict(self, X):
#         return self.clf.predict(X)
    
#     def orig_predict_proba(self, X):
#         return self.clf.predict_proba(X)


#     def evaluate(self, X, Y, X_CF=None, Z_CF=None, Y_CF=None, use_counterfactuals=False):
        
#         results = dict()
#         y = self.orig_predict(X)
#         results["Accuracy"] = self.orig_score(X,Y)
        
#         # PIP calculations
#         if use_counterfactuals:
#             results["Accuracy using CFs"] = self.orig_score(X_CF,Y_CF)
#             y_cf = self.orig_predict(X_CF) 
#             y_cf_fict_nu_only = self.cf_predict(X, Z_CF, no_sampling=True)
#             y_cf_fict_w_sampling = self.cf_predict(X, Z_CF, no_sampling=False)
#             results["PIP (CFRs deterministic)"] = np.mean(y_cf == y_cf_fict_nu_only)
#             results["PIP (CFRs stochastic)"] = np.mean(y_cf == y_cf_fict_w_sampling)

#         # y_cf_fict_nu_only = self.cf_predict(X, Z_CF, no_sampling=True)
#         # ATE_regression = np.mean(y-y_cf_fict_nu_only)
#         # results["ATE_regression"] = ATE_regression
#         ATE_labels = lambda a, b : np.mean(a == b)
#         results["ATE (labels) estimation"] = ATE_labels(y, self.cf_predict(X, Z_CF, no_sampling=True)) # Estimation of the ATE
        


#         # ATE calculations
#         ATE = lambda a, b : 0.5*np.mean(np.sum(np.abs(a - b), axis = 1))
#         y_cf_fict_probs = self.cf_predict_proba(X, Z_CF, no_sampling=True) # p( X(s)_{Z<-z} )
#         y_probs = self.orig_predict_proba(X) # p(X(s))
#         results["ATE estimation"] = ATE(y_probs, y_cf_fict_probs) # Estimation of the ATE
        
#         if use_counterfactuals:
#             y_cf_probs = self.orig_predict_proba(X_CF) # p( X( s_{Z<-z} ) )
#             results["ATE"] = ATE(y_probs, y_cf_probs) # real observations if CFs available
#             results["ATV"] = ATE(y_cf_probs, y_cf_fict_probs)
#         return results
    
class CFR:

    def __init__(self, projection='orth', rcond=1e-5, lcva_MSE_alpha=1e-4, lcva_MSE_lr=1e-2, random_state=42):
        self.linear_erasure = Erasure(projection=projection, rcond=rcond)
        self.LCVA_mse = LinearConceptValueAssignmentWithMSEloss(random_state=random_state)
        self.LCVA_mse.alpha = lcva_MSE_alpha
        self.LCVA_mse.lr = lcva_MSE_lr
        self.z_label2id = {}

    def fit(self,X,Z,X_val,Z_val,labels=None, verbose=False):
        if labels is None:
            z_values = sorted(np.unique(Z))
        else:
            z_values = sorted(labels)
        self.z_label2id = {z_values[i]:i for i in range(len(z_values))}

        filter, filter_val = np.in1d(Z, z_values), np.in1d(Z_val, z_values)
        Z_, Z_val_ = Z[filter], Z_val[filter_val]

        for z in z_values:
            Z_[Z_ == z] = self.z_label2id[z]
            Z_val_[Z_val_ == z] = self.z_label2id[z]
        Z_, Z_val_ = Z_.astype(int), Z_val_.astype(int)
        
        Z_1hot_ = np.zeros((Z_.size, Z_.max()+1))
        Z_1hot_[np.arange(Z_.size), Z_] = 1
   
        self.linear_erasure.fit(X[filter], Z_1hot_)
        self.LCVA_mse.LinearErasure = self.linear_erasure
        self.LCVA_mse.fit(X[filter], Z_, validate=True,X_validation=X_val[filter_val], Z_validation=Z_val_)

        if verbose: print("Mean error (train):     ", f"{self.LCVA_mse.score(X[filter], Z_):.4f}")
        if verbose: print("Mean error (validation):", f"{self.LCVA_mse.score(X_val[filter_val], Z_val_):.4f}")
        if verbose: print("Avg. norm of obs.:      ", np.mean(np.linalg.norm(X[filter], axis=1)))
    
    def predict(self, X, Z_assigned, no_concept_label=None, no_sampling=True):
        Z_assigned_ = Z_assigned.copy()
        for z in list(self.z_label2id.keys()):
            Z_assigned_[Z_assigned_ == z] = self.z_label2id[z]
        X_CF = np.empty(X.shape)
        filter = np.in1d(Z_assigned, list(self.z_label2id.keys()))
        X_CF[filter] = self.LCVA_mse.predict(X[filter],Z_assigned_[filter].astype(int),no_sampling)
        if no_concept_label:
            X_CF[Z_assigned == no_concept_label] = self.linear_erasure.erase_concept(X[Z_assigned == no_concept_label])
        return X_CF







