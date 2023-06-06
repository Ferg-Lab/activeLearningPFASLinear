import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process.kernels import Matern
from sklearn.preprocessing import  MinMaxScaler
from sklearn.preprocessing import  StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import ShuffleSplit
from scipy.optimize import minimize
from scipy.integrate import simps
from scipy.stats import norm

# helper functions 
def ucb(mu, std, kappa):
    return mu + kappa * std

def loguniform(low=0, high=1, size=None):
    return np.exp(np.random.uniform(low, high, size))

def get_scalarization():
    a = np.random.uniform()
    return np.array([a, 1.0 - a])

scalarize = lambda x, y, theta: theta[0] * x + theta[1] * y 
scalarize_std = lambda a, b, theta: np.sqrt((theta[0] * a) ** 2 + (theta[1] * b) ** 2)


def expected_improvement(x, gaussian_process, evaluated_loss, greater_is_better=True, n_params=1):
    """ expected_improvement
    Expected improvement acquisition function.
    Arguments:
    ----------
        x: array-like, shape = [n_samples, n_hyperparams]
            The point for which the expected improvement needs to be computed.
        gaussian_process: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.
        evaluated_loss: Numpy array.
            Numpy array that contains the values off the loss function for the previously
            evaluated hyperparameters.
        greater_is_better: Boolean.
            Boolean flag that indicates whether the loss function is to be maximised or minimised.
        n_params: int.
            Dimension of the hyperparameter space.
    """

    x_to_predict = x.reshape(-1, n_params)

    mu, sigma = gaussian_process.predict(x_to_predict, return_std=True)

    if greater_is_better:
        loss_optimum = np.max(evaluated_loss)
    else:
        loss_optimum = np.min(evaluated_loss)
        
    #print(loss_optimum)

    scaling_factor = (-1) ** (not greater_is_better)

    # In case sigma equals zero
    with np.errstate(divide='ignore'):
        Z = scaling_factor * (mu - loss_optimum) / sigma.reshape(-1,1)
        expected_improvement = scaling_factor * (mu - loss_optimum) * norm.cdf(Z) + sigma.reshape(-1,1) * norm.pdf(Z)
        expected_improvement[sigma == 0.0] == 0.0
        
        
    #print(X.shape, mu.shape, sigma.shape, loss_optimum, Z.shape, expected_improvement.shape)

    return scaling_factor * expected_improvement


def trainGPR(trainData, objective, npcs, isotropic):
    
    pcListNames=[]
    for (item1, item2) in zip(['pc']*npcs, np.arange(0,npcs)):
        pcListNames.append(item1+str(item2))

    normalizedXData = trainData[pcListNames].values.reshape(-1,npcs) 
    #normalizedXData = trainData[['pc0', 'pc1']].values.reshape(-1,2) 

    standardYData = trainData['f_obj_' + objective].values.reshape(-1,1)

    standardYerrorData = trainData['f_obj_error_' + objective].values
    ysdplot = standardYerrorData
    
    standardYerrorData = standardYerrorData/(np.std(standardYData))


    X = normalizedXData
    y = standardYData
    ysd = standardYerrorData

    ysdvar= ysd*ysd 

    #np.shape(X)[1]
    if isotropic:
        nk = 1
    else:
        nk = npcs
    
    kernel = RBF([100]*nk, length_scale_bounds=[(1e-3, 2e2)]*nk )

    #print("Init lengthscale: ", kernel)

    gprModel = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=1000, alpha=ysdvar, normalize_y=True)
    gprModel.fit(X, y)

    print("Accuracy score for training data: %.4f" % (mean_squared_error(y, gprModel.predict(X))))
    print("R2 score for training data: %.4f" % (r2_score(y, gprModel.predict(X))))

    print(gprModel.kernel_.get_params()['length_scale'])
    kernelParam = gprModel.kernel_.get_params()['length_scale']
    
    result = {'X' : X, 
              'y' : y, 
              'ysdplot' : ysdplot, 
              'gprModel' : gprModel,
              'kernelParam': kernelParam}
    
    return result


