import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score, roc_curve, plot_confusion_matrix, plot_roc_curve

class Model_Analysis:
    
    '''
    A class that allows for comparing the performance of a model on multiple datasets.
    
    Attributes
    ----------
    
    model : Classifier - an untrained classification alogorithm for comparing against data. A pipeline is recommended in order to control scaling and imputing.
    dataset : Dictionary - a dictionary of the format: {name: {x_train, y_train}...} where the name is the title of the dataset and x/y_train are the datasets for testing (as a DataFrame and series, respectively)
    score_table : DataFrame - produced by the calc_scores method. Provides Test and Train scores for each dataset in a single dataframe.
    test_scores : DataFrame - a DataFrame showing the testing scores for each of the datasets and highlights the max value in each row for easy comparison
    
    Methods
    -------
    
    calc_scores():
        Fits the model to the three datasets and creates score_table and test_scores attributes
    
    plot_roc():
        Creates a plot with the ROC curves for all three datasets
        
    confusion_compare():
        Creates three confusion matrices of the accuracy scores of the three models on the three data_sets
    '''
    
    def __init__(self, model, dataset, X_test, y_test):
        '''
        
        Constructs necessary attributes for the Model_Analysis object.
        
        Parameters:
        ----------
        
        model : Classifier
            An untrained classification alogorithm for comparing against data. A pipeline is recommended in order to control scaling and imputing.
        dataset : dict
            A dictionary of the format: {name: {x_train, y_train}...} where the name is the title of the dataset and x/y_train are the datasets for testing (as a DataFrame and series, respectively)
        X_test : df
            A dataframe of the testing data for accuracy
        y_train : series
            A series of outcomes, with 0 being a failure and 1 being a success, corresponding to values in X_test
        
        '''
        self.model = model
        self.dataset = dataset
        self.X_test = X_test
        self.y_test = y_test
    
    def calc_scores(self):
        
        '''
        Fits the model to the three datasets and creates a DataFrame showing all testing and training and training scores for each dataset
        
        Parameters:
        ----------
        
        None
        
        Returns:
        --------
        
        score_table : DataFrame
            A DataFrame providing Test and Train scores for each dataset
        
        test_scores: DataFrame
            A DataFrame showing the testing scores for each of the datasets and highlights the max value in each row for easy comparison
        '''
        
        df_list = []
        for name in self.dataset.keys():
            x = self.dataset[name]['x']
            y = self.dataset[name]['y']
            algo = self.model.fit(x, y)
            y_pred_train = algo.predict(x)
            y_pred_test = algo.predict(self.X_test)
            
            
            dictionary = {
                'Accuracy': [accuracy_score(y, y_pred_train), accuracy_score(self.y_test, y_pred_test)],
                'Precision': [precision_score(y, y_pred_train), precision_score(self.y_test, y_pred_test)],
                'Recall': [recall_score(y, y_pred_train), recall_score(self.y_test, y_pred_test)],
                'F1 Score': [f1_score(y, y_pred_train), f1_score(self.y_test, y_pred_test)],
                'ROC-AUC Score': [roc_auc_score(y, y_pred_train), roc_auc_score(self.y_test, y_pred_test)]
            }
            
            df = pd.DataFrame.from_dict(dictionary, orient='index', columns=[f'{name} Train',f'{name} Test'])
            df_list.append(df)
    
        final_df = pd.concat(df_list, axis=1)
        
        test = final_df[[x for x in final_df.columns if x.endswith('Test')]]
        
        self.score_table = final_df
        self.test_scores = test.style.highlight_max(color='lightgreen', axis=1)
    
    def plot_roc(self):
        
        '''
        Fits the model with the three datasets and plots the ROC curves for each on a single axis
        
        Parameters
        ----------
        
        None
        
        Returns
        -------
        
        Roc Plot: Plot
            A plot of the three ROC curves corresponding to the three datasets
        '''
        
        plt.figure(figsize=(7,7))
        ax = plt.gca()
        for name in self.dataset.keys():
            x = self.dataset[name]['x']
            y = self.dataset[name]['y']
            aglo = self.model.fit(x, y)
            plot_roc_curve(aglo, self.X_test, self.y_test, ax=ax, label=name)
        plt.title('ROC Curve by Model')
        plt.show()
        
    def confusion_compare(self):
        
        '''
        Creates three confusion matrices, one for the testing data of each dataset. You must run method calc_scores before you run this method.
        
        Parameters:
        -----------
        
        None
        
        Returns:
        --------
        
        Confusion Matrix Subplot : Plot
            A subplot of confusion matricies showing the performace of the testing data on each of the datasets
        
        test_scores: DataFrame
            A DataFrame showing the testing scores for each of the datasets and highlights the max value in each row for easy comparison
        '''
        
        for name in self.dataset.keys():
            x = self.dataset[name]['x']
            y = self.dataset[name]['y']
            algo = self.model
            algo.fit(x, y)
            disp = plot_confusion_matrix(algo, self.X_test, self.y_test, cmap=plt.cm.Blues)
            disp.ax_.set_title(f'{name} Testing Matrix')
        plt.show()
        return self.test_scores

def scores(model, X_train, X_test, y_train, y_test):

    '''
    Creates confusion matrix subplots comparing the scores for the training data and the testing data.
    It also returns a dataframe of the scores together for comparison.

    Parameters
    ----------
    model : classifier object
        A model that you are analyzing

    X_train : df
        A dataframe with the training predictors
    
    X_test : df
        A dataframe with the testing predictors

    y_train : series
        A series with the training outcomes (0 for failure and 1 for success)
    
    y_test : series
        A series with the testing outcomes (0 for failure and 1 for success)
    '''

    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    dictionary = {
        'Accuracy': [accuracy_score(y_train, y_pred_train), accuracy_score(y_test, y_pred_test)],
        'Precision': [precision_score(y_train, y_pred_train), precision_score(y_test, y_pred_test)],
        'Recall': [recall_score(y_train, y_pred_train), recall_score(y_test, y_pred_test)],
        'F1 Score': [f1_score(y_train, y_pred_train), f1_score(y_test, y_pred_test)],
        'ROC-AUC Score': [roc_auc_score(y_train, y_pred_train), roc_auc_score(y_test, y_pred_test)]
    }
    
    df = pd.DataFrame.from_dict(dictionary, orient='index', columns=['Train','Test'])
    
    disp = plot_confusion_matrix(model, X_train, y_train, cmap=plt.cm.Blues)
    disp.ax_.set_title('Training Matrix')
    disp_2 = plot_confusion_matrix(model, X_test, y_test, cmap=plt.cm.Blues)
    disp_2.ax_.set_title('Testing Matrix')
    plt.show()
    return df