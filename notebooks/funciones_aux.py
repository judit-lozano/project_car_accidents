import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import recall_score, roc_auc_score, f1_score, precision_recall_curve, classification_report, confusion_matrix,\
                            ConfusionMatrixDisplay, roc_curve, accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
import scikitplot as skplt
import pickle



df_data= pd.read_csv("../data/df_data.csv")
with open('../data/train_ridge.pickle', 'rb') as f:
    X_train, y_train, X_test, y_test = pickle.load(f)

def dame_curvas(modelo):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), dpi=90)
    model = modelo
        
    # predecimos probabilidades
    prob_predictions = model.predict_proba(X_test)
    # mantener las probabilidades para resultados positivos solamente
    yhat = prob_predictions[:, 1]
    fpr, tpr, thresholds = roc_curve([elem == 1 for elem in y_test], yhat)
    gmeans = np.sqrt(tpr * (1-fpr))
    ix = np.argmax(gmeans)

    # graficamos la curva roc del modelo
    ax1.plot([0,1], [0,1], linestyle='--', label='No Skill')
    ax1.plot(fpr, tpr, marker='.', label=str(model)[:10])
    ax1.scatter(fpr[ix], tpr[ix], s=100, marker='o', color='black', label='Best')
    # axis labels
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title(str(model)[:10]+ ' - Best Threshold=%f, G-Mean=%.3f'% (thresholds[ix], gmeans[ix]))
    ax1.legend()
    
     #calculate pr-curve
    precision, recall, thresholds = precision_recall_curve(y_test, yhat)
    
    
    # plot the pr- curve for the model
    no_skill = len(y_test[y_test==1]) / len(y_test)
    ax2.plot([0,1], [no_skill,no_skill], linestyle='--', label='No Skill') 
    ax2.plot(recall, precision, marker='.', label=str(model)[:10]) 
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.legend()
    
    # show the plot
    plt.tight_layout() 
    
    f, (ax3, ax4) = plt.subplots(1, 2, figsize=(10, 5), dpi=90)
    #graficamos la curva de ganancia
    skplt.metrics.plot_cumulative_gain(y_test, prob_predictions, ax=ax3) 
    ax3.set_title(str(model)[:10]+" - Cumulative Gains curve")
    
     # graficamos la curva lift
    skplt.metrics.plot_lift_curve(y_test, prob_predictions, ax=ax4) 
    ax4.set_title(str(model)[:10]+" - Lift_curve")
    
    
    # show the plot
    plt.tight_layout()
    
    

def dame_matrices(modelo_entrenado):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    model = modelo_entrenado
    cm = confusion_matrix(y_test, y_pred=model.predict(X_test).astype(int), normalize=None)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    disp.plot(cmap='Blues', ax=ax1)
    disp.ax_.set_title((str(model)[:10]+" - Confusion matrix, without normalization"))

    cm2 = confusion_matrix(y_test, y_pred=model.predict(X_test).astype(int), normalize='true')
    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2)
    disp2.plot(cmap='Blues', ax=ax2)
    disp2.ax_.set_title((str(model)[:10]+" - Normalized confusion matrix"))
    
    
    plt.tight_layout() 
    
    
def matrices_threshold(modelo_entrenado):
    model = modelo_entrenado
    # predecimos probabilidades
    predict_p = model.predict_proba(X_test)[:,1]
    fpr, tpr, thresholds = roc_curve(y_test, predict_p)
    gmeans = np.sqrt(tpr*(1-fpr))
    index = np.argmax(gmeans)
    #predictions and optimal thresholds
    y_pred = np.where(predict_p>thresholds[index], 1 ,0)
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    cm = confusion_matrix(y_test, y_pred=y_pred, normalize=None)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    disp.plot(cmap='Blues', ax=ax1)
    disp.ax_.set_title("Applicando threshold \n " + str(model)[:10]+" - Confusion matrix, without normalization")

    cm2 = confusion_matrix(y_test, y_pred=y_pred, normalize='true')
    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2)
    disp2.plot(cmap='Blues', ax=ax2)
    disp2.ax_.set_title("Applicando threshold \n " + str(model)[:10]+" - Normalized confusion matrix")
    
    
    plt.tight_layout() 
    

def duplicate_columns(frame):
    groups = frame.columns.to_series().groupby(frame.dtypes).groups 
    dups = []

    for t, v in groups.items():
        cs = frame[v].columns 
        vs = frame[v]
        lcs = len(cs)
        
        for i in range(lcs):
            ia = vs.iloc[:,i].values 
            for j in range(i+1, lcs):
                ja = vs.iloc[:,j].values 
                if np.array_equal(ia, ja):
                    dups.append(cs[i]) 
                    break
                
    return dups


def plot_feature(col_name, isNumeric): 
    """
    Visualize a variable with and without faceting on the fatalities yes/no
    - col_name is the variable name
    - numeric is True if the variable is numeric, False otherwise 
    """

    f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,3), dpi=90)

    # Plot without TARGET
    if isNumeric:
        sns.distplot(df_data.loc[df_data[col_name].notnull(), col_name], kde=False, ax=ax1)
    else:
        sns.countplot(df_data[col_name], order=sorted(df_data[col_name].unique()), color='#5975A4', saturation=1, ax=ax1, hue=df_data['TARGET'], palette = "Set2")
    ax1.set_xlabel(col_name) 
    ax1.set_ylabel('Count') 
    ax1.set_title("Conductores involucrados en accidentes totales")

    # Plot with TARGET
    if isNumeric:
        sns.kdeplot(x=df_data[col_name].loc[df_data['TARGET']== 0],
           shade= True, label='no_fatalities')
        sns.kdeplot(x=df_data[col_name].loc[df_data['TARGET']== 1],
           shade= True, label='fatalities')
        plt.legend()
        ax2.set_ylabel('')
        ax2.set_title('Distribución segun '+col_name)
        
    
    else:
        plt_data = df_data.groupby(col_name)['TARGET'].value_counts(normalize=True).to_frame('proportion').reset_index() 
        sns.barplot(x = col_name, y = 'proportion', hue= "TARGET", data = plt_data, saturation=1, ax=ax2) 
        ax2.set_ylabel('TARGET fraction')
        ax2.set_xlabel(col_name)
        ax2.set_title('% fatalidad entre los accidentes segun '+col_name)
    plt.tight_layout()


def plot_feature_fatal(col_name): 
    f, ax = plt.subplots()
    sns.countplot(df_data[df_data['TARGET']== 1][col_name], order=sorted(df_data[df_data['TARGET']== 1][col_name].unique()), color='#5975A4', saturation=1, palette = "Set2")
    ax.set_xlabel(col_name) 
    ax.set_ylabel('Count') 
    ax.set_title("Conductores involucrados en accidentes mortales")
    
    plt.tight_layout()

def evaluate_model(y_test, y_pred, y_pred_proba = None):
    if y_pred_proba is not None:
        print('ROC-AUC score of the model: {}'.format(roc_auc_score(y_test, y_pred_proba[:, 1])))
    print('Accuracy of the model: {}\n'.format(accuracy_score(y_test, y_pred)))
    print('Classification report: \n{}\n'.format(classification_report(y_test, y_pred)))
    print('Confusion matrix: \n{}\n'.format(confusion_matrix(y_test, y_pred)))