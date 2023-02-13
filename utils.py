
import pandas as pd
import plotly.express as px
from xgboost import XGBClassifier
from xgboost import plot_importance
#from matplotlib import pyplot
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

def get_plot(quest,data):
    #bar_tab=data.groupby([quest,'cluster']).count()
    #bar_tab=bar_tab[[bar_tab.columns[0]]]
    #bar_tab=bar_tab.rename(columns={bar_tab.columns[0]:'TOTAL'}).reset_index()
    #creating crosstab with variables and cluster
    a=pd.crosstab(data['cluster'],data[quest]).T.reset_index()
    # unpivoting
    bar_tab=pd.melt(a,id_vars=quest,value_vars=list(a.columns))
    total_bar=bar_tab.groupby(['cluster']).sum()
    print(total_bar)
    #assigning percentage for every cluster
    for i in bar_tab.index:
        cluster_n=bar_tab.loc[i,'cluster']
        bar_tab.loc[i,'perc']=bar_tab.loc[i,'value']/total_bar.loc[cluster_n,'value']
    color_map={i:j for i,j in enumerate(px.colors.qualitative.D3)}
    #plotting
    fig=px.bar(bar_tab,x=quest,y='perc',color='cluster',barmode='group',color_discrete_map=color_map)
    fig.update_layout(title=quest)
    return fig



def train_model(data,model_type='LogisticReg'):
        # oh encoding
        dummies=pd.get_dummies(data[data.columns[:-1]])
        # splitting data
        X_train, X_test, y_train, y_test = train_test_split(dummies, data['cluster'].values, test_size=0.2, random_state=0)
        # algorithm menu
        if model_type=='Xgbc':
            model=XGBClassifier()
        if model_type=='Catboost':
            model=CatBoostClassifier()
        if model_type=='RandomForest':
            model=RandomForestClassifier(random_state=0)
        if model_type=='LogisticReg':
            model= LogisticRegression(random_state=0,max_iter=1000)
        

        #model=CatBoostClassifier()
        #model= LogisticRegression(random_state=0,max_iter=1000)
        #model=XGBClassifier()
        #model=RandomForestClassifier(random_state=0)
        #model = AutoSklearnClassifier(time_left_for_this_task=2*60, per_run_time_limit=30, n_jobs=8)
        model.fit(X_train,y_train)
        #para ensembles y arboles
        #dict_importances={dummies.columns[i]:[model.feature_importances_[i]] for i in range(len(model.feature_importances_))}
        #para logisitc

        # depeding of each algorithm, importances are mapped in different way 
        if model_type=='LogisticReg':
            dict_importances={dummies.columns[i]:[model.coef_[0][i]] for i in range(len(model.coef_[0]))}
        else:
            #para ensembles y arboles
            dict_importances={dummies.columns[i]:[model.feature_importances_[i]] for i in range(len(model.feature_importances_))}
        #oranizing scores
        features=pd.DataFrame(dict_importances).T
        features=features.rename(columns={0:'importance'})
        imp_df=features.sort_values(by='importance',ascending=False)
        imp_df['abs_importance']=imp_df['importance'].apply(lambda x:abs(x))
        imp_df=imp_df.sort_values(by='abs_importance',ascending=False)
        preds= model.predict(X_test)

        return classification_report(y_test,preds,output_dict=True),confusion_matrix(y_test, preds),imp_df