import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import GaussianNB
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
# from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler, Normalizer
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, f1_score
import seaborn as sns
import os
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')

def SS_it(data_input_train, data_input_test, unproc_list=['Group'], SS_type='SS', if_01_proc=False, if_both_trainset_param=False):
    '''
    unproc_list是不想处理的列组成的list, 一般设为[目标列]即可
    '''

    # 变量类型筛查
    data_unprocess = data_input_train.loc[:,unproc_list]
    data_toprocess = data_input_train.drop(unproc_list, axis=1)
    columns_proc = data_toprocess.columns
    index_01 = []
    index_SS = []
    for col in columns_proc:
        col_now = data_toprocess[col]
        if len(col_now[col_now==0]) + len(col_now[col_now==1]) == len(col_now):
            index_01.append(col)
        else:
            index_SS.append(col)


    data = data_input_train
    data_unprocess = data.loc[:,unproc_list]

    data_01 = data.loc[:, index_01]
    data_SS = data.loc[:, index_SS]
    
    if if_01_proc:
        data_01 = 2*(data_01-0.5)

    column_SS = data_SS.columns

    if SS_type == 'minmax':
        sccc = MinMaxScaler(feature_range=(0, 1))
    elif SS_type == 'normalize':
        sccc = Normalizer()
    else:
        sccc = StandardScaler()
    
    data_SSed = sccc.fit_transform(data_SS)
    data_SSed_df = pd.DataFrame(data_SSed, columns=column_SS, index=data.index)

    data_SS_train = pd.concat([data_unprocess, data_SSed_df, data_01], axis=1)

    data = data_input_test
    data_unprocess = data.loc[:,unproc_list]
    data_01 = data.loc[:, index_01]
    data_SS = data.loc[:, index_SS]
    
    if if_01_proc:
        data_01 = 2*(data_01-0.5)

    column_SS = data_SS.columns

    if if_both_trainset_param:
        data_SSed = sccc.transform(data_SS)
    else:
        data_SSed = sccc.fit_transform(data_SS)
    data_SSed_df = pd.DataFrame(data_SSed, columns=column_SS, index=data.index)

    data_SS_test = pd.concat([data_unprocess, data_SSed_df, data_01], axis=1)

    return data_SS_train, data_SS_test

def smote_it(data_input, data_input_test, aim='Group', smo_type='smote', rate_smo=1):
    features = data_input.drop([aim], axis=1)
    targets = data_input[aim]
    # col_fea = features.columns

    ada = SMOTE(
        sampling_strategy=rate_smo
    )

    features2, targets2 = ada.fit_resample(features, targets)

    data_smote_train = pd.concat([targets2, features2], axis=1)
    # data_smote_test = data_input_test
    return data_smote_train, data_input_test

def cut_traintest_set(df, aim='Group', random_st=0, cut_rate=0.2):
    df0 = df.loc[df[aim]==0,:]
    df1 = df.loc[df[aim]==1,:]

    data_train_0, data_test_0 = train_test_split(df0, test_size=cut_rate, random_state=random_st)
    data_train_1, data_test_1 = train_test_split(df1, test_size=cut_rate, random_state=random_st)
    data_train = pd.concat([data_train_0, data_train_1])
    data_test = pd.concat([data_test_0, data_test_1])

    return data_train, data_test

def get_out_df(classifier, X_test0, y_test0, model_name):
    # 模型测试结果
    f_y_pred = classifier.predict(X_test0)
    f_y_proba = classifier.predict_proba(X_test0)[:, 1]
    f_y_true = y_test0

    # 计算模型指标
    f_acc = accuracy_score(f_y_true, f_y_pred)
    tn, fp, fn, tp = confusion_matrix(f_y_true, f_y_pred).ravel()
    f_auc = roc_auc_score(f_y_true, f_y_proba)
    f_sec = tp / (tp + fn)
    f_scf = tn / (fp + tn)
    f_pcs = tp / (tp + fp)
    f_npv = tn / (fn + tn)

    audf = pd.DataFrame((f_acc, f_auc, f_sec, f_scf, f_pcs, f_npv), columns=[model_name])

    return audf

st.image(r'treeandawn_title.jpg')
st.title(' 🩺 Data Check')
# X_col = [
#     'TS(um)', 'ST(um)', 'SN(um)', 'NS(um)', 'NI(um)', 'IN(um)',
#     'IT(um)', 'TI(um)', 'All(um)', 'Thk Peripapillary',
#     'Density/Whole Image_RPC', 'Whole Image-Capillary_RPC',
#     'Inside Disc-All_RPC', 'Inside Disc-Capillary_RPC',
#     'Peripapillary-All_RPC', 'Peripapillary-Capillary_RPC',
#     'GH S-Hemi-All_RPC', 'GH I-Hemi-All_RPC', 'GH S-Hemi-Capillary_RPC',
#     'GH I-Hemi-Capillary_RPC', 'GH-NS_RPC', 'GH-NI_RPC', 'GH-IN_RPC',
#     'GH-IT_RPC', 'GH-TI_RPC', 'GH-TS_RPC', 'GH-ST_RPC', 'GH-SN_RPC',
#     'SRCP-Whole', 'SRCP-Fovea', 'SRCP-Para', 'SRCP-Para T', 'SRCP-Para S',
#     'SRCP-Para N', 'SRCP-Para I', 'DRCP-Whole', 'DRCP-Fovea', 'DRCP-Para',
#     'DRCP-Para T', 'DRCP-Para S', 'DRCP-Para N', 'DRCP-Para I', 'GCC-Fovea',
#     ' GCC T thick 13', ' GCC S thick 13', ' GCC N thick 13',
#     ' GCC I thick 13', ' GCC whole thick 13', ' GCC whole thick 03',
#     'RETINA central thick', 'RETINA T thick 13', 'RETINA S thick 13',
#     'RETINA N thick 13', 'RETINA I thick 13', 'RETINA whole thick 13',
#     'RETINA whole thick 03'
# ]

st.markdown("## Choose the features")

col1, col2 = st.columns(2)
with col1:
    selected_cols1 = st.multiselect(
        'Thick', 
        ['Thk Peripapillary', 'All(um)', 'TS(um)', 'ST(um)', 'SN(um)', 
         'NS(um)', 'NI(um)', 'IN(um)','IT(um)', 'TI(um)'], 
        ['ST(um)', 'IT(um)']
    )
    selected_cols2 = st.multiselect(
        'XXX(long)_RPC', 
        ['Density/Whole Image_RPC', 'Whole Image-Capillary_RPC',
         'Inside Disc-All_RPC', 'Inside Disc-Capillary_RPC', 
         'Peripapillary-All_RPC', 'Peripapillary-Capillary_RPC'], 
        ['Inside Disc-All_RPC']
    )
    selected_cols3 = st.multiselect(
        'GH', 
        ['GH S-Hemi-All_RPC', 'GH I-Hemi-All_RPC', 'GH S-Hemi-Capillary_RPC',
         'GH I-Hemi-Capillary_RPC', 'GH-NS_RPC', 'GH-NI_RPC', 'GH-IN_RPC',
         'GH-IT_RPC', 'GH-TI_RPC', 'GH-TS_RPC', 'GH-ST_RPC', 'GH-SN_RPC'], 
        ['GH-ST_RPC']
    )
    selected_cols8 = st.multiselect(
        'SRCP', 
        ['SRCP-Whole', 'SRCP-Fovea', 'SRCP-Para', 'SRCP-Para T', 
         'SRCP-Para S','SRCP-Para N', 'SRCP-Para I'], 
        ['SRCP-Para I']
    )
with col2:
    selected_cols4 = st.multiselect(
        'DRCP', 
        ['DRCP-Whole', 'DRCP-Fovea', 'DRCP-Para',
         'DRCP-Para T', 'DRCP-Para S', 'DRCP-Para N', 'DRCP-Para I'], 
        ['DRCP-Fovea', 'DRCP-Para N']
    )
    selected_cols5 = st.multiselect(
        'GCC', 
        ['GCC-Fovea', ' GCC T thick 13', ' GCC S thick 13', 
         ' GCC N thick 13', ' GCC I thick 13', ' GCC whole thick 13', 
         ' GCC whole thick 03']
    )
    selected_cols6 = st.multiselect(
        'RETINA', 
        ['RETINA central thick', 'RETINA T thick 13', 'RETINA S thick 13',
         'RETINA N thick 13', 'RETINA I thick 13', 'RETINA whole thick 13',
         'RETINA whole thick 03'], 
        ['RETINA I thick 13', 'RETINA N thick 13', 'RETINA whole thick 13']
    )
    # 外院数据无临床数据
    # selected_cols7 = st.multiselect(
    #     'clinic', 
    #     ['exopha', 'IOP', 'MD', 'spherical equivalent', 
    #      'LogMAR', 'Age', 'Gender', 'CAS']  
    # )

selected_cols = selected_cols1 + selected_cols2 + selected_cols3 + selected_cols4 + selected_cols5 + selected_cols6 + selected_cols8 # + selected_cols7
st.markdown('---')
st.markdown("## Control the params")

col11, col22, col33 = st.columns(3)

with col11:
    cut_rate = st.slider('Train/Test Cut Rate', value=0.3, min_value=0.2, max_value=0.5, step=0.01)
with col22:
    smote_rate = st.slider('SMOTE Rate', value=1.0, min_value=0.6, max_value=1.0, step=0.01)
with col33:
    random_seed = st.number_input('Random Seed', value=43, min_value=0, max_value=100, step=1)
if_run = st.button('Run !', key=102)    
st.markdown('---')

if if_run:

    st.title('Machine Learning')

    random_seed = 43

    L = 7
    process_text = 'Read Data... ' + '{:.1%}'.format(0/L) + ' (0/' + str(L) + ')'
    bar = st.progress(0.0, text=process_text)
    
    # data input
    data = pd.read_csv('data4ML_V3.csv', index_col=0).drop(
        ['exopha', 'IOP', 'MD', 'spherical equivalent', 
        'LogMAR', 'Age', 'Gender', 'CAS'], axis=1
    )
    data_outer = pd.read_csv('data_outer.csv', index_col=0)
    
    ttt = 1
    process_text = 'Data Pre-Processing... ' + '{:.1%}'.format(ttt/L) + ' (' + str(ttt)+ '/' + str(L) + ')'
    bar.progress(ttt/L, text=process_text)

    aim='group'
    data_trainori, data_test = cut_traintest_set(data, aim=aim, cut_rate=cut_rate, random_st=random_seed)
    
    ttt = 2
    process_text = 'Data Pre-Processing... ' + '{:.1%}'.format(ttt/L) + ' (' + str(ttt)+ '/' + str(L) + ')'
    bar.progress(ttt/L, text=process_text)
    
    data_temp_ori_dddd = data_trainori.copy()
    data_trainori, data_test = SS_it(data_trainori, data_test, unproc_list=[aim], if_both_trainset_param=True)
    data_temp_ori_dddd, data_outer = SS_it(data_temp_ori_dddd, data_outer, unproc_list=[aim], if_both_trainset_param=True)
    
    ttt = 3
    process_text = 'Data Pre-Processing... ' + '{:.1%}'.format(ttt/L) + ' (' + str(ttt)+ '/' + str(L) + ')'
    bar.progress(ttt/L, text=process_text)

    data_train, data_test = smote_it(data_trainori, data_test, aim=aim, rate_smo=smote_rate)

    ttt = 4
    process_text = 'Data Pre-Processing... ' + '{:.1%}'.format(ttt/L) + ' (' + str(ttt)+ '/' + str(L) + ')'
    bar.progress(ttt/L, text=process_text)

    X = data_train[selected_cols]
    y = data_train[aim]
    X_test = data_test[selected_cols]
    y_test = data_test[aim]
    X_outer = data_outer[selected_cols]
    y_outer = data_outer[aim]
        
    models = [
        "LogisticRegression(max_iter=5000)",
        # "DecisionTreeClassifier()",
        # "RandomForestClassifier()",
        # "MLPClassifier()",
        # "GaussianNB()",
        "SVC(kernel='rbf', probability=True)",
        # "LGBMClassifier()",
        # "XGBClassifier(max_depth=5, learning_rate=0.1, objective='binary:logistic', nthread=-1, scale_pos_weight = len(y[y == 0])/len(y[y == 1]))",
        # "KNeighborsClassifier(n_neighbors=5)"
    ]

    model_names = [
        'Logistic',
        # 'Decision Tree',
        # 'Random Forest',
        # 'Neural Network',
        # 'Bayes',
        'SVM',
        # 'LightGBM',
        # 'XGBoost',
        # 'KNeighbor',
    ]

    outcome = pd.DataFrame()
    # infos_list = []
    columns_input = X.columns
    for j in range(0, len(models)):
        classifier = eval(models[j])
        model_name = model_names[j]
        
        classifier.fit(X, y)

        # infos_list.append(classifier.get_params())

        model_name = model_names[j]
        out_df_train = get_out_df(classifier, X, y, model_name+"_train")
        out_df_inner = get_out_df(classifier, X_test, y_test, model_name+"_inner")
        out_df_outer = get_out_df(classifier, X_outer, y_outer, model_name+"_outer")
        
        out_df_budf = pd.concat([out_df_train, out_df_inner], axis=1)
        out_df_audf = pd.concat([out_df_budf, out_df_outer], axis=1)
        # 保存模型指标值
        
        outcome = pd.concat([outcome, out_df_audf], axis=1)  # 这里是保存了每个模型的指标值的，如果需要可以拿来输出
        
        process_text = 'Model Running... ' + '{:.1%}'.format((j+5)/L) + ' (' + str(j+5)+ '/' + str(L) + ')'
        bar.progress((j+5)/L, text=process_text)

        if model_name == 'Logistic':
            importance = classifier.coef_
            feature_names = classifier.feature_names_in_
    outcome.index = ['ACC', 'AUC', 'SEC', 'SCF', 'PCS', 'NPV']

    ttt = 7
    process_text = 'Done! ' + '{:.1%}'.format(ttt/L) + ' (' + str(ttt)+ '/' + str(L) + ')'
    bar.progress(ttt/L, text=process_text)

    st.write("  ")

    col_001, col111, col221, col331 = st.columns(4)

    if outcome.loc['AUC', 'SVM_inner'] > outcome.loc['AUC', 'Logistic_inner']:
        out_choose_one = "SVM"
    else:
        out_choose_one = "Logistic"
    
    with col_001:
        st.metric(
            label='Best MODEL', 
            value=out_choose_one
        )
    with col111:
        st.metric(
            label='Best AUC in TRAIN-SET', 
            value=outcome.loc['AUC', out_choose_one+'_train']
        )
    with col221:
        st.metric(
            label='Best AUC in INNER-TEST-SET', 
            value=outcome.loc['AUC', out_choose_one+'_inner']
        )
    with col331:
        st.metric(
            label='Best AUC in OUTER-TEST-SET', 
            value=outcome.loc['AUC', out_choose_one+'_outer']
        )

    st.table(outcome)

    st.markdown("---")

    st.markdown("## Coef of Logistic")

    st.table(pd.DataFrame(importance, columns=columns_input, index=['Coef of Logis']).T.sort_values(by='Coef of Logis', ascending=False))

    st.markdown("---")
