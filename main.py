import streamlit as st
import credit_scorecard
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, auc, precision_recall_curve
from scipy import stats

st.sidebar.title('CREDIT SCORING APP')
st.sidebar.write('This is a mini project using Home \nEquity Loans dataset from\ncreditriskanalytics.net to build\na Credit Scorecard model.')
st.sidebar.write('Thank you for your consideration. You can find out all source codes of this project and my others projects via https://github.com/tmtuong')
st.sidebar.write('My contact: tmtuong.02@gmail.com')

selected_tab = st.sidebar.selectbox('**MENU**',["Calculate Your Credit Score","Our Model","About Credit Scoring Model"])
@st.cache_data
def load_data():
    return pd.read_csv('http://www.creditriskanalytics.net/uploads/1/9/5/1/19511601/hmeq.csv', header=0, sep=',')

data = load_data()

@st.cache_data
def get_processed_data(data):
    df, x_col = credit_scorecard.fill_missing(data, 'BAD')
    nomial_col = ['DEROG', 'DELINQ']
    temp = x_col[0]
    x_col = list(x_col[1:])
    x_col.append(temp)
    df_WOE, WOE_dict, columns = credit_scorecard.iv_score(df, x_col, nomial_col)
    df, X, y = credit_scorecard.ml_input(WOE_dict, df)
    return df_WOE, WOE_dict, df, X, y,columns

df_WOE, WOE_dict, df, X, y,columns = get_processed_data(data)

@st.cache_resource
def credit_model(df_WOE, WOE_dict, df, X, y,columns):
    ids = np.arange(X.shape[0])
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(X, y, ids, test_size=0.2, stratify=y, shuffle=True, random_state=1)
    alpha,coef_, fpr, tpr, thres_roc, roc_auc, precision, recall, thres_pre, y_pred_prob_test, acc_test, acc_train = credit_scorecard.build_model(X_train, X_test, y_train, y_test, id_train, id_test)
    df_WOE = credit_scorecard.get_beta(X_train,alpha,coef_, columns, WOE_dict)
    df = credit_scorecard.cal_score(df, columns, WOE_dict, df_WOE)
    return df, y_pred_prob_test, fpr, tpr, thres_roc, roc_auc, precision, recall, thres_pre, y_test, acc_test, acc_train,alpha,coef_,X_train
df, y_pred_prob_test, fpr, tpr, thres_roc, roc_auc, precision, recall, thres_pre, y_test, acc_test, acc_train,alpha,coef_,X_train = credit_model(df_WOE, WOE_dict, df, X, y,columns)



scorecard = credit_scorecard.score_segment(df)
# Display content based on the selected tab
if selected_tab == "About Credit Scoring Model":
    st.sidebar.write('**ScoreCard Model**')
    st.sidebar.write('**Dataset**')
    st.sidebar.write('**Data Preparation**')
    st.sidebar.write('**Logistic Regression Model**')
    st.sidebar.write('**Kolmogorov-Smirnov Testing**')
    st.sidebar.write('**Credit Scoring**')
    st.sidebar.write('**Scorecard Building**')
    st.sidebar.write('**Reference**')

    st.title('About Credit Scoring Model')
    st.write("This section provides you with a comprehensive understanding of scorecard models and how to build them from stratch.")
    st.header('ScoreCard Model')
    st.subheader('Overview')
    st.write('The ScoreCard model is a type of model applied in various fields such as finance, business, and social management. The ScoreCard model quantifies individual or organizational profiles into a credit score based on the likelihood of specific events occurring, such as defaulting on a loan or legal violations. This credit score enables financial institutions or governments to offer better products and services to individuals or entities with higher credit scores and less favorable terms to those with lower credit scores.')
    st.subheader('Model Application')
    st.write('**In Banking Risk Management**')
    st.write('''In credit operations, the goal of the ScoreCard model is to assess the repayment capability of borrowers in the future. Input for the ScoreCard model includes information from customer profiles, predominantly collected from credit bureaus or the bank's data warehouse. Information is categorized into basic groups:

- Demographic Data: Includes personal characteristics like education level, income, gender, age, occupation, marital status, family size, dependents, etc.

- Credit History: Information managed centrally at credit bureaus, encompassing a borrower's credit-related data from various banks.

- Transaction Data: Transaction history on credit cards or ATMs provides insight into the borrower's financial capacity.

- Collateral Information: Pertains to collateral accompanying secured loans, acting as risk coverage in case of default.''')
    st.write('**In Social Credit Rating**')
    st.write('''Beyond credit risk, the ScoreCard model is also applied in some countries to develop social credit rating systems. China, for example, integrates data from various sources like surveillance cameras, financial data, demographics, and social networks to create a comprehensive social credit score. This score influences citizens' access to public services, job opportunities, and compliance with laws, contributing to a reduction in crime rates.

However, social credit systems have raised concerns regarding individual privacy and potential discrimination against those with lower scores. Nonetheless, the model has shown success in China.''')
    st.subheader('Building a ScoreCard Model')
    st.write('''In developing countries like Vietnam, where credit is a significant profit source for banks, managing credit risks is crucial. The ScoreCard model helps quantify and manage these risks. The model uses machine learning to estimate the probability of credit-related events, such as defaults or late payments. These models are relatively simple, using around 10 to 20 input variables. Data comes from credit bureaus and internal bank records.

To construct a credit ScoreCard model:

1. Compile default and non-default records.
2. Preprocess data to transform input variables into predictive features.
3. Use algorithms like logistic regression for modeling. Logistic regression offers explanatory power.
4. Evaluate individual variables' creditworthiness, and aggregate to obtain a final credit score.

While other algorithms like neural networks and random forests might provide better classification results, their lack of interpretability makes them less suitable for practical credit ScoreCard models.''')
    st.write('**WOE approach**')
    st.write('''WOE is a feature engineering and selection technique often applied in scorecard models. It ranks variables into strong, medium, weak, or non-influential categories based on their predictive power for bad debt. The ranking criterion is the Information Value (IV), calculated using the WOE method. WOE creates feature values for each variable, measuring the difference in distribution between "good" and "bad" samples.

For continuous variables, WOE assigns observations to bins. The bins consist of contiguous ranges determined to have roughly equal numbers of observations. For categorical variables, each class can be considered a bin, or some classes can be grouped into bins if they have few observations. The difference in distribution between good and bad samples is measured using the WOE value, which is the logarithm of the ratio of the percentage of "good" to "bad."''')

    st.header('Dataset')
    st.write('''The "hmeq" dataset consists of features related to overdue loan information for 5960 home mortgage loans. These loans are secured by the borrowers using their owned properties as collateral. The dataset includes the following fields:

- BAD: 1 = Loan application is in violation or the borrower is defaulting; 0 = Loan application is in good standing and the borrower is paying back.
- LOAN: The requested loan amount.
- MORTDUE: The amount due on the existing mortgage.
- VALUE: The current value of the property.
- REASON: DebtCon = Debt consolidation; HomeImp = Home improvement.
- JOB: The type of job/occupation.
- YOJ: The number of years in the current job/occupation.
- DEROG: The number of derogatory reports.
- DELINQ: The number of delinquent credit lines.
- CLAGE: The age of the oldest credit line in months.
- NINQ: The number of recent credit inquiries.
- CLNO: The number of credit lines.
- DEBTINC: The debt-to-income ratio.
There are a total of 12 input variables, including both numeric and categorical ones. The dataset has a sufficiently large number of observations to build a credit scorecard model.''')
    st.header('Data Preparation')
    st.write('''Firstly, we need to import necessary libraries. It then reads a CSV dataset from a specified URL using Pandas. The dataset contains credit risk-related information such as loan details, property values, and borrower characteristics. The data is loaded into a DataFrame named df, which can be used for data analysis and visualization.''')
    st.code("""
#Import các thư viện
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, auc, precision_recall_curve
from scipy import stats

#Import data
df = pd.read_csv('http://www.creditriskanalytics.net/uploads/1/9/5/1/19511601/hmeq.csv', header = 0, sep = ',')
    """, language="python")

    st.write("Addressing missing values in two stages: first, it identifies columns with numeric data types and replaces missing values with the mean of their respective columns; second, it identifies columns with object data types (typically categorical) and fills in missing values with the string 'Missing'. These actions enhance the data's quality and completeness, readying it for further analysis or modeling.")
    st.code("""
#Fill missing value
columns_num = df.select_dtypes(['float', 'int']).columns
df[columns_num] = df[columns_num].apply(lambda x: x.fillna(x.mean()), axis=0)

columns_obj = df.select_dtypes(['object']).columns
df[columns_obj] = df[columns_obj].apply(lambda x: x.fillna('Missing'), axis=0)
            
#Split x and y
y_col = 'BAD'
x_col = df.columns.drop(y_col)
""", language="python")
    st.write("""Calculating the Information Value (IV) and performs data transformation for a specified column in a given DataFrame. Depending on the data type of the specified column (numeric or categorical), the function performs binning using either quantiles or custom bins. For numeric columns, the function computes metrics such as counts, the number of 'Bad' outcomes, the number of 'Good' outcomes, the 'Good/Bad' ratio, '%Bad', '%Good', Weight of Evidence (WOE), and the IV. It then returns a DataFrame containing these calculated values along with the IV value for the column. The IV quantifies the predictive power of the column regarding the 'Bad' outcomes in the dataset, with higher IV values indicating stronger predictive ability.""")
    st.code('''
#Calculate IV, WOE
def get_iv(data, col, bin, qcut = None):
    MAX_VAL = 999999999
    MIN_VAL = -MAX_VAL
    coltype = data[col].dtype

    if coltype in ['float', 'int']:
      if qcut is None:
        try:
          bins, thres = pd.qcut(data[col], q = bin, retbins=True)
          # Thay thế threshold đầu và cuối của thres
          thres[0] = MIN_VAL
          thres[-1] = MAX_VAL
          bins, thres = pd.cut(data[col], bins=thres, retbins=True)
          data['bins'] = bins
        except:
          # print('n_bins must be lower to bin interval is valid!')
          pass
      else:
        bins, thres = pd.cut(data[col], bins=qcut, retbins=True)
        data['bins'] = bins
    elif coltype == 'object':
      data['bins'] = data[col]

    dfgb = data.groupby(['bins']).agg({col:'count','BAD':'sum'}).reset_index()

    dfgb.columns = ['bins','Obs','#Bad']
    dfgb['#Good'] = dfgb['Obs'] - dfgb['#Bad']


    dfgb['Good/Bad'] = dfgb['#Good']/dfgb['#Bad']
    dfgb['%Bad'] = dfgb['#Bad']/dfgb['#Bad'].sum()
    dfgb['%Good'] = dfgb['#Good']/dfgb['#Good'].sum()
    dfgb['WOE'] = np.log(dfgb['%Good']/dfgb['%Bad'])
    dfgb['IV'] = (dfgb['%Good']-dfgb['%Bad'])*dfgb['WOE']
    dfgb['COLUMN'] = col
    IV = dfgb['IV'].sum()
    dfgb.set_index(dfgb['bins'],inplace =True)
    # print('Information Value of {} column: {}'.format(col, IV))
    return dfgb, IV''',language='python')
    st.write('''Calculating the Weight of Evidence (WOE) and Information Value (IV) for different columns in the dataset. It categorizes columns as nominal, numeric, or categorical. For nominal columns, it uses custom bins, while for numeric columns, it determines the optimal bin count based on WOE points. For numeric columns, the fuction woe_point help finding out the best bins with condition is the woe increase and shrink among bins. IV values are calculated and printed for each column, and the results are stored in the WOE_dict dictionary.''')
    st.code('''
#Optimize function
def woe_point(x):
  point = 0
  for k in range(0,len(x)-2):
    if (x[k+1] > x[k]) and (x[k+1] > x[k+2]):
      point +=1
    elif (x[k+1] < x[k]) and (x[k+1]<x[k+2]):
      point +=1
  return point

#Get IV of all columns
nomial_col = ['DEROG','DELINQ']
WOE_dict = {}
for i in x_col:
  if i in nomial_col:
    df_woe, iv = get_iv(df, i, 5, [-999,2,999])
    print(f'IV of {i}: {iv}')

  elif df[i].dtypes in ['float', 'int']:
    point_dict = {}
    for n in range(3,11):
      try:
        df_woe, iv = get_iv(df, i, n)
        point_dict[n] = woe_point(list(df_woe['WOE']))
      except:
        point_dict[n] = -1
        continue
    max_key = min(sorted(point_dict.items(), key=lambda item: (-item[1], item[0])))[0]
    df_woe, iv = get_iv(df, i, max_key)
    print(f'IV of {i}: {iv}')

  elif df[i].dtypes == 'object':
    df_woe, iv = get_iv(df, i, 5)
    print(f'IV of {i}: {iv}')

  WOE_dict[i] = {'table':df_woe, 'IV':iv}

#Rank IV
columns = []
IVs = []
for col in x_col:
  if col != 'BAD':
    columns.append(col)
    IVs.append(WOE_dict[col]['IV'])
df_WOE = pd.DataFrame({'column': columns, 'IV': IVs})

def _rank_IV(iv):
  if iv <= 0.02:
    return 'Useless'
  elif iv <= 0.1:
    return 'Weak'
  elif iv <= 0.3:
    return 'Medium'
  elif iv <= 0.5:
    return 'Strong'
  else:
    return 'suspicious'

df_WOE['rank']=df_WOE['IV'].apply(lambda x: _rank_IV(x))
df_WOE.sort_values('IV', ascending=False)
''', language='python')
    st.write("""Applying the calculated Weight of Evidence (WOE) values to the corresponding columns in the dataset. It iterates through the keys of the WOE_dict dictionary, mapping the WOE values to each column in the dataset and creating new columns with "_WOE" appended to the original column names. The resulting DataFrame X contains only the columns with "_WOE" suffix, and the target variable 'BAD' is assigned to the variable y. Additionally, if any exceptions occur during the mapping process, the column name is printed.""")
    st.code('''
#Prepare model input
for col in WOE_dict.keys():
  try:
    key = list(WOE_dict[col]['table']['WOE'].index)
    woe = list(WOE_dict[col]['table']['WOE'])
    d = dict(zip(key, woe))
    col_woe = col+'_WOE'
    df[col_woe] = df[col].map(d)
  except:
    print(col)
X = df.filter(like='_WOE', axis = 1)
y = df['BAD']
''', language='python')
    st.header("Logistic Regression Model")
    st.write("""Spliting into training and testing sets using the train_test_split function. Then, a logistic regression model is initialized using the LogisticRegression class with specified parameters such as the solver, maximum iterations, penalty, and regularization strength. The model is fitted (trained) on the training data using the fit method.

After training, predictions are made for both the training and testing sets using the predict method. The accuracy of the model is calculated by comparing the predicted labels to the actual labels using the accuracy_score function. This accuracy is computed separately for both the training and testing sets.

Additionally, the model's predictive probabilities are obtained using the predict_proba method on the testing set, specifically for the positive class (class 1). The Receiver Operating Characteristic (ROC) curve is then constructed using the roc_curve function, and the Area Under the Curve (AUC) score is calculated using the auc function. These metrics help assess the model's performance in classification tasks, particularly in binary classification scenarios.""")
    st.code('''
#Train test split
ids = np.arange(X.shape[0])
X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(X, y, ids, test_size = 0.2, stratify = y, shuffle = True, random_state = 123)

#Fit model            
logit_model = LogisticRegression(solver = 'lbfgs', max_iter=1000, fit_intercept=True, tol=0.0001, C=1, penalty='l2')
logit_model.fit(X_train, y_train)

#Model evaluation            
y_pred_train = logit_model.predict(X_train)
acc_train = accuracy_score(y_pred_train, y_train)
y_pred_test = logit_model.predict(X_test)
acc_test = accuracy_score(y_pred_test, y_test)
y_pred_prob_test = logit_model.predict_proba(X_test)[:, 1]
fpr, tpr, thres = roc_curve(y_test, y_pred_prob_test)
roc_auc = auc(fpr, tpr)
''', language='python')
    st.write('Ploting ROC and Precision vs Recall Curve function')
    st.code('''
# ROC
def _plot_roc_curve(fpr, tpr, thres, auc):
    plt.figure(figsize = (10, 8))
    plt.plot(fpr, tpr, 'b-', color='darkorange', lw=2, linestyle='--', label='ROC curve (area = %0.2f)'%auc)
    plt.plot([0, 1], [0, 1], '--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.title('ROC Curve')

#Precision vs Recall
def _plot_prec_rec_curve(prec, rec, thres):
    plt.figure(figsize = (10, 8))
    plt.plot(thres, prec[:-1], 'b--', label = 'Precision')
    plt.plot(thres, rec[:-1], 'g-', label = 'Recall')
    plt.xlabel('Threshold')
    plt.ylabel('Probability')
    plt.title('Precsion vs Recall Curve')
    plt.legend()      

''', language='python')
    st.header('Kolmogorov-Smirnov Testing')
    st.write('''This is a test measuring the difference in distribution between GOOD and BAD based on various threshold ratios. If the model is capable of classifying GOOD and BAD effectively, the cumulative distribution function (CDF) curve between GOOD and BAD should exhibit significant separation. Conversely, if the model is weak and its predictions are only as good as random chance, the cumulative probability distribution curves of GOOD and BAD will closely align and approach the 45-degree diagonal line.

The Kolmogorov-Smirnov test assesses the hypothesis that the probability distributions of GOOD and BAD are not different. When the P-value is less than 0.05, the null hypothesis is rejected.''')
    st.write('''Calculating and ploting cumulative distribution percentages of predicted probabilities for different thresholds. It takes y_pred (predicted probabilities) and the number of bins (n_bins) as inputs. The function calculates the percentage of bad and good records within each bin defined by the thresholds. It then returns arrays cmd_BAD and cmd_GOOD containing the cumulative percentages of bad and good records respectively, along with the thresholds. Finally, these arrays are computed using the _KM function for the predicted probabilities y_pred_prob_test with 20 bins.''')
    st.code('''
#Calculate
def _KM(y_pred, n_bins):
  _, thresholds = pd.qcut(y_pred, q=n_bins, retbins=True)
  cmd_BAD = []
  cmd_GOOD = []
  BAD_id = set(np.where(y_test == 0)[0])
  GOOD_id = set(np.where(y_test == 1)[0])
  total_BAD = len(BAD_id)
  total_GOOD = len(GOOD_id)
  for thres in thresholds:
    pred_id = set(np.where(y_pred <= thres)[0])
    # Đếm % số lượng hồ sơ BAD có xác suất dự báo nhỏ hơn hoặc bằng thres
    per_BAD = len(pred_id.intersection(BAD_id))/total_BAD
    cmd_BAD.append(per_BAD)
    # Đếm % số lượng hồ sơ GOOD có xác suất dự báo nhỏ hơn hoặc bằng thres
    per_GOOD = len(pred_id.intersection(GOOD_id))/total_GOOD
    cmd_GOOD.append(per_GOOD)
  cmd_BAD = np.array(cmd_BAD)
  cmd_GOOD = np.array(cmd_GOOD)
  return cmd_BAD, cmd_GOOD, thresholds

cmd_BAD, cmd_GOOD, thresholds = _KM(y_pred_prob_test, n_bins=20)
            
#Plot curve
def _plot_KM(cmd_BAD, cmd_GOOD, thresholds):
  plt.figure(figsize = (10, 8))
  plt.plot(thresholds, cmd_BAD, 'y-', label = 'BAD')
  plt.plot(thresholds, cmd_GOOD, 'g-', label = 'GOOD')
  plt.plot(thresholds, cmd_BAD-cmd_GOOD, 'b--', label = 'DIFF')
  plt.xlabel('% observation')
  plt.ylabel('% total GOOD/BAD')
  plt.title('Kolmogorov-Smirnov Curve')
  plt.legend()

_plot_KM(cmd_BAD, cmd_GOOD, thresholds)
''')
    st.write('Using the stats.ks_2samp function from the scipy.stats module to perform the Kolmogorov-Smirnov (KS) two-sample test. This test is used to compare two sets of data (cmd_BAD and cmd_GOOD in this case) to determine if they come from the same distribution. The function returns the KS statistic and the p-value. This test is often used in statistics to assess the similarity between two samples.')
    st.code('''stats.ks_2samp(cmd_BAD, cmd_GOOD)''', language='python')
    st.header('Credit Scoring')
    st.write("Calculating credit scores using the Weight of Evidence (WOE) technique and logistic regression coefficients. It defines a function _CreditScore to calculate scores based on WOE and logistic regression parameters. The loop iterates through columns and their WOE values, calculating scores for each combination and storing them in a DataFrame. Another function, _total_score, computes the total score for an observation across all columns by searching for relevant scores in the DataFrame. The final loop applies this process to the dataset, generating total scores for each observation and adding them as a 'Score' column in the original DataFrame. This code aims to create a credit scoring model by leveraging WOE and logistic regression concepts.")
    st.code('''
#Scoring formula
def _CreditScore(beta, alpha, woe, n = 12, odds = 1/4, pdo = -50, thres_score = 600):
  factor = pdo/np.log(2)
  offset = thres_score - factor*np.log(odds)
  score = (beta*woe+alpha/n)*factor+offset/n
  return score
            
cols = []
features = []
woes = []
betas = []
scores = []

for col in columns:
  for feature, woe in WOE_dict[col]['table']['WOE'].to_frame().iterrows():
      cols.append(col)
      # Add feature
      feature = str(feature)
      features.append(feature)
      # Add woe
      woe = woe.values[0]
      woes.append(woe)
      # Add beta
      col_woe = col+'_WOE'
      beta = betas_dict[col_woe]
      betas.append(beta)
      # Add score
      score = _CreditScore(beta = beta, alpha = alpha, woe = woe, n = 12)
      scores.append(score)

df_WOE = pd.DataFrame({'Columns': cols, 'Features': features, 'WOE': woes, 'Betas':betas, 'Scores':scores})
df_WOE.head()

#Score for a attribute   
def _search_score(obs, col):
  feature = [str(inter) for inter in list(WOE_dict[col]['table'].index) if obs[col].values[0] in inter][0]
  score = df_WOE[(df_WOE['Columns'] == col) & (df_WOE['Features'] == feature)]['Scores'].values[0]
  return score

#Score for a profile
def _total_score(obs, columns = columns):
  scores = dict()
  for col in columns:
    scores[col] = _search_score(obs, col)
  total_score = sum(scores.values())
  return scores, total_score

#Score for all profile
total_scores = []
for i in np.arange(df[columns].shape[0]):
  obs = df[columns].iloc[i:(i+1), :]
  _, score = _total_score(obs)
  total_scores.append(score)
df['Score'] = total_scores
''', language='python')
    st.write("Visualizing the distribution of credit scores. In the first subplot, the distribution of scores for the entire dataset is shown. In the second subplot, the distribution of scores is divided into two groups: default (BAD=1) and non-default (BAD=0) cases. The distributions are visualized using kernel density estimation (KDE) curves and histograms. The left subplot shows the distribution of scores for all data points, while the right subplot compares the score distributions between default and non-default cases. The code aims to provide insights into how credit scores are distributed across the dataset and how they differ between default and non-default situations.")
    st.code('''
#Plot credit score distribution
plt.figure(figsize=(16, 4))
plt.subplot(121)
sns.distplot(df['Score'])
plt.title('Distribution Score of Total data')
plt.subplot(122)
sns.distplot(df[df['BAD']==1]['Score'], label='Default')
sns.distplot(df[df['BAD']==0]['Score'], label='Non-Default',
             kde_kws={"color": "r"},
             hist_kws={"color": "g", "alpha":0.5})
plt.legend(loc = 'lower right')
plt.title('Distribution Score in Default vs Non-Default')
''', language='python')
    st.header('Scorecard Building')
    st.write("Segmenting the data into credit score ranges, calculates statistics for each range, and creates a summary 'scorecard' DataFrame. It bins the 'Score' column into ranges, computes counts of 'BAD' cases and total scores within each bin, and calculates probabilities of 'BAD' cases. The resulting 'scorecard' DataFrame presents the range, rating, percentage of people, and percentage of fraud cases for each credit score bin. This aids in understanding the credit score distribution and its relationship with defaults.")
    st.code('''
#Segment score
df['credit_score_bin'], thres = pd.cut(df['Score'], bins=[0,350,500,650,800,1000], retbins=True)
            
#Calculate metrics
seg = df.groupby(['credit_score_bin']).agg({'BAD':'sum','Score':'count'}).reset_index()
seg['prob'] = seg['BAD']/seg['Score']
total_people = seg['Score'].sum()
seg['Score Range'] = [str(i) + ' - ' +str(j) for i,j in zip(thres[:-1], thres[1:])]
seg['% of people'] = [str(round(i/j*100,2)) + '%' for i, j in zip(seg['Score'], [total_people]*len(seg))]
seg['% Fraud'] = [str(round(i*100,2)) + '%' for i in seg['prob']]
            
#Label segment
seg['Rating'] = ['Very Poor','Poor','Fair','Good','Very Good']

#Final Scorecard            
scorecard = seg[['Score Range','Rating','% of people','% Fraud']]
''')

    st.write('The provided content consists solely of the source code for this project. To view the outcomes generated by this code, please refer to the "Our Model" section."')
    st.header('Reference')
    st.write('https://phamdinhkhanh.github.io/2020/01/17/ScoreCard.html?fbclid=IwAR2GdvHnkBr2i2mtlrGP87Oac6RmNuXow7WJko45QjWihOkZdBRWmOxQd9U#245-t%C3%ADnh-%C4%91i%E1%BB%83m-credit-score-cho-m%E1%BB%97i-feature')
elif selected_tab == 'Our Model':
    st.title('Our Model')
    st.write('Several outcomes from our model, along with supporting references, are presented to foster confidence in the quality of our efforts.')
    st.sidebar.write("")
    st.sidebar.write("**Scorecard**")
    st.sidebar.write("**Kolmogorov-Smirnov Testing**")
    st.sidebar.write("**Logistic Model**")
    st.sidebar.write("**Information Value**")

    st.header("Scorecard")
    st.write('')
    st.dataframe(scorecard)
    st.write('''The final scorecard presents credit score ranges and their corresponding classifications, along with the associated proportions of loans falling into each category. Here's how we can interpret and draw conclusions from the given scorecard:

- Very Poor (0 - 350): This category comprises 1.59% of the loans. These individuals have a high likelihood (92.63%) of defaulting on their loans. It's a high-risk segment that requires special attention.

- Poor (350 - 500): This category constitutes 8.59% of the loans. The default rate remains substantial at 66.8%. Borrowers falling into this range also exhibit a considerable risk of default.

- Fair (500 - 650): This is the largest category, representing 37.27% of loans. However, the default rate decreases to 26.83%. This group contains a diverse mix of borrowers with varying levels of creditworthiness.

- Good (650 - 800): About half (49.95%) of the loans fall into this category. The default rate drops significantly to 5.44%. Borrowers in this range are generally reliable with a relatively low risk of default.

- Very Good (800 - 1000): This category accounts for 2.6% of the loans. The default rate is extremely low at 0.65%. These borrowers showcase excellent credit profiles and are unlikely to default.''')
    ax, fig = credit_scorecard.plot_score(df)
    st.pyplot(fig)
    st.write('''The scorecard's divisions provide clear insights into the distribution of credit scores and their association with default rates. It's evident that as the credit score range improves, the risk of default decreases significantly. Borrowers with higher credit scores (Good and Very Good) have minimal default risk, indicating their creditworthiness. Conversely, those in the Poor and Very Poor categories pose higher default risks and might need additional scrutiny or tailored financial products. The scorecard facilitates efficient decision-making by categorizing borrowers based on their credit profiles, enabling lenders to manage risk effectively and offer appropriate loan terms to different segments.''')
    st.header('Kolmogorov-Smirnov Testing')
    st.write('''This is a test measuring the difference in distribution between GOOD and BAD based on various threshold ratios. If the model is capable of classifying GOOD and BAD effectively, the cumulative distribution function (CDF) curve between GOOD and BAD should exhibit significant separation. Conversely, if the model is weak and its predictions are only as good as random chance, the cumulative probability distribution curves of GOOD and BAD will closely align and approach the 45-degree diagonal line.

The Kolmogorov-Smirnov test assesses the hypothesis that the probability distributions of GOOD and BAD are not different. When the P-value is less than 0.05, the null hypothesis is rejected.''')
    cmd_BAD, cmd_GOOD, thresholds = credit_scorecard._KM(y_pred_prob_test, 20, y_test)
    st.write('**Kolmogorov-Smirnov Curve**')
    ax,fig = credit_scorecard._plot_KM(cmd_BAD, cmd_GOOD, thresholds)
    st.pyplot(fig)
    
    st.write('**P-value of Kolmogorov-Smirnov Test**')
    st.write('P-value:',stats.ks_2samp(cmd_BAD, cmd_GOOD)[1])
    st.write(f'''The P-value of 0.015 is less than the common significance level of 0.05. This suggests that we have enough evidence to reject the null hypothesis. In the context of the test, it indicates that there is a statistically significant difference in the probability distributions between GOOD and BAD. Therefore, we can conclude that the model's ability to classify GOOD and BAD is better than random chance, and the distributions indeed exhibit meaningful distinctions.''')
    
    st.header('Logistic Model')
    st.write('**Model Evaluation Result**')
    st.write('Accuracy score of train data is: ',acc_train)
    st.write('Accuracy score of test data is: ',acc_test)

    st.write('**ROC Curve of the model**')
    ax,fig  = credit_scorecard._plot_roc_curve(fpr, tpr, thres_roc, roc_auc)
    st.pyplot(fig)
    st.write('**Precision and Recall Curve of the model**')

    ax,fig  = credit_scorecard._plot_prec_rec_curve(precision, recall, thres_pre)
    st.pyplot(fig)

    st.write('')
    st.write('The model exhibits both a significant accuracy score and a commendable ROC score. Specifically, with accuracy scores of 0.843 for the training set and 0.838 for the test set, there is a relatively minor variance between them, indicating a potential risk of overfitting. Despite this concern, the model can still be employed effectively for calculating credit scores.')
    st.header('Information Value')
    df_WOE.sort_values(by='IV', inplace=True, ascending=False)
    st.dataframe(df_WOE,height=460)

    st.write('''The information value table rates variables' ability to predict credit outcomes. Here's what we find:

- High Impact: Variables like DEBTINC and LOAN have strong predictive power (IV = 1.0228). However, their similarity suggests potential overlap.

- Moderate Influence: DELINQ, DEROG, CLAGE, NINQ, and JOB show moderate predictiveness (IV ≈ 0.17 to 0.34).

- Limited Impact: Variables like YOJ, MORTDUE, and VALUE offer weaker predictiveness (IV < 0.05).

- Minimal Effect: REASON and CLNO show little influence on credit outcomes (IV < 0.01).

Prioritize high IV variables like DEBTINC and LOAN, and consider correlations between variables. These insights guide variable selection for robust credit scoring models.''')
elif selected_tab == "Calculate Your Credit Score":
    st.title('Calculate Your Credit Score')
    st.write('An interactive application designed to compute your credit score using our established model.\nFurther insights into the details of our model are available within this application.')
    st.header("Input Form")

    st.write('Fill out the following information to calculate your credit score.')
    if st.button('Clear'):
        st.experimental_rerun()
    # LOAN
    loan = st.number_input('LOAN: Enter the requested loan amount.', value=0, step=100)

    # MORTDUE
    mortdue = st.number_input('MORTDUE: Enter the amount remaining on your existing mortgage.', value=0, step=100)

    # VALUE
    value = st.number_input('VALUE: Enter the current value of your assets.', value=0, step=100)

    # REASON
    reason_options = ['DebtCon', 'HomeImp']
    reason = st.selectbox('REASON: Choose the reason for the loan request:', reason_options)

    # JOB
    job_options = ['Mgr', 'Office', 'Other', 'ProfExe', 'Sales', 'Self']
    job = st.selectbox('JOB: Select your current occupation:', job_options)

    # YOJ
    yoj = st.slider('YOJ: Enter the number of years of experience in your current profession.', min_value=0, max_value=70, value=0)

    # DEROG
    derog = st.number_input('DEROG: Enter the number of derogatory reports (negative credit reports).', value=0, step=1)

    # DELINQ
    delinq = st.number_input('DELINQ: Enter the number of delinquent credit accounts.', value=0, step=1)

    # CLAGE
    clage = st.number_input('CLAGE: Enter the age of your oldest credit account in months.', value=0, step=1)

    # NINQ
    ninq = st.number_input('NINQ: Enter the number of recent credit inquiries.', value=0, step=1)

    # CLNO
    clno = st.number_input('CLNO: Enter the total number of credit lines.', value=0, step=1)

    # DEBTINC
    debtinc = st.slider('DEBTINC: Enter the debt-to-income ratio.', min_value=0, max_value=100, value=0)


    # Button to submit the form
    submitted = st.button("Submit")

    if submitted:
        st.header('Scoring Result')
        try:
            obs = pd.DataFrame({'LOAN':[loan], 'MORTDUE':[mortdue], 'VALUE':[value], 'REASON':[reason], 'JOB':[job], 'YOJ':[yoj], 'DEROG':[derog], 'DELINQ':[delinq],
                                'CLAGE':[clage], 'NINQ':[ninq], 'CLNO':[clno], 'DEBTINC':[debtinc]})
            df_WOE = credit_scorecard.get_beta(X_train,alpha,coef_, columns, WOE_dict)
            score ,obs_score = credit_scorecard._total_score(obs, columns, WOE_dict, df_WOE)
            st.write('Your credit score is:', obs_score)

            if obs_score >= 0 and obs_score <= 350:
                conclusion = "Your credit score is {}, placing you in the 'Very Poor' segment according to our model. With a high default rate of 92.63%, improving your credit is essential before considering new credit applications.".format(obs_score)
            elif obs_score > 350 and obs_score <= 500:
                conclusion = "Your credit score is {}, falling in the 'Poor' segment. While the default rate is 66.8%, there's room for improvement. Focus on timely payments and reducing debts.".format(obs_score)
            elif obs_score > 500 and obs_score <= 650:
                conclusion = "Your credit score of {}, places you in the 'Fair' segment. While default risk is moderate at 26.83%, responsible credit behavior can further improve your score.".format(obs_score)
            elif obs_score > 650 and obs_score <= 800:
                conclusion = "Congratulations! Your credit score is {}, indicating the 'Good' segment. With a low default rate of 5.44%, you're in a strong position for credit opportunities.".format(obs_score)
            elif obs_score > 800:
                conclusion = "Excellent! Your credit score of {}, places you in the 'Very Good' segment. Your default rate is impressively low at 0.65%, leading to favorable credit terms.".format(obs_score)
            st.write(conclusion)
        except:
            st.error('Invalid input. Please enter again!')
