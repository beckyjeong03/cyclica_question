import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_train = pd.read_csv("af2_dataset_training_labeled.csv")
df_test = pd.read_csv("af2_dataset_testset_unlabeled.csv")
def clean_df(df: pd.DataFrame) -> None:
  df.drop(['Unnamed: 0','coord_X','coord_Y','coord_Z','annotation_atomrec','entry', 'annotation_sequence'], axis=1, inplace=True)
  # cols = ['feat_PHI', 'feat_PSI', 'feat_TAU', 'feat_THETA', 'feat_BBSASA', 'feat_SCSASA', 'feat_pLDDT',
  #                  'feat_DSSP_6', 'feat_DSSP_7', 'feat_DSSP_8', 'feat_DSSP_9', 'feat_DSSP_10', 'feat_DSSP_11',
  #                  'feat_DSSP_12', 'feat_DSSP_13']
  # for col in cols:
  #   if (df[col] == 0).all():
  #     df[col] = None
  #   df[col].fillna(df[col].median(), inplace=True)

test_ids = df_test['Unnamed: 0']

clean_df(df_train)
clean_df(df_test)
from sklearn.model_selection import train_test_split

X = df_train.drop('y_Ligand', axis=1)
y = df_train['y_Ligand']

from imblearn.under_sampling import RandomUnderSampler

undersampler = RandomUnderSampler()

X_under, y_under = undersampler.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_under, y_under, test_size=0.2)

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(max_iter=10000)
clf = clf.fit(X_train, y_train)
from sklearn.metrics import accuracy_score

preds = clf.predict(X_test)
accuracy_score(y_test, preds)
from sklearn.metrics import f1_score, recall_score, precision_score

f1 = f1_score(y_test, preds)
recall = recall_score(y_test, preds)
precision = precision_score(y_test, preds)
print('F1 score:', f1)
print('recall: ', recall)
print('precision: ', precision)
submission_preds = clf.predict(df_test)
result = pd.DataFrame({
    'id': test_ids,
    'predictions': submission_preds
})

result.to_csv('sdfd.csv', index=False)