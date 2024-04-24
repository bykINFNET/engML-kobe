import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics

COL_TARGET = 'shot_made_flag'      # 'shot_made_flag' {'success': 1, 'fail': 0}


prod_file = '../data/processed/prediction_prod.parquet'
dev_file = '../data/processed/prediction_test.parquet'

# Carregamento de dados
df_prod = pd.read_parquet(prod_file)

df_dev = pd.read_parquet(dev_file)


fignum = plt.figure(figsize=(6,4))

# Saida do modelo dados dev
sns.distplot(df_dev.prediction_score_1,
             ax = plt.gca(),
             label='Teste')

# Saida do modelo dados prod
sns.distplot(df_prod.predict_score,
             ax = plt.gca(),
             label='Producao')

# User wine
plt.title('Resposta do Modelo')
plt.ylabel('Densidade Estimada')
plt.xlabel('Probabilidade Shot Made Flag')
plt.xlim((0,1))
plt.grid(True)
plt.legend(loc='best')

st.pyplot(fignum)

st.dataframe(metrics.classification_report(df_dev[COL_TARGET], df_dev.prediction_label, output_dict = True))
