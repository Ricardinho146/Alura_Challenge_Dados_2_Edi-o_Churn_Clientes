#pip install -r requirements.txt

import pandas as pd
import numpy as np
from dython.nominal import associations
from dython.nominal import identify_nominal_columns
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import streamlit as st

df = pd.read_json('https://challenge-data-science-3ed.s3.amazonaws.com/Telco-Customer-Churn.json')
# Expandindo os dados da coluna "customer" e inserindo colunas no DataFrame
expanded_data1 = pd.json_normalize(df['customer'])

# Expandindo os dados da coluna "phone" e inserindo colunas no DataFrame
expanded_data2 = pd.json_normalize(df['phone'])

# Expandindo os dados da coluna "internet" e inserindo colunas no DataFrame
expanded_data3 = pd.json_normalize(df['internet'])

# Expandindo os dados da coluna "internet" e inserindo colunas no DataFrame
expanded_data4 = pd.json_normalize(df['account'])
df = pd.concat([df, expanded_data1, expanded_data2, expanded_data3, expanded_data4], axis=1)

# Excluir as colunas originais 'customer','phone','internet'e'account'
df.drop(columns=['customer','phone','internet','account'], inplace=True)

# Remover espaços em branco dos valores e substituir valores vazios por NaN condidos na coluna "Churn"
df['Churn'] = df['Churn'].str.replace(' ', '').replace('', pd.NA)

# Remover espaços em branco dos valores e substituir valores vazios por NaN
df['Charges.Total'] = df['Charges.Total'].str.replace(' ', '').replace('', pd.NA)

# Converter para float
df['Charges.Total'] = pd.to_numeric(df['Charges.Total'], errors='coerce')

# Removendo dados nulos
df = df.dropna(axis=0)

lb = LabelEncoder()
df['Churn']  = lb.fit_transform(df['Churn'])

# Removendo a coluna "customerID" do dataframe pois não faz sentido para análise

df_tst = df.drop("customerID", axis =1)

# Gerando uma lista com as colunas de dados categóricos
dados_categóricos =identify_nominal_columns(df_tst)


le = LabelEncoder()

for coluna in dados_categóricos:
    df_tst[coluna] = le.fit_transform(df_tst[coluna])

df_tst.head()

x_1 = df_tst.drop("Churn", axis =1)
y_1 = df_tst.Churn

ros = RandomOverSampler(sampling_strategy = 1)

x_ros_1, y_ros_1 = ros.fit_resample(x_1,y_1)

x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(x_ros_1, y_ros_1, test_size = 0.2, random_state = 42 )


modelo_tst = RandomForestClassifier( random_state = 42)

modelo_tst.fit(x_train_1, y_train_1)

y_pred_1 = modelo_tst.predict(x_test_1)


st.title('# My First Data App')
st.write('## Churn Predict Model with RandomForestClassifier')

gender = st.selectbox("Select Gender: ", ['Female','Male'])
SeniorCitizen = st.selectbox("Select SeniorCitizen: ", ['0','1'])
partner = st.selectbox("Select Partner: ", ['Yes','No'])
dependents = st.selectbox("Select Dependents: ", ['Yes','No'])
Tenure = st.slider("Select Tenure: ", 1, 72)
phoneService = st.selectbox("Select PhoneService: ", ['Yes','No'])
multipleLines = st.selectbox("Select MultipleLines: ", ['No','Yes','No phone service'])
internetservice = st.selectbox("Select InternetService: ", ['DSL','Fiber optic','No'])
onlinesecurity = st.selectbox("Select OnlineSecurity: ", ['No','Yes','No internet service'])
onlinebackup = st.selectbox("Select OnlineBackup: ", ['Yes','No','No internet service'])
deviceProtection = st.selectbox("Select DeviceProtection: ", ['No','Yes','No internet service'])
techSupport = st.selectbox("Select TechSupport: ", ['Yes','No','No internet service'])
streamingTV = st.selectbox("Select StreamingTV: ", ['Yes','No','No internet service'])
StreamingMovies = st.selectbox("Select StreamingMovies: ", ['No', 'Yes', 'No internet service'])
contract = st.selectbox("Select Contract: ", ['One year','Month-to-month','Two year'])
paperlessbilling = st.selectbox("Select PaperlessBilling: ", ['Yes','No'])
paymentMethod = st.selectbox("Select PaymentMethod: ", ['Mailed check','Electronic check','Credit card (automatic)','Bank transfer (automatic)'])
ChargesMonthly= st.slider("Select ChargesMonthly ", 18.25, 118.75)
ChargesTotal= st.slider("Select ChargesTotal",18.8, 8684.8)


def predict():
    input_data = {
        'gender': gender,
        'SeniorCitizen': SeniorCitizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': Tenure,
        'PhoneService': phoneService,
        'MultipleLines': multipleLines,
        'InternetService': internetservice,
        'OnlineSecurity': onlinesecurity,
        'OnlineBackup': onlinebackup,
        'DeviceProtection': deviceProtection,
        'TechSupport': techSupport,
        'StreamingTV': streamingTV,
        'StreamingMovies': StreamingMovies, 
        'Contract': contract,
        'PaperlessBilling': paperlessbilling,
        'PaymentMethod': paymentMethod,
        'Charges.Monthly': ChargesMonthly,
        'Charges.Total': ChargesTotal
    }

    # Criar um DataFrame a partir dos dados de entrada do usuário
    input_df = pd.DataFrame([input_data])

    # Mapear 'Yes' para 1 e 'No' para 0 nos dados booleanos
    boolean_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    input_df[boolean_cols] = input_df[boolean_cols].replace({'Yes': 1, 'No': 0})

    # Mapear 'Female' para 1 e 'Male' para 0 na coluna 'gender'
    input_df['gender'] = input_df['gender'].map({'Female': 1, 'Male': 0})

    # Criar um LabelEncoder separado para os outros dados categóricos
    le_input = LabelEncoder()

    # Converter outros dados categóricos usando LabelEncoder
    for w in dados_categóricos:
        if w not in boolean_cols + ['gender']:
            input_df[w] = le_input.fit_transform(input_df[w])

    # Mostrar os dados de entrada
    st.write('Input Data:')
    st.write(input_df)

    # Fazer a previsão
    prediction = modelo_tst.predict(input_df)

    # Calcular a acurácia usando os dados de teste
    accuracy = modelo_tst.score(x_test_1, y_test_1)

    st.write(f'Model Accuracy: {accuracy:.2%}')

    if prediction[0] == 1:
        st.success('Non-Quitting Customer :thumbsup:')
    else:
        st.error('Customer Abandonment :thumbsdown:')

trigger = st.button('Predict', on_click=predict)





