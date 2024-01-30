import numpy as np
import pickle
import pandas as pd
import streamlit as st

from PIL import Image

pickle_in = open("insurance_catboost.pkl","rb")
classifier=pickle.load(pickle_in)

def welcome():
    return "Добро Пожаловать"

def predict_note_authentication(CLAIM_AMOUNT, MARITAL_STATUS, AGE, TENURE,
                                NO_OF_FAMILY_MEMBERS, RISK_SEGMENTATION, HOUSE_TYPE,
                                SOCIAL_CLASS, CUSTOMER_EDUCATION_LEVEL, CLAIM_STATUS,
                                INCIDENT_SEVERITY, AUTHORITY_CONTACTED, ANY_INJURY,
                                POLICE_REPORT_AVAILABLE, INCIDENT_HOUR_OF_THE_DAY, Health, Life,
                                Mobile, Motor, Property, Travel):
    prediction = classifier.predict([[CLAIM_AMOUNT, MARITAL_STATUS, AGE, TENURE,
                                    NO_OF_FAMILY_MEMBERS, RISK_SEGMENTATION, HOUSE_TYPE,
                                    SOCIAL_CLASS, CUSTOMER_EDUCATION_LEVEL, CLAIM_STATUS,
                                    INCIDENT_SEVERITY, AUTHORITY_CONTACTED, ANY_INJURY,
                                    POLICE_REPORT_AVAILABLE, INCIDENT_HOUR_OF_THE_DAY, Health, Life,
                                    Mobile, Motor, Property, Travel]])
    print(prediction)
    return prediction[0]

def main():
    st.title("Insurance price prediction")
    html_temp = """
    <div style="background-color:yellow;padding:10px">
    <h2 style="color:black;text-align:center;">Добро пожаловать в систему предсказаний стоимости страхового полиса  </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    
    html_temp = """
    <div style="background-color:yellow;padding:10px">
    <h2 style="color:black;text-align:center;">Вопросы для клиента  </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    Health = st.radio('Хотите ли вы получить страховку здоровья?(0 - да , 1 - нет)', (0, 1))
    Life = st.radio('Хотите ли вы получить страховку жизни?(0 - да , 1 - нет)', (0, 1))
    Mobile = st.radio('Хотите ли вы получить страховку на мобильный телефон?(0 - да , 1 - нет)', (0, 1))
    Motor = st.radio('Хотите ли вы получить страховку на  машину?(0 - да , 1 - нет)', (0, 1))
    Property = st.radio('Хотите ли вы получить страховку на имущество?(0 - да , 1 - нет)', (0, 1))
    Travel = st.radio('Хотите ли вы получить страховку на время путешествия?(0 - да , 1 - нет)', (0, 1))
    CLAIM_AMOUNT = st.number_input('Сколько вы хотите получить от страховой компании в случае несчастного проишествия(используйте только цифры)?', step=1, value=0)
    MARITAL_STATUS = st.radio('Вы в браке?(0 - да, 1 - нет)', (0, 1))
    AGE = st.number_input('Сколько вам лет(используйте только цифры)?', step=1, value=0)
    TENURE = st.number_input('На какой срок вы хотите страховой полис(введите количество дней)?', step=1, value=0)
    NO_OF_FAMILY_MEMBERS = st.number_input('Сколько у вас членов семьи(без вас)(используйте только цифры)?', step=1, value=0)
    HOUSE_TYPE = st.radio('Тип вашего дома(0 - Собственный , 1 - Арендуемый, 2 - Ипотечный)', (0, 1, 2))
    CUSTOMER_EDUCATION_LEVEL = st.radio('Ваш уровень образования (0 - докторантура, 1 - магмстратура, 2 - доктор философии, 3 - бакалавриат, 4 - колледж, 5 - старшая школа, 6 - 0)', (0, 1, 2, 3, 4, 5, 6))
    
    html_temp = """
    <div style="background-color:yellow;padding:10px">
    <h2 style="color:black;text-align:center;">Вопросы для клиента со страховой историей  </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    
    
    INCIDENT_SEVERITY = st.radio('Оцениваемый ущерб инцидента, если у вас есть страховая история с проишествием (0 - незначительный, 1 - средний, 2 - критический)', (0, 1, 2))
    AUTHORITY_CONTACTED = st.radio('Кто связался с вами после происшествия, если у вас есть страховая история с проишествием (0 - никто, 1 - другое, 2 - скорая, 3 - полиция)', (0, 1, 2, 3))
    ANY_INJURY = st.number_input('Были ли у вас какие-то повреждения, если у вас есть страховая история с проишествием (0 - да, 1 - нет)?', step=1, value=0)
    POLICE_REPORT_AVAILABLE = st.number_input('Имеется отчет полиции, если у вас есть страховая история с проишествием (0 - да, 1 - нет)?', step=1, value=0)
    INCIDENT_HOUR_OF_THE_DAY = st.number_input('Время происшествия, если у вас есть страховая история с проишествием (Введите только часы, формат 24-часовой)?', step=1, value=0)
    
    html_temp = """
    <div style="background-color:yellow;padding:10px">
    <h2 style="color:black;text-align:center;">Вопросы для сотрудника страхового агента  </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    RISK_SEGMENTATION = st.radio('Риск предшествия несчастного случая вашего клиента (0 - низкий, 1 - средний, 2 - высокий)', (0, 1, 2))
    SOCIAL_CLASS = st.radio('К какому социальному классу относится ваш клиент (0 - нижний, 1 - средний, 2 - высший)', (0, 1, 2))
    CLAIM_STATUS = st.radio('Статус обращения за страховым полисом вашего клиента? (0 - A, 1 - D)', (0, 1))
    
    result = ""
    if st.button("Predict"):
        result = predict_note_authentication(CLAIM_AMOUNT, MARITAL_STATUS, AGE, TENURE,
                                NO_OF_FAMILY_MEMBERS, RISK_SEGMENTATION, HOUSE_TYPE,
                                SOCIAL_CLASS, CUSTOMER_EDUCATION_LEVEL, CLAIM_STATUS,
                                INCIDENT_SEVERITY, AUTHORITY_CONTACTED, ANY_INJURY,
                                POLICE_REPORT_AVAILABLE, INCIDENT_HOUR_OF_THE_DAY, Health, Life,
                                Mobile, Motor, Property, Travel)
        st.success('Predicted cost of the house is {} USD'.format(result))
    
    if st.button("About program"):
        st.text("Built by Aziz Rasulov")
              
if __name__=='__main__':
    main()

                
                
                
                
   