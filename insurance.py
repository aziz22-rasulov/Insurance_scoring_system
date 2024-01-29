import numpy as np
import pickle
import pandas as pd
#from flasgger import Swagger
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
    prediction=classifier.predict([[CLAIM_AMOUNT, MARITAL_STATUS, AGE, TENURE,
                                    NO_OF_FAMILY_MEMBERS, RISK_SEGMENTATION, HOUSE_TYPE,
                                    SOCIAL_CLASS, CUSTOMER_EDUCATION_LEVEL, CLAIM_STATUS,
                                    INCIDENT_SEVERITY, AUTHORITY_CONTACTED, ANY_INJURY,
                                    POLICE_REPORT_AVAILABLE, INCIDENT_HOUR_OF_THE_DAY, Health, Life,
                                Mobile, Motor, Property, Travel]])
    print(prediction)
    return prediction

def main():
    st.title("Insurance price prediction")
    html_temp = """
    <div style="background-color:blue    ;padding:10px">
    <h2 style="color:white;text-align:center;">Добро пожаловать в систему предсказаний стоимости страхового полиса  </h2>
    </div>
    """
    
    
    
    st.markdown(html_temp,unsafe_allow_html=True)
    CLAIM_AMOUNT = st.number_input('Сколько вы хотите получить от страховой компании в случае несчастного проишествия(используйте только цифры)?', step=1, value=0)
    MARITAL_STATUS = st.radio('У вас есть семья?(0 - Y , 1 - N)', (0, 1))
    AGE = st.number_input('Сколько вам лет(используйте только цифры)?', step=1, value=0)
    TENURE = st.number_input('На какой срок вы хотите страховой полис(введите количество дней)?', step=1, value=0)
    NO_OF_FAMILY_MEMBERS = st.number_input('Сколько у вас членов семьи(используйте только цифры)?', step=1, value=0)
    RISK_SEGMENTATION = st.radio('Риск предшествия несчастного случая(0 - L , 1 - M, 2 - H)', (0, 1, 2))
    HOUSE_TYPE = st.radio('Тип вашего дома(0 - Own , 1 - Rent, 2 - Mortgage)', (0, 1, 2))
    SOCIAL_CLASS = st.radio('Тип вашего дома(0 - LI , 1 - MI, 2 - HI)', (0, 1, 2))
    CUSTOMER_EDUCATION_LEVEL = st.radio('Ваш уровень образования(0 - MD , 1 - Masters, 2 - PhD, 3 - Bachelor, 4 - College, 5 - High School, 6 - 0)', (0, 1, 2, 3, 4, 5, 6))
    CLAIM_STATUS = st.radio('Статус обращения за страховым полисом?(0 - A , 1 - D)', (0, 1))
    INCIDENT_SEVERITY = st.radio('Оцениваемый ущерб инцидента(0 - Minor Loss, 1 - Major Loss, 2 - Total Loss)', (0, 1, 2))
    AUTHORITY_CONTACTED = st.radio('Кто связался с вами после проишествия(0 - None, 1 - Other, 2 - Ambulance, 3 - Police)', (0, 1, 2, 3))
    ANY_INJURY = st.number_input('Были ли у вас какие то повреждения(0 - да, 1 - нет)?', step=1, value=0)
    POLICE_REPORT_AVAILABLE = st.number_input('Имеется отчет полиции(0 - да, 1 - нет)?', step=1, value=0)
    INCIDENT_HOUR_OF_THE_DAY = st.number_input('Время проишествия(Введите только часы (формат 24 часовой)?', step=1, value=0)
    Health = st.radio('Имеется ли страховка здоровья?(0 - да , 1 - нет)', (0, 1))
    Life = st.radio('Имеется ли страховка жизни?(0 - да , 1 - нет)', (0, 1))
    Mobile = st.radio('Имеется ли страховка на мобильный телефон?(0 - да , 1 - нет)', (0, 1))
    Motor = st.radio('Имеется ли страховка на  машину?(0 - да , 1 - нет)', (0, 1))
    Property = st.radio('Имеется ли страховка на имущество?(0 - да , 1 - нет)', (0, 1))
    Travel = st.radio('Имеется ли страховка на время путешествия?(0 - да , 1 - нет)', (0, 1))
    
    result=""
    if st.button("Predict"):
        result=(predict_note_authentication((CLAIM_AMOUNT, MARITAL_STATUS, AGE, TENURE,
                                NO_OF_FAMILY_MEMBERS, RISK_SEGMENTATION, HOUSE_TYPE,
                                SOCIAL_CLASS, CUSTOMER_EDUCATION_LEVEL, CLAIM_STATUS,
                                INCIDENT_SEVERITY, AUTHORITY_CONTACTED, ANY_INJURY,
                                POLICE_REPORT_AVAILABLE, INCIDENT_HOUR_OF_THE_DAY, Health, Life,
                                Mobile, Motor, Property, Travel))
   
    st.success('Predicted cost of the house is {}'.format(result)+" TJS")
                     
    
    if st.button("About program"):
        
        st.text("Built by Aziz Rasulov")
              
                  
        
if __name__=='__main__':
    main()
    
    
