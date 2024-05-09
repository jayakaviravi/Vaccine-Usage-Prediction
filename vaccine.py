import pandas as pd
import numpy as np
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

def streamlit_configuration():

    # page configuration
    st.set_page_config(page_title='Vaccine Prediction')

    # page header transparent color
    page_background_color = """
    <style>

    [data-testid="stHeader"] 
    {
    background: rgba(0,0,0,0);
    }

    </style>
    """
    st.markdown(page_background_color, unsafe_allow_html=True)

    st.markdown(f'<h1 style="text-align: center; color: violet">Vaccine Usage Analysis and Prediction</h1>'
                f'<hr style="border: 0; height: 3px; background-image: linear-gradient(to right, violet, indigo, blue, green, yellow, orange, red);">',
                unsafe_allow_html=True)

    
# custom style for submit button - color and width

def style_submit_button():

     st.markdown("""
                    <style>
                    div.stButton > button:first-child {
                                                        background-color: #367F89;
                                                        color: white;
                                                        width: 70%}
                    </style>
                """, unsafe_allow_html=True)

    

# custom style for prediction result text - color and position

def style_prediction():

   st.markdown(
            """
            <style>
            .center-text {
                text-align: center;
                color: green
            }
            </style>
            """,
            unsafe_allow_html=True
        )

class user_input:
  
    education= [ '12 Years','< 12 Years', 'College Graduate', 'Some College']

    education_dict= { '12 Years':0,'< 12 Years':1, 'College Graduate':2, 'Some College':3}

    race =['Black', 'Hispanic','Other or Multiple','White']

    race_dict={'Black':0, 'Hispanic':1,'Other or Multiple':2,'White':3}

    sex=['Female', 'Male']

    sex_dict={'Female':0, 'Male':1}

    income=['<= $75,000, Above Poverty', '> $75,000','Below Poverty']

    income_dict={'<= $75,000, Above Poverty':0, '> $75,000':1,'Below Poverty':2}

    marital=['Married','Not Married']

    marital_dict={'Married':0,'Not Married':1}

    housing = ['Own', 'Rent']

    housing_dict={'Own':0, 'Rent':1}

    employement= ['Employed','Not in Labor Force','Unemployed']

    employement_dict= {'Employed':0,'Not in Labor Force':1,'Unemployed':2}

    census=['MSA, Not Principle  City', 'MSA, Principle City','Non-MSA']

    census_dict={'MSA, Not Principle  City':0, 'MSA, Principle City':1,'Non-MSA':2}

class data_prediction:

    def classification():

        with st.form('classification_form'):

            col1, col2, col3 = st.columns([0.5,0.1,0.5])

            with col1:

                h1n1_worry = st.text_input(label='H1N1 worry (min:0.0,max:3.0)')

                h1n1_awarness=  st.text_input(label='H1N1 awarness (min:0.0,max:2.0)')

                antiviral_medication=  st.text_input(label='Antiviral medication')

                contact_avoidance= st.text_input(label='Contact avoidance')

                bought_facemask= st.text_input(label='Bought facemask')

                wash_hand= st.text_input(label='Wash_hand')

                avoid_large_gathering= st.text_input(label='Avoid large gathering')

                reduced_outside_home_cont = st.number_input(label='Reduced outside home_contact')

                avoid_touch_face= st.number_input(label='Avoid_touch_face')

                dr_recc_h1n1_vacc= st.text_input(label='Docter recom H1N1 vaccine')

                dr_recc_seasonal_vacc = st.text_input(label='Docter recom seasonal vaccine')

                chronic_medic_condition = st.number_input(label='Chronic medical condition (min:0.0,max:3.0')

                cont_child_under_6_months =st.number_input(label='Contact child under 6_months')

                is_health_worker = st.number_input(label='Health worker')

                has_health_insurance = st.number_input(label='Health insurance')

                h1n1_vacc_effect= st.text_input(label='H1N1 vaacine effective (min:0.0,max:5.0)')

            with col3:

                h1n1_risky = st.text_input(label='H1N1 risky')

                sick_h1n1_vacc = st.text_input(label='Sick from H1N1 vaccine')

                seasonal_vac_effec= st.text_input(label='Seasonal vaacine effective')

                seas_risky = st.text_input(label='seasonal risky')

                sick_seas_vacc=  st.text_input(label='sick from seasonal vaccine')

                age_bracket = st.text_input(label='Age')

                qualification = st.selectbox(label='Qualification',options=user_input.education)

                Race = st.selectbox(label='Race',options=user_input.race)

                Sex = st.selectbox(label='Gender',options=user_input.sex)

                Income_level = st.selectbox(label='Income',options=user_input.income)

                Marital = st.selectbox(label='Marital status',options=user_input.marital)

                housing = st.selectbox(label='Housing status',options=user_input.housing)

                Employement = st.selectbox(label='Employement status',options=user_input.employement)

                census_msa = st.selectbox(label='Census_MSA',options=user_input.census)

                no_of_adult = st.number_input(label='Number of adult  (min:0.0,max:3.0)')

                no_of_child = st.number_input(label='Number of children ')

                st.write('')
                st.write('')
                
                button = st.form_submit_button(label='SUBMIT')
                
                style_submit_button()

        # give information to users
        col1,col2 = st.columns([0.65,0.35])
        with col2:
            
            st.caption(body='*Min and Max values are reference only')

        # user entered the all input values and click the button
        if button:
            
            # load the classification pickle model
            
            with open('C:/Users/JAYAKAVI/New folder/classification_vaccine.pkl', 'rb') as f_2:
                vacc_model = pickle.load(f_2)

            # make array for all user input values in required order for model prediction

            user_data = np.array([[float(h1n1_worry),float(h1n1_awarness),float(antiviral_medication),float(contact_avoidance),float(bought_facemask),
                                   float(wash_hand),float(avoid_large_gathering),reduced_outside_home_cont,avoid_touch_face,float(dr_recc_h1n1_vacc),
                                   float(dr_recc_seasonal_vacc),chronic_medic_condition,cont_child_under_6_months,is_health_worker,
                                   has_health_insurance,float(h1n1_vacc_effect),float(h1n1_risky),float(sick_h1n1_vacc),float(seasonal_vac_effec),
                                   float(seas_risky),float(sick_seas_vacc),float(age_bracket),user_input.education_dict[qualification],
                                   user_input.race_dict[Race],user_input.sex_dict[Sex],user_input.income_dict[Income_level],
                                   user_input.marital_dict[Marital],user_input.housing_dict[housing],user_input.employement_dict[Employement],
                                   user_input.census_dict[census_msa],no_of_adult,no_of_child]])
            
            
            # model predict the probability based on user input
            prediction =vacc_model.predict_proba(user_data)[:, 1][0]

            return prediction

streamlit_configuration()

with st.sidebar:

    selected = option_menu(None,  ["Home", "Prediction", "Overview"],
                        default_index=0,
                        orientation="horizontal",
                        styles={"nav-link": {"font-size": "20px", "text-align": "centre", "margin": "-3px",
                                                "--hover-color": "#545454"},
                                "icon": {"font-size": "20px"},
                                "container": {"max-width": "3000px"},
                                "nav-link-selected": {"background-color": "violet"}})


if selected=='Home':

    st.subheader(':red[Introduction]')
    st.markdown('#####  Vaccination is a critical tool in public health, offering protection against infectious diseases and safeguarding communities worldwide. However, the successful deployment of vaccines relies not only on their availability but also on factors such as public acceptance, healthcare infrastructure, and governmental policies.')
    st.markdown('##### In this analysis, Predicting the probability of individuals taking an H1N1 flu vaccine based on their characteristics and attitudes.')

    st.subheader(':red[Tools used]')
    st.markdown('#####  Python scripting, Pandas,Data Visualisation and Machine Learning')

if selected=='Prediction':
    
    vaccine_prediction=data_prediction.classification()

    if vaccine_prediction:
        
        style_prediction()
        st.markdown(f'### <div class="left-text"><span style="color:green;">Probability of H1N1 Vaccine Acceptance = {round(vaccine_prediction, 2)}</span></div>', unsafe_allow_html=True)
        st.balloons()

if selected=='Overview':

    st.subheader(':orange[Objectives]')
    st.markdown('##### The main goal of this project is to predict the likelihood of individuals taking an H1N1 flu vaccine based on their characteristics and attitudes. By analyzing a dataset containing various features related to individuals behaviors, perceptions, and demographics, the project aims to build a predictive model using logistic regression.')

    st.subheader(':orange[Approach]')
    
    st.markdown("""
        1. :blue[Data Exploration and Preprocessing:]
                identifying  any missing values or inconsistencies.Preprocessing steps such as handling missing values, encoding categorical variables, and scaling numerical features are performed to prepare the data for modeling.
        2. :blue[Model Development:]
                I used four machine learning models to make the predictions. I used Logistic Regression,K nearest neighbour,Random forest and Decision Tree. Logistic regression is chosen as the predictive model due to its simplicity and interpretability. The model is trained on the preprocessed dataset to predict the probability of individuals accepting the H1N1 flu vaccine based on their characteristics and attitudes.
        3. :blue[Model Evaluation and Tuning:]
                The trained model is evaluated using appropriate performance metrics such as accuracy, precision, recall, and ROC-AUC score. Hyperparameter tuning techniques like grid search is applied to optimize the model's performance.""")
    
    button_1 = st.button("EXIT!")
    
    if button_1:
        st.success("**Thank you for utilizing this platform. I hope you have received the probability of individuals taking an H1N1 flu vaccine!**")

