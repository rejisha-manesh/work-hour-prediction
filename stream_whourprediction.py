from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np
model = load_model('workPrediction')

def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions

def run():
    from PIL import Image
    image = Image.open('employee.jpg')
    image_head = Image.open('employee_head.jpg')
    st.image(image,use_column_width=True)
    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))
    st.sidebar.info('This app is created to predict work hour prediction')
    st.sidebar.success('https://www.pycaret.org')
    st.sidebar.image(image_head)
    st.title("Predicting Work Hour")
    if add_selectbox == 'Online':
        
        age=st.number_input('Age' , min_value=1, max_value=85, value=1)
        work_class = st.selectbox('Work Class', ['Federal-gov', 'Loal-gov', 'Never-Worked', 'Private','Self-emp-inc','Self-emp-not-inc','State-gov','Without-pay'])
        education = st.selectbox('Education', ['10th', '11th', '12th', '1st-4th','5th-6th','7th-8th','9th','Assoc-acdm','Assoc-voc','Bachelors','Doctorate','HS-grad','Masters','Preschool','Prof-school','Some-college'])
        education_num = st.number_input('Education Number' , min_value=1, max_value=16, value=1)
        marital_status = st.selectbox('Marital Status', ['Never-married', 'Married-civ-spouse', 'Married-AF-spouse', 'Married-spouse-absent','Divorced','Separated','Widowed'])
        occupation = st.selectbox('Occupation', ['Adm-clerical','Armed-forces','Machine-op-inspct', 'Farming-fishing', 'Exec-managerial','Tech-support','Prof-speciality','Sales','Craft-repair','Handlers-cleaners','Other-service','Priv-house-serve','Protective-serv','Transport-moving'])
        gender = st.selectbox('Gender', ['Male', 'Female'])
        income = st.selectbox('Income', ['<=50K', '>50K']) 
        capital_gain=st.number_input('Capital Gain')
        capital_loss=st.number_input('Capital Loss')
        native_country = st.selectbox('Native Country', ['Cambodia',
                                                         'Canada',
                                                         'China',
                                                         'Columbia',
                                                         'Cuba',
                                                         'Dominican-Republic',
                                                         'Ecuador',
                                                         'El-Salvador',
                                                         'England',
                                                         'France',
                                                         'Germany',
                                                         'Greece',
                                                         'Guatemala',
                                                         'Haiti',
                                                         'Holand-Netherlands',
                                                         'Honduras',
                                                         'Hong',
                                                         'Hungary',
                                                         'India',
                                                         'Iran',
                                                         'Ireland',
                                                         'Italy',
                                                         'Jamaica',
                                                         'Japan',
                                                         'Laos',
                                                         'Mexico',
                                                         'Nicaragua',
                                                         'Outlying-US(Guam-USVI-etc)',
                                                         'Peru',
                                                         'Philippines',
                                                         'Poland',
                                                         'Portugal',
                                                         'Puerto-Rico',
                                                         'Scotland',
                                                         'South',
                                                         'Taiwan',
                                                         'Thailand',
                                                         'Trinadad&Tobago',
                                                         ' United-States',
                                                         'Vietnam',
                                                         'Yugoslavia'])

                        
        output=""
        input_dict={'age':age,'workclass':work_class,'education':education,'education-num':education_num,'marital-status': marital_status,'occupation':occupation,'sex' : gender,'capital-gain' : capital_gain,'capital-loss' : capital_loss,'income' : income,'native-country':native_country}
        input_df = pd.DataFrame([input_dict])
        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)
            output = str(output)
        st.success('The output is {}'.format(output))
    if add_selectbox == 'Batch':
        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
        if file_upload is not None:
            data = pd.read_csv(file_upload)            
            predictions = predict_model(estimator=model,data=data)
            st.write(predictions)

if __name__ == "__main__":
  run()
