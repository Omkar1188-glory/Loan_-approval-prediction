import streamlit as st
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
import datetime
import base64
# Load the trained model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Inject custom CSS for styling
st.markdown(
    """
    <style>
    body {
        background-color: #E5E4E2;
    }
    .stButton>button {
        background-color: #4CAF50; /* Green */
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 8px;
    }
    
    </style>
    """, unsafe_allow_html=True
)
# Function to load image and encode it to base64
def load_image(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Load the image as a base64 string (change the path to your image)
image_base64 = load_image("bank-loan-successfully-illustration-concept-on-white-background-vector.jpg")
# Define a function to show the form page
def show_form():
    st.title('📊 Loan Prediction App By Omkar Kadam.')

    # Input form - all fields on the main page
    col1, col2 = st.columns(2)

    with col1:
        # Display the image using base64 encoding
        st.markdown(
            f'<img src="data:image/png;base64,{image_base64}" style="width:100%;" />',
            unsafe_allow_html=True
        )

    with col2:
        # Input form - all fields on the right side
        applicant_name = st.text_input("Applicant Name")
        dob = st.date_input(
        "Date of Birth",
        value=datetime.date(2000, 1, 1),  # Default date
        min_value=datetime.date(1900, 1, 1),  # Minimum date
        max_value=datetime.date.today()  # Maximum date set to today
    )
        married_status = st.selectbox("Married Status", ("Single", "Married"))
        gender = st.selectbox("Gender", ("Male", "Female", "Other"))
        city = st.text_input("City")
    # Model input fields
    no_of_dep = st.slider('Choose Number of Dependents', 0, 5)
    grad = st.selectbox('Choose Education', ['Graduated', 'Not Graduated'])
    self_emp = st.selectbox('Self Employed?', ['Yes', 'No'])
    Annual_Income = st.slider('Choose Annual Income', 0, 10000000)
    Loan_Amount = st.slider('Choose Loan Amount', 0, 10000000)
    Loan_Dur = st.slider('Choose Loan Duration (in years)', 0, 20)
    Cibil = st.slider('Choose Cibil Score', 0, 1000)
    Assets = st.slider('Choose Assets', 0, 10000000)

    grad = 1 if grad == 'Graduated' else 0
    self_emp = 1 if self_emp == 'Yes' else 0

    # Submit button
    if st.button("Submit"):
        # Model prediction (input fields for prediction)
        input_data = [no_of_dep, grad, self_emp, Annual_Income, Loan_Amount, Loan_Dur, Cibil, Assets]
        input_data_scaled = scaler.transform([input_data])

        # Calculate probability and prediction
        probability = model.predict_proba(input_data_scaled)[0][1]
        prediction = model.predict(input_data_scaled)[0]
        
        # Store user info and model result in session state
        st.session_state['applicant_info'] = {
            'Applicant Name': applicant_name,
            'DOB': dob,
            'Married Status': married_status,
            'Gender': gender,
            'City': city,
            'Probability': probability,
            'Prediction': prediction
        }

        # Set the state to view the result page
        st.session_state['page'] = 'result'

# Function to display the result page
def show_result():
    # Ensure 'applicant_info' exists in session state
    if 'applicant_info' not in st.session_state:
        st.error("No applicant information found. Please submit the form first.")
        return
    
    info = st.session_state['applicant_info']
    
    # Main header
    st.title('📊 Loan Prediction App')
    
    # Display applicant details
    st.subheader(f"Applicant: {info['Applicant Name']}")
    st.write(f"Date of Birth: {info['DOB']}")
    st.write(f"Married Status: {info['Married Status']}")
    st.write(f"Gender: {info['Gender']}")
    st.write(f"City: {info['City']}")
    
    # Columns for buttons
    col1, col2 = st.columns(2)

    # Button to see chance of approval
    if col1.button('🔍 See Chance of Approval'):
        st.session_state.prob = info['Probability'] * 100  # Store approval chance in session state
        st.markdown(f"<h3 style='color:blue;'>There is a {st.session_state.prob:.2f}% chance your loan will be approved</h3>", unsafe_allow_html=True)

    # Button to see final result
    if col2.button('📜 See Final Result'):
        result = 'approved' if info['Prediction'] == 1 else 'rejected'
        
        if result == 'approved':
            st.markdown(f"<h3 style='color:green;'>🎉 Congratulations! Your loan is {result}.</h3>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h3 style='color:red;'>😞 Unfortunately, your loan is {result}.</h3>", unsafe_allow_html=True)

    # Progress bar for approval chances
    if col1.button('Show Progress'):
        if 'prob' in st.session_state:
            st.progress(st.session_state.prob / 100)
        else:
            st.warning("Please check the chance of approval first.")
    
    # Footer with some extra info
    st.markdown("---")
    st.markdown("💡 *Tip: To improve your loan chances, ensure that your CIBIL score is above 750 and maintain a low debt-to-income ratio.*")

    # Back button to go back to the form page
    if st.button("Go Back"):
        st.session_state['page'] = 'form'


# Initialize the session state to show the form page initially
if 'page' not in st.session_state:
    st.session_state['page'] = 'form'

# Page navigation logic
if st.session_state['page'] == 'form':
    show_form()
elif st.session_state['page'] == 'result':
    show_result()
