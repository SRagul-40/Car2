import streamlit as st
import pickle
import numpy as np

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AutoMiles | Mileage Predictor",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------------------------------------------
# 2. CUSTOM CSS (The "Very Design" Part)
# -----------------------------------------------------------------------------
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    }
    
    /* Headings */
    h1, h2, h3 {
        color: white;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Input Area Styling */
    .stNumberInput > label {
        color: #e0e0e0 !important;
        font-size: 1.2rem;
        font-weight: bold;
    }
    
    /* Result Card Styling */
    .result-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 30px;
        margin-top: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        text-align: center;
        animation: fadeIn 1s;
    }
    
    /* Button Styling */
    .stButton > button {
        width: 100%;
        background-color: #ff4b1f;
        background-image: linear-gradient(to right, #ff4b1f 0%, #ff9068 51%, #ff4b1f 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 15px 30px;
        text-transform: uppercase;
        font-weight: bold;
        letter-spacing: 1px;
        transition: 0.5s;
        background-size: 200% auto;
        box-shadow: 0 0 20px #eee;
    }

    .stButton > button:hover {
        background-position: right center; /* change the direction of the change here */
        color: #fff;
        text-decoration: none;
        transform: scale(1.02);
    }
    
    /* Custom Classes for Text */
    .highlight-text {
        font-size: 4rem;
        font-weight: 800;
        background: -webkit-linear-gradient(#ffdf40, #ff8359);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .unit-text {
        font-size: 1.5rem;
        color: #ffffff;
    }
    
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. LOAD MODEL
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        with open('car_mileage_lr_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("⚠️ Model file not found. Please ensure 'car_mileage_lr_model.pkl' is in the same directory.")
        return None

model = load_model()

# -----------------------------------------------------------------------------
# 4. HEADER SECTION
# -----------------------------------------------------------------------------
col1, col2 = st.columns([1, 4])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/3202/3202926.png", width=100)
with col2:
    st.title("AutoMiles Predictor")
    st.markdown("### 🚀 AI-Powered Fuel Efficiency Estimation")

st.markdown("---")

# -----------------------------------------------------------------------------
# 5. INPUT SECTION
# -----------------------------------------------------------------------------
st.write("")
col_input, col_space, col_info = st.columns([1, 0.2, 1])

with col_input:
    st.markdown("#### ⚖️ Vehicle Details")
    # Weight in dataset is typically in 1000 lbs (e.g., 2.5 = 2500 lbs)
    weight_input = st.number_input(
        "Enter Weight (1000 lbs)",
        min_value=0.5,
        max_value=10.0,
        value=3.0,
        step=0.1,
        help="Enter the weight of the car. Example: 3.2 means 3,200 lbs."
    )

with col_info:
    st.info(
        """
        **Did you know?**
        
        Heavier cars generally consume more fuel because the engine works harder 
        to overcome inertia and rolling resistance. This model uses Linear Regression 
        to calculate the impact of weight on MPG.
        """
    )

# -----------------------------------------------------------------------------
# 6. PREDICTION LOGIC & DISPLAY
# -----------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)

if st.button("Calculate MPG"):
    if model:
        # Prepare input (2D array as expected by sklearn)
        input_data = np.array([[weight_input]])
        
        # Predict
        try:
            prediction = model.predict(input_data)[0]
            
            # Display Result in Custom Card
            st.markdown(f"""
                <div class="result-card">
                    <h3 style="margin-bottom: 0;">Estimated Fuel Efficiency</h3>
                    <div class="highlight-text">{prediction:.1f}</div>
                    <div class="unit-text">Miles Per Gallon (MPG)</div>
                    <br>
                    <p style="color: white; opacity: 0.8;">Based on a vehicle weight of {weight_input*1000:.0f} lbs</p>
                </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
    else:
        st.warning("Model could not be loaded.")

# -----------------------------------------------------------------------------
# 7. FOOTER
# -----------------------------------------------------------------------------
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align: center; color: rgba(255,255,255,0.5); font-size: 0.8rem;">
        Powered by Scikit-Learn & Streamlit | Linear Regression Model
    </div>
    """, 
    unsafe_allow_html=True
)
