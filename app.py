import streamlit as st
import pickle
import numpy as np

# Set wider layout for desktop view
st.set_page_config(
    page_title="Titanic Survival Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Model Loading (Kept for functionality) ---
try:
    with open('titanic_model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('label_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)

    le_sex = encoders['Sex']
    le_embarked = encoders['Embarked']
except FileNotFoundError:
    st.error("Model or encoder files not found. Please ensure 'titanic_model.pkl' and 'label_encoders.pkl' are in the directory.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading files: {e}")
    st.stop()

# --- Title and Header ---
st.title("🚢 Titanic Survival Estimate")
st.markdown("## Enter Passenger Details")

# --- Input Form ---
with st.form(key='prediction_form'):
    st.subheader("👤 Passenger Information")

    # Use columns for desktop, but inputs will stack vertically on mobile
    col1, col2 = st.columns(2)

    with col1:
        # Pclass and Sex are placed in the first column
        pclass = st.selectbox(
            "🪪 Passenger Class",
            [1, 2, 3],
            format_func=lambda x: f"{x}st Class" if x == 1 else (f"{x}nd Class" if x == 2 else f"{x}rd Class"),
        )
        
        sex = st.radio(
            "🚻 Sex",
            ["male", "female"],
            horizontal=True
        )
        
        # Age is placed here
        age = st.slider(
            "📅 Age (Years)",
            min_value=0.42,
            max_value=80.0,
            value=30.0,
            step=0.1,
        )

    with col2:
        # Embarked and Family are placed in the second column
        embarked = st.selectbox(
            "⚓ Port of Embarkation",
            ["S (Southampton)", "C (Cherbourg)", "Q (Queenstown)"],
        )
        embarked_map = {"C (Cherbourg)": "C", "Q (Queenstown)": "Q", "S (Southampton)": "S"}
        embarked_code = embarked_map[embarked]

        st.markdown("##### Family Aboard")
        # Use inner columns to keep family inputs side-by-side on desktop
        col2a, col2b = st.columns(2)
        with col2a:
            sibsp = st.number_input(
                "👨‍👩‍ Siblings/Spouses",
                min_value=0,
                max_value=8,
                value=0,
            )
        with col2b:
            parch = st.number_input(
                "👶 Parents/Children",
                min_value=0,
                max_value=6,
                value=0,
            )

    st.markdown("---")
    submitted = st.form_submit_button("🔮 Predict Survival Outcome", type="primary", use_container_width=True)

# --- Prediction Logic ---
if submitted:
    try:
        sex_enc = le_sex.transform([sex])[0]
        embarked_enc = le_embarked.transform([embarked_code])[0]

        features = np.array([[pclass, sex_enc, age, sibsp, parch, embarked_enc]])

        prediction = model.predict(features)[0]

        st.markdown("## Prediction Result")

        if prediction == 1:
            st.balloons() 
            st.success("🎉 **LIKELY SURVIVED** 🎉", icon="✅")
            st.markdown(f"""
                <div style='background-color: #d4edda; color: #155724; padding: 15px; border-radius: 5px; border: 1px solid #c3e6cb;'>
                    The model estimates that, based on the input details, the passenger **would likely have survived** the sinking.
                </div>
            """, unsafe_allow_html=True)
        else:
            st.error("😔 **LIKELY DID NOT SURVIVE** 😔", icon="❌")
            st.markdown(f"""
                <div style='background-color: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px; border: 1px solid #f5c6cb;'>
                    The model estimates that, based on the input details, the passenger **would likely not have survived** the sinking.
                </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.exception(f"An error occurred during prediction: {e}")

# --- Sidebar ---
st.sidebar.header("Model Information")
st.sidebar.markdown("""
    This is a **machine learning classifier** (e.g., Logistic Regression, SVM, or similar) 
    trained on the historical Titanic dataset. 
    
    The prediction is based on the patterns the model learned from the input features:
    - **Pclass** (Ticket Class)
    - **Sex**
    - **Age**
    - **SibSp** (Siblings/Spouses Aboard)
    - **Parch** (Parents/Children Aboard)
    - **Embarked** (Port of Embarkation)

    *Disclaimer: This is a historical prediction model and is for educational/demonstration purposes only.*
""")