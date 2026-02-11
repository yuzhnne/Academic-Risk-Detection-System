import streamlit as st
import joblib
from pathlib import Path

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Early Student At-Risk Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------
# Hide "Press Enter to submit form" (tooltip + helper text)
# -------------------------
st.markdown("""
<style>
/* Hide Streamlit's form helper text (small text under widgets in forms) */
div[data-testid="stForm"] small { display: none !important; }

/* Hide tooltip overlays (the popup you see) */
div[role="tooltip"] { display: none !important; visibility: hidden !important; opacity: 0 !important; }

/* Extra safety for newer Streamlit builds */
[data-testid="stTooltipContent"] { display: none !important; }
[data-testid="stWidgetTooltip"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# -------------------------
# Resolve artifact paths (YOUR structure: student/artifacts/*.joblib)
# -------------------------
APP_DIR = Path(__file__).resolve().parent          # .../student
ARTIFACTS_DIR = APP_DIR / "artifacts"              # .../student/artifacts

MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
DEFAULT_ROW_PATH = ARTIFACTS_DIR / "default_row.joblib"

# -------------------------
# Load model + default row
# -------------------------
@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not DEFAULT_ROW_PATH.exists():
        raise FileNotFoundError(f"Default row file not found: {DEFAULT_ROW_PATH}")

    model_ = joblib.load(MODEL_PATH)
    default_row_ = joblib.load(DEFAULT_ROW_PATH)  # schema row
    return model_, default_row_

model, default_row = load_artifacts()

# -------------------------
# Helpers
# -------------------------
def set_if_exists(row, col, value):
    if col in row.columns:
        row.loc[row.index[0], col] = value

def risk_band(p):
    if p is None:
        return "Unknown"
    if p >= 0.70:
        return "High Risk"
    if p >= 0.40:
        return "Moderate Risk"
    return "Low Risk"

# -------------------------
# Sidebar Navigation
# -------------------------
st.sidebar.title("Early Student At-Risk Detection")
page = st.sidebar.radio(
    "Navigation",
    ["Prediction", "How It Works", "About"],
    index=0
)

st.sidebar.markdown(
    """
    **Usage Steps**
    - Enter student details
    - Submit for prediction
    - Review probability and recommendation
    """
)

# -------------------------
# Pages
# -------------------------
if page == "Prediction":
    st.title("Early Student At-Risk Detection")
    st.caption(
        "This application predicts whether a student is at risk of failing, "
        "allowing educators to provide early intervention."
    )

    st.divider()

    left, right = st.columns([1.05, 0.95], gap="large")

    # -------------------------
    # Student Form
    # -------------------------
    with left:
        st.subheader("Student Information")

        with st.form("student_form", border=True):
            col1, col2, col3 = st.columns(3)

            with col1:
                age = st.number_input(
                    "Age",
                    min_value=10,
                    max_value=25,
                    value=None,
                    placeholder="Enter age",
                )
                subject = st.selectbox(
                    "Subject",
                    ["Math", "Portuguese"],
                    index=None,
                    placeholder="Select subject",
                )

            with col2:
                failures = st.selectbox(
                    "Past Failures",
                    [0, 1, 2, 3],
                    index=None,
                    placeholder="Select number of failures",
                )
                absences = st.number_input(
                    "Absences",
                    min_value=0,
                    max_value=100,
                    value=None,
                    placeholder="Enter absences",
                )

            with col3:
                studytime = st.selectbox(
                    "Weekly Study Time (1=Low, 4=High)",
                    [1, 2, 3, 4],
                    index=None,
                    placeholder="Select study time",
                )

            st.markdown("Support and Environment")
            col4, col5, col6, col7 = st.columns(4)

            with col4:
                schoolsup = st.selectbox(
                    "Extra School Support",
                    ["yes", "no"],
                    index=None,
                    placeholder="Select",
                )
            with col5:
                famsup = st.selectbox(
                    "Family Educational Support",
                    ["yes", "no"],
                    index=None,
                    placeholder="Select",
                )
            with col6:
                internet = st.selectbox(
                    "Internet Access at Home",
                    ["yes", "no"],
                    index=None,
                    placeholder="Select",
                )
            with col7:
                higher = st.selectbox(
                    "Plans for Higher Education",
                    ["yes", "no"],
                    index=None,
                    placeholder="Select",
                )

            st.markdown("Academic Performance (Earlier Terms)")
            st.caption(
                "G1 and G2 are earlier-term grades used for early risk detection. "
                "Final grade G3 is not used as an input."
            )

            col8, col9 = st.columns(2)
            with col8:
                g1 = st.number_input(
                    "Grade 1 (G1)",
                    min_value=0,
                    max_value=20,
                    value=None,
                    placeholder="Enter G1 (0–20)",
                )
            with col9:
                g2 = st.number_input(
                    "Grade 2 (G2)",
                    min_value=0,
                    max_value=20,
                    value=None,
                    placeholder="Enter G2 (0–20)",
                )

            submitted = st.form_submit_button("Predict Risk", use_container_width=True)

    # -------------------------
    # Prediction Output
    # -------------------------
    with right:
        st.subheader("Prediction Output")

        if not submitted:
            st.info("Submit the form to view the prediction.")
        else:
            required_fields = [
                age, subject, failures, absences, studytime,
                schoolsup, famsup, internet, higher, g1, g2
            ]

            if any(v is None for v in required_fields):
                st.warning("Please fill in all fields before predicting.")
                st.stop()

            with st.spinner("Generating prediction..."):
                row = default_row.copy()

                set_if_exists(row, "age", age)
                set_if_exists(row, "subject", subject)
                set_if_exists(row, "failures", failures)
                set_if_exists(row, "absences", absences)
                set_if_exists(row, "studytime", studytime)
                set_if_exists(row, "schoolsup", schoolsup)
                set_if_exists(row, "famsup", famsup)
                set_if_exists(row, "internet", internet)
                set_if_exists(row, "higher", higher)
                set_if_exists(row, "G1", g1)
                set_if_exists(row, "G2", g2)

                pred = model.predict(row)[0]
                proba = (
                    float(model.predict_proba(row)[0][1])
                    if hasattr(model, "predict_proba")
                    else None
                )

            if pred == 1:
                st.error("Prediction Result: At Risk")
            else:
                st.success("Prediction Result: Not At Risk")

            if proba is not None:
                st.metric("Probability of Being At Risk", f"{proba:.2%}")
                st.progress(min(max(proba, 0.0), 1.0))
                st.write(f"Risk Level: {risk_band(proba)}")

            st.markdown("Recommended Action")
            if pred == 1:
                st.write(
                    "- Provide early academic support\n"
                    "- Monitor attendance and engagement\n"
                    "- Consider counselling or remedial assistance"
                )
            else:
                st.write(
                    "- Continue normal academic monitoring\n"
                    "- Encourage consistent study habits"
                )

            with st.expander("View Full Model Input"):
                st.dataframe(row, use_container_width=True)

elif page == "How It Works":
    st.title("How It Works")
    st.write(
        "The application collects student information, formats it into the required "
        "feature set, and applies a trained machine learning model to estimate the "
        "likelihood of academic risk."
    )
    st.write(
        "Final grade (G3) is excluded to avoid data leakage. Earlier-term grades (G1, G2) "
        "are used as they are available before final outcomes."
    )

elif page == "About":
    st.title("About This Application")
    st.write(
        "This application demonstrates an early student at-risk detection system "
        "designed for educational support staff. It serves as a decision-support tool "
        "and should be used alongside professional judgement."
    )
