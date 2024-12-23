import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the trained model
with open('rf_selected.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize the app
st.title("To Grant or Not to Grant")
st.header("Tell us about the case and we will predict the outcome")

# Raw features
age = st.number_input("Age at Injury", min_value=0, max_value=100, step=1)
days_since_accident = st.number_input("Days Since Accident", min_value=0, max_value=10000, step=1)
adr = st.selectbox("Alternative Dispute Resolution", ["Yes", "No"])
attorney = st.selectbox("Attorney/Representative", ["Yes", "No"])
carrier_type = st.selectbox("Carrier Type", ["Self-Insured", "Group"])
covid_indicator = st.selectbox("COVID-19 Indicator", ["Yes", "No"])
district_name = st.selectbox("District Name", ['SYRACUSE', 'ROCHESTER', 'ALBANY', 'HAUPPAUGE', 'NYC', 'BUFFALO',
       'BINGHAMTON', 'STATEWIDE'])
medical_fee_region = st.selectbox("Medical Fee Region", ['I', 'II', 'III', 'IV', 'UK'])

# Age and Wage Groups
age_bins = [-np.inf, 30, 50, np.inf]  # Example thresholds
wage_bins = [-np.inf, 200, 400, np.inf]  # Example thresholds
age_group = pd.cut([age], bins=age_bins, labels=["Young", "Middle-aged", "Senior"])[0]
wage_group = pd.cut([200], bins=wage_bins, labels=["Low", "Medium", "High"])[0]  # Adjust wage input


# Created Features
cause_of_injury_mapping = {
    'Burn or Scald – Heat or ColdExposures– Contact With': [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 14, 84],
    'Caught In, Under or Between': [10, 12, 13, 20],
    'Cut, Puncture, Scrape Injured By': [15, 16, 17, 18, 19],
    'Fall, Slip or Trip Injury': [25, 26, 27, 28, 29, 30, 31, 32, 33],
    'Motor Vehicle': [40, 41, 45, 46, 47, 48, 50],
    'Strain or Injury By': [52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 97],
    'Striking Against or Stepping On': [65, 66, 67, 68, 69, 70],
    'Struck or Injured By': [74, 75, 76, 77, 78, 79, 80, 85, 86],
    'Rubbed or Abraded By': [94, 95],
    'Miscellaneous Causes': [82, 83, 87, 88, 89, 90, 91, 93, 96, 98, 99]
}
cause_injury_category = st.selectbox("Cause Injury Category", list(cause_of_injury_mapping.keys()))

nature_of_injury_mapping = {
    'Specific Injury': [1, 2, 3, 4, 5, 6, 7, 10, 13, 16, 19, 22, 25, 28, 30, 31, 32, 34, 36, 37, 38, 40, 41, 42, 43, 46, 47, 49, 52, 53, 54, 55, 58, 59],
    'Occupational Disease or Cumulative Injury': [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 83],
    'Multiple Injuries': [90, 91]
}
nature_injury_category = st.selectbox("Injury Nature Category", list(nature_of_injury_mapping.keys()))

body_part_mapping = {
    'Head': [10, 11, 12],
    'Ears': [13],
    'Eyes': [14, 15, 16, 17, 18, 19],
    'Neck': [20, 21, 22, 23, 24, 25, 26],
    'Upper Extremities': [30, 31, 32, 33, 34, 35],
    'Fingers': [36],
    'Thumb': [37],
    'Shoulders': [38],
    'Wrists and Hands': [39],
    'Trunk': [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 60, 61, 62, 63],
    'Lower Extremities': [50, 51, 52, 53, 54, 55, 56],
    'Toes and Great Toe': [57, 58],
    'Multiple Body Parts': [64, 65, 66, 90]
}
body_part_category = st.selectbox("Body Part Category", list(body_part_mapping.keys()))

industry_code_mapping = {
    'Education & Health': [61.0, 62.0],
    'Leisure & Hospitality': [71.0, 72.0],
    'Trade & Transportation': [42.0, 44.0, 45.0, 48.0],
    'Manufacturing & Construction': [23.0, 31.0, 32.0, 33.0],
    'Agriculture, Mining, Utilities': [11.0, 21.0, 22.0],
    'Information & Finance': [51.0, 52.0, 53.0],
    'Professional Services': [54.0, 55.0, 56.0],
    'Public & Other Services': [81.0, 92.0],
}
industry_category = st.selectbox("Industry Category", list(industry_code_mapping.keys()))

# IME-4 Category
ime4_count = st.number_input("IME-4 Count", min_value=0, max_value=100, step=1)
if ime4_count == 0:
    ime4_category = 'None'
elif 1 <= ime4_count <= 2:
    ime4_category = 'Few'
elif 3 <= ime4_count <= 5:
    ime4_category = 'Moderate'
elif 6 <= ime4_count <= 8:
    ime4_category = 'High'
else:
    ime4_category = 'Very High'

year_of_accident = st.selectbox(
    "Year of Accident",
    options=["Missing", 2018, 2019, 2020, 2021, 2022, 2023],
    format_func=lambda x: "Missing" if x == "Missing" else str(x)  # Display "Missing" properly
)

# Function to create Year Accident Grouping
def year_group(year):
    if year == "Missing":
        return "Missing"
    elif year < 2020:
        return "Pre-2020"
    else:
        return "Post-2020"

# Create Year Accident Grouping based on the selected year
year_accident_grouping = year_group(year_of_accident)


# Joined features
industry_code = st.text_input("Industry Code")
cause_description = st.selectbox(
    "WCIO Cause of Injury Description", 
    [
        'FROM LIQUID OR GREASE SPILLS', 'REPETITIVE MOTION', 
        'OBJECT BEING LIFTED OR HANDLED', 'HAND TOOL, UTENSIL; NOT POWERED',
        'FALL, SLIP OR TRIP, NOC', 'CUT, PUNCTURE, SCRAPE, NOC',
        'OTHER - MISCELLANEOUS, NOC', 'STRUCK OR INJURED, NOC',
        'FALLING OR FLYING OBJECT', 'CHEMICALS',
        'COLLISION OR SIDESWIPE WITH ANOTHER VEHICLE', 'LIFTING', 'TWISTING',
        'ON SAME LEVEL', 'STRAIN OR INJURY BY, NOC', 'MOTOR VEHICLE, NOC',
        'FROM DIFFERENT LEVEL (ELEVATION)', 'PUSHING OR PULLING',
        'FOREIGN MATTER (BODY) IN EYE(S)',
        'FELLOW WORKER, PATIENT OR OTHER PERSON', 'STEAM OR HOT FLUIDS',
        'STATIONARY OBJECT', 'ON ICE OR SNOW',
        'ABSORPTION, INGESTION OR INHALATION, NOC', 'PERSON IN ACT OF A CRIME',
        'INTO OPENINGS', 'ON STAIRS', 'FROM LADDER OR SCAFFOLDING',
        'SLIP, OR TRIP, DID NOT FALL', 'JUMPING OR LEAPING', 'MOTOR VEHICLE',
        'RUBBED OR ABRADED, NOC', 'REACHING', 'OBJECT HANDLED',
        'HOT OBJECTS OR SUBSTANCES', 'ELECTRICAL CURRENT', 'HOLDING OR CARRYING',
        'CAUGHT IN, UNDER OR BETWEEN, NOC', 'FIRE OR FLAME', 'CUMULATIVE, NOC',
        'POWERED HAND TOOL, APPLIANCE', 'STRIKING AGAINST OR STEPPING ON, NOC',
        'MACHINE OR MACHINERY', 'COLD OBJECTS OR SUBSTANCES', 'BROKEN GLASS',
        'COLLISION WITH A FIXED OBJECT', 'STEPPING ON SHARP OBJECT',
        'OBJECT HANDLED BY OTHERS', 'DUST, GASES, FUMES OR VAPORS',
        'OTHER THAN PHYSICAL CAUSE OF INJURY', 'CONTACT WITH, NOC',
        'USING TOOL OR MACHINERY', 'SANDING, SCRAPING, CLEANING OPERATION',
        'CONTINUAL NOISE', 'ANIMAL OR INSECT', 'MOVING PARTS OF MACHINE', 'GUNSHOT',
        'WIELDING OR THROWING', 'MOVING PART OF MACHINE', 'TEMPERATURE EXTREMES',
        'HAND TOOL OR MACHINE IN USE', 'VEHICLE UPSET',
        'COLLAPSING MATERIALS (SLIDES OF EARTH)', 'TERRORISM', 'PANDEMIC',
        'WELDING OPERATION', 'NATURAL DISASTERS', 'EXPLOSION OR FLARE BACK',
        'RADIATION', 'CRASH OF RAIL VEHICLE', 'MOLD', 'ABNORMAL AIR PRESSURE',
        'CRASH OF WATER VEHICLE', 'CRASH OF AIRPLANE'
    ]
)
nature_description = st.selectbox(
    "WCIO Nature of Injury Description", 
    [
        'CONTUSION', 'SPRAIN OR TEAR', 'CONCUSSION', 'PUNCTURE',
       'LACERATION', 'ALL OTHER OCCUPATIONAL DISEASE INJURY, NOC',
       'ALL OTHER SPECIFIC INJURIES, NOC', 'INFLAMMATION', 'BURN',
       'STRAIN OR TEAR', 'FRACTURE', 'FOREIGN BODY',
       'MULTIPLE PHYSICAL INJURIES ONLY', 'RUPTURE', 'DISLOCATION',
       'ALL OTHER CUMULATIVE INJURY, NOC', 'HERNIA', 'ANGINA PECTORIS',
       'CARPAL TUNNEL SYNDROME', 'NO PHYSICAL INJURY', 'INFECTION',
       'CRUSHING', 'SYNCOPE', 'POISONING - GENERAL (NOT OD OR CUMULATIVE',
       'RESPIRATORY DISORDERS', 'HEARING LOSS OR IMPAIRMENT',
       'MENTAL STRESS', 'SEVERANCE', 'ELECTRIC SHOCK', 'LOSS OF HEARING',
       'DUST DISEASE, NOC', 'DERMATITIS', 'ASPHYXIATION',
       'MENTAL DISORDER', 'CONTAGIOUS DISEASE', 'AMPUTATION',
       'MYOCARDIAL INFARCTION',
       'POISONING - CHEMICAL, (OTHER THAN METALS)',
       'MULTIPLE INJURIES INCLUDING BOTH PHYSICAL AND PSYCHOLOGICAL',
       'VISION LOSS', 'VASCULAR', 'COVID-19', 'CANCER',
       'HEAT PROSTRATION', 'AIDS', 'ENUCLEATION', 'ASBESTOSIS',
       'POISONING - METAL', 'VDT - RELATED DISEASES', 'FREEZING',
       'BLACK LUNG', 'SILICOSIS',
       'ADVERSE REACTION TO A VACCINATION OR INOCULATION', 'HEPATITIS C',
       'RADIATION', 'BYSSINOSIS'
       ]
)
body_part_description = st.selectbox("WCIO Part Of Body Description", ['BUTTOCKS', 'SHOULDER(S)', 'MULTIPLE HEAD INJURY', 'FINGER(S)',
       'LUNGS', 'EYE(S)', 'ANKLE', 'KNEE', 'THUMB', 'LOWER BACK AREA',
       'ABDOMEN INCLUDING GROIN', 'LOWER LEG', 'HIP', 'UPPER LEG',
       'MOUTH', 'WRIST', 'SPINAL CORD', 'HAND', 'SOFT TISSUE',
       'UPPER ARM', 'FOOT', 'ELBOW', 'MULTIPLE UPPER EXTREMITIES',
       'MULTIPLE BODY PARTS (INCLUDING BODY',
       'BODY SYSTEMS AND MULTIPLE BODY SYSTEMS', 'MULTIPLE NECK INJURY',
       'CHEST', 'WRIST (S) & HAND(S)', 'EAR(S)',
       'MULTIPLE LOWER EXTREMITIES', 'DISC', 'LOWER ARM', 'MULTIPLE',
       'UPPER BACK AREA', 'SKULL', 'TOES', 'FACIAL BONES', 'TEETH',
       'NO PHYSICAL INJURY', 'MULTIPLE TRUNK', 'WHOLE BODY',
       'INSUFFICIENT INFO TO PROPERLY IDENTIFY - UNCLASSIFIED', 'PELVIS',
       'NOSE', 'GREAT TOE', 'INTERNAL ORGANS', 'HEART', 'VERTEBRAE',
       'LUMBAR & OR SACRAL VERTEBRAE (VERTEBRA', 'BRAIN',
       'SACRUM AND COCCYX', 'ARTIFICIAL APPLIANCE', 'LARYNX', 'TRACHEA'])

industry_cause = f"{industry_code}_{cause_description}"
industry_nature = f"{industry_code}_{nature_description}"
industry_body_part = f"{industry_code}_{body_part_description}"
age_nature = f"{age_group}_{nature_description}"

# Prediction button
if st.button("Predict"):
    # Create input DataFrame with raw user inputs
    input_data = pd.DataFrame({
        'Age at Injury': [age],
        'Days_Since_Accident': [days_since_accident],
        'Alternative Dispute Resolution': [adr],
        'Attorney/Representative': [attorney],
        'Carrier Type': [carrier_type],
        'COVID-19 Indicator': [covid_indicator],
        'District Name': [district_name],
        'Medical Fee Region': [medical_fee_region],
        'Age Group': [age_group],
        'Wage Group': [wage_group],
        'Cause Injury Category': [cause_injury_category],
        'Injury Nature Category': [nature_injury_category],
        'Body Part Category': [body_part_category],
        'Industry Category': [industry_category],
        'IME-4 Category': [ime4_category],
        'Industry_Cause': [industry_cause],
        'Industry_Nature': [industry_nature],
        'Industry_BodyPart': [industry_body_part],
        'Age_Nature': [age_nature],
        'Year Accident Grouping': [year_accident_grouping],
    })

    # Encode categorical variables to match training data format
    encoding_map = {
        'Alternative Dispute Resolution': {"Yes": 1, "No": 0},
        'Attorney/Representative': {"Yes": 1, "No": 0},
        'Carrier Type': {"Self-Insured": 1, "Group": 0},
        'COVID-19 Indicator': {"Yes": 1, "No": 0},
    }

    for column, mapping in encoding_map.items():
        if column in input_data.columns:
            input_data[column] = input_data[column].map(mapping)

    # Ensure all remaining categorical columns are factorized (converted to numerical)
    for col in input_data.select_dtypes(include='object').columns:
        input_data[col], _ = pd.factorize(input_data[col])

    # Perform prediction
    try:
        prediction = model.predict(input_data)
        st.success(f"The predicted class is: {prediction[0]}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")