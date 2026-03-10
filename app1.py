import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_csv("/content/Women_s_E-Commerce_Clothing_Reviews_1594_1.csv", sep=';')

df = df[['Age','Rating']]
df = df.dropna()

# -----------------------------
# Add Weather + Mood
# -----------------------------

np.random.seed(42)

df["temperature"] = np.random.randint(5,40,len(df))

moods = ["Lazy","Productive","Going Out"]
df["mood"] = np.random.choice(moods,len(df))

# -----------------------------
# Create Aesthetic Label
# -----------------------------

def assign_aesthetic(temp,mood):

    if temp < 15 and mood == "Lazy":
        return "Cozy"

    elif temp < 20 and mood == "Productive":
        return "Dark Academia"

    elif temp >= 25 and mood == "Going Out":
        return "Y2K"

    elif temp >= 20 and mood == "Productive":
        return "Soft Girl"

    else:
        return "Streetwear"

df["aesthetic"] = df.apply(
    lambda x: assign_aesthetic(x["temperature"],x["mood"]),
    axis=1
)

# -----------------------------
# Encode Data
# -----------------------------

mood_encoder = LabelEncoder()
aesthetic_encoder = LabelEncoder()

df["mood_encoded"] = mood_encoder.fit_transform(df["mood"])
df["aesthetic_encoded"] = aesthetic_encoder.fit_transform(df["aesthetic"])

# -----------------------------
# Train Model
# -----------------------------

X = df[["temperature","mood_encoded"]]
y = df["aesthetic_encoded"]

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)

# -----------------------------
# Model Accuracy
# -----------------------------

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)

# -----------------------------
# STREAMLIT INTERFACE
# -----------------------------

st.title("👗 Aesthetic Outfit Forecaster")

st.write("Predict outfit aesthetic based on temperature and mood")

st.write("Model Accuracy:",accuracy)

# Temperature Slider
temperature = st.slider(
    "Temperature (Cold → Hot)",
    min_value=5,
    max_value=40,
    value=25
)

# Mood Dropdown
mood = st.selectbox(
    "Select Mood",
    ["Lazy","Productive","Going Out"]
)

# Encode mood
mood_encoded = mood_encoder.transform([mood])[0]

# Predict button
if st.button("Predict Outfit Aesthetic"):

    input_data = np.array([[temperature,mood_encoded]])

    prediction = model.predict(input_data)

    result = aesthetic_encoder.inverse_transform(prediction)

    st.success("Recommended Aesthetic: " + result[0])