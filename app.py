from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Supply Chain AI Core", description="API for predicting late delivery and demand")

# Cors settings taaki frontend connects bina error ke ho sakein
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Models & Encoders
with open('classifier_model.pkl', 'rb') as f:
    clf_model = pickle.load(f)
with open('regressor_model.pkl', 'rb') as f:
    reg_model = pickle.load(f)
with open('label_encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

# Input Schema
class OrderInput(BaseModel):
    Type: str
    Days_for_shipment_scheduled: int
    Category_Name: str
    Customer_Segment: str
    Department_Name: str
    Market: str
    Order_Region: str
    Product_Price: float

@app.get("/")
def home():
    return {"status": "Online", "message": "Supply Chain Multi-Model API is Live!"}

@app.post("/predict")
def make_prediction(data: OrderInput):
    # 1. Map input to DataFrame columns exactly as trained
    raw_data = {
        'Type': data.Type,
        'Days for shipment (scheduled)': data.Days_for_shipment_scheduled,
        'Category Name': data.Category_Name,
        'Customer Segment': data.Customer_Segment,
        'Department Name': data.Department_Name,
        'Market': data.Market,
        'Order_Region': data.Order_Region, # Check if Column name matches your training data (e.g. 'Order Region')
        'Product Price': data.Product_Price
    }
    
    # Note: Agar training ke waqt space tha name me toh 'Order Region' likhein
    # data matching exact columns of X_train
    input_df = pd.DataFrame({
        'Type': [data.Type],
        'Days for shipment (scheduled)': [data.Days_for_shipment_scheduled],
        'Category Name': [data.Category_Name],
        'Customer Segment': [data.Customer_Segment],
        'Department Name': [data.Department_Name],
        'Market': [data.Market],
        'Order Region': [data.Order_Region],
        'Product Price': [data.Product_Price]
    })
    
    # 2. Transform text labels to numeric codes using label encoders
    processed_df = input_df.copy()
    for col in ['Type', 'Category Name', 'Customer Segment', 'Department Name', 'Market', 'Order Region']:
        processed_df[col] = encoders[col].transform(input_df[col])
        
    # 3. Running Multi-Model Inference
    risk_pred = int(clf_model.predict(processed_df)[0])
    risk_prob = float(clf_model.predict_proba(processed_df)[0][1])
    
    demand_pred = reg_model.predict(processed_df)[0]
    final_demand = int(np.ceil(demand_pred)) if demand_pred > 0 else 1
    
    # 4. JSON Output
    return {
        "late_delivery_risk": risk_pred,
        "delay_probability": risk_prob,
        "predicted_demand_units": final_demand
    }