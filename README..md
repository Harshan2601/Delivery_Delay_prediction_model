# 🚚 Delivery Delay Prediction

A **Streamlit web application** that predicts whether a delivery is likely to be delayed. The system uses **real-time route information from OSRM** and a **Random Forest machine learning model** trained using historical delivery data.

---

## 🎯 Project Objective

Delivery delays can affect customer satisfaction and business operations. This project helps to:

- **Predict delivery delays** before or during delivery.
- Calculate the **real driving distance and estimated travel time** using OSRM.
- Search for previous orders using their **Order ID**.
- Compare the **actual delivery time with the estimated time**.
- Show model performance using **accuracy, confusion matrix, ROC curve, and feature importance**.
- Provide a simple interface that can be used by **delivery managers and operations teams** without writing code.

---

## 🏗️ Project Architecture

```text
                ┌──────────────────────┐
                │       data.csv       │
                │  Historical Orders   │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │     load_data()      │
                │ Clean and prepare    │
                │ the historical data  │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ enrich_with_routing()│
                │ Get route information│
                │ from OSRM            │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ train_delay_model()  │
                │ Random Forest Model  │
                │ Training & Testing   │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │    Streamlit App     │
                │                      │
                │ Order Search         │
                │ Delay Prediction     │
                │ Route Map            │
                │ Model Performance    │
                └──────────────────────┘

                    ▲
                    │
              ┌─────┴─────┐
              │    OSRM   │
              │  Routing  │
              │    API    │
              └───────────┘
```

---

## 🔧 Main Components

| Component                  | What it does                                                                                       |
| -------------------------- | -------------------------------------------------------------------------------------------------- |
| `load_data()`              | Loads and cleans the historical delivery data. Removes duplicate and incomplete records.           |
| `enrich_with_routing()`    | Gets real route distance and travel time from OSRM and adds this information to the training data. |
| `train_delay_model()`      | Trains the Random Forest model and calculates its performance.                                     |
| `get_real_time_distance()` | Gets the current route distance and estimated travel time from OSRM.                               |
| `plot_model_performance()` | Displays accuracy, confusion matrix, ROC curve, and feature importance.                            |
| `Streamlit UI`             | Provides the user interface for searching orders and predicting delivery delays.                   |

---

## 🔄 How the System Works

### 1. Data Loading

The application reads historical delivery information from `data.csv`.

It:

- Removes duplicate records.
- Removes rows with important missing information.
- Converts date and time values into the correct format.
- Handles invalid time values without crashing the application.

### 2. Route Information

For model training, the application sends the pickup and drop-off coordinates to the **OSRM Routing API**.

OSRM provides:

- Driving distance.
- Estimated driving time.

The results are saved in `data_with_routes.csv` so that the application does not need to call OSRM repeatedly.

### 3. Delay Detection

The system compares the **actual delivery time** with the **OSRM estimated travel time**.

An order is considered delayed when:

```text
Actual delivery time > Estimated time × 1.20
```

In other words, if the actual delivery takes more than **20% longer** than the estimated time, it is marked as delayed.

```text
0 → On Time
1 → Delayed
```

### 4. Model Training

The system uses a **Random Forest Classifier** to learn patterns from historical deliveries.

The model uses information such as:

- Pickup coordinates.
- Drop-off coordinates.
- Route distance.
- Estimated route duration.

The trained model is saved as:

```text
delay_prediction_model.pkl
```

### 5. Delay Prediction

When a user enters a new delivery or searches for an existing order:

1. Pickup and drop-off locations are obtained.
2. OSRM calculates the current route.
3. Route information is sent to the Random Forest model.
4. The model calculates the probability of a delay.
5. The result is displayed in the Streamlit application.

Example:

```text
Delay Probability: 78%

Prediction: HIGH RISK OF DELAY
```

### 6. Map Visualization

The application uses **Folium** to display:

- Pickup location.
- Drop-off location.
- Delivery route.

This makes it easier for users to understand the delivery visually.

### 7. Model Performance

The application also provides a performance dashboard containing:

- Accuracy
- Confusion Matrix
- ROC Curve
- AUC Score
- Feature Importance

This helps users understand how well the machine learning model is performing.

---

## 💼 Use Cases

### 🚚 Delivery Management

Operations teams can check whether a delivery is likely to be delayed and take action early.

### 🔍 Order Tracking

Users can search for an order using its **Order ID** and check its delivery performance.

### 📍 New Route Checking

Users can enter new pickup and drop-off coordinates to estimate the delay risk for a new delivery.

### 📊 Model Analysis

Managers and developers can view the model's performance through graphs and evaluation metrics.

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd delivery-delay-prediction
```

### 2. Install Required Libraries

```bash
 pandas numpy joblib requests folium streamlit streamlit-folium scikit-learn matplotlib
```

### 3. Add the Dataset

Place your `data.csv` file inside the project folder.

### 4. Run the Application

```bash
streamlit run app.py
```

The application will open in your web browser.

---

## 📄 Dataset Requirements

The `data.csv` file should contain the following columns:

| Column            | Description                        |
| ----------------- | ---------------------------------- |
| `Order_ID`        | Unique ID for each order           |
| `Store_Latitude`  | Pickup location latitude           |
| `Store_Longitude` | Pickup location longitude          |
| `Drop_Latitude`   | Delivery location latitude         |
| `Drop_Longitude`  | Delivery location longitude        |
| `Pickup_Time`     | Time when the order was picked up  |
| `Delivery_Time`   | Delivery time or delivery duration |

---

## 🧰 Technologies Used

- **Python** — Main programming language
- **Streamlit** — Web application interface
- **Pandas** — Data processing
- **NumPy** — Numerical calculations
- **Scikit-learn** — Machine learning
- **Random Forest** — Delay prediction algorithm
- **OSRM** — Route and travel-time calculation
- **Folium** — Interactive maps
- **Matplotlib** — Performance graphs
- **Joblib** — Saving and loading the trained model
- **Requests** — Connecting to the OSRM API

---

## ⚠️ Limitations

- The project uses the **public OSRM server**, which has usage limits.
- For a production application, it is better to use a **self-hosted OSRM server or a commercial routing service**.
- The 20% delay threshold is a simple rule and can be changed according to business requirements.
- The model's accuracy depends heavily on the quality and quantity of historical delivery data.
- Traffic, weather, road closures, and other real-world conditions may affect actual delivery times but are not directly included in the current model.
- The trained model and evaluation results are stored locally.

The following files are created by the application:

```text
delay_prediction_model.pkl
delay_model_metrics.pkl
data_with_routes.csv
```

If the historical dataset is changed, these files can be deleted to train the model again.

---

## 🚀 Project Summary

The **Delivery Delay Prediction System** combines **Machine Learning, real-time routing, and an interactive Streamlit interface** to predict delivery delays.

The system takes historical delivery data, gets route information from OSRM, trains a Random Forest model, and then uses the trained model to predict the probability of delays for new deliveries.

This project demonstrates how **Machine Learning and real-time location data can be combined to support smarter delivery planning and decision-making.**
