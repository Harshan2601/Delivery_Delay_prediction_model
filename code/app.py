import pandas as pd
import numpy as np
import joblib
import requests
import folium
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_folium import folium_static
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
)

# OSRM API for route calculation
OSRM_URL = "http://router.project-osrm.org/route/v1/driving/{},{};{},{}?overview=full"


def get_real_time_distance(start_lat, start_lon, end_lat, end_lon):
    """Fetch real-time distance and duration from OSRM"""
    url = OSRM_URL.format(start_lon, start_lat, end_lon, end_lat)
    response = requests.get(url).json()
    if 'routes' in response and response['routes']:
        route = response['routes'][0]
        return route['distance'] / 1000, route['duration'] / 60  # Convert meters to km, seconds to minutes
    return None, None


def load_data():
    """Load and preprocess the dataset"""
    df = pd.read_csv("C:\\Program Files\\Delivary Delay Prediction\\code\\data.csv")
    df.rename(columns={
        'Order_ID': 'order_id',
        'Store_Latitude': 'start_lat',
        'Store_Longitude': 'start_lon',
        'Drop_Latitude': 'end_lat',
        'Drop_Longitude': 'end_lon',
        'Pickup_Time': 'pickup_time',
        'Delivery_Time': 'delivery_time'
    }, inplace=True)

    # --- Clean the raw data: remove duplicates and rows with missing key values ---
    initial_rows = len(df)

    # Drop exact duplicate rows
    df = df.drop_duplicates()

    # Drop duplicate orders (keep the first occurrence of each order_id), if present
    if 'order_id' in df.columns:
        df = df.drop_duplicates(subset='order_id', keep='first')

    # Drop rows missing required routing/location fields.
    # Keep rows with missing or malformed pickup/delivery times so existing order
    # lookups still work; invalid-timestamp rows are filtered only during model training.
    required_columns = ['order_id', 'start_lat', 'start_lon', 'end_lat', 'end_lon']
    df = df.dropna(subset=required_columns)

    # Strip stray whitespace from order_id (common in CSVs) so lookups match reliably
    if 'order_id' in df.columns:
        df['order_id'] = df['order_id'].astype(str).str.strip()

    cleaned_rows = len(df)
    print(f"Data cleaning: removed {initial_rows - cleaned_rows} duplicate/null rows "
          f"({initial_rows} -> {cleaned_rows})")
   

    # Parse pickup_time (HH:MM:SS format)
    df['pickup_time'] = df['pickup_time'].astype(str)
    df['pickup_time'] = pd.to_datetime(
        '1970-01-01 ' + df['pickup_time'],
        format='%Y-%m-%d %H:%M:%S',
        errors='coerce',
    )

    # Handle delivery_time: it's stored as delivery duration in minutes (numeric), not a timestamp
    df['delivery_duration'] = pd.to_numeric(df['delivery_time'], errors='coerce')


def enrich_with_routing(df, cache_path="data_with_routes.csv"):
    """
    Add real_distance_km / real_duration_min (via OSRM) and the is_delayed label.
    Only needed for model training, not for simple order lookups - so it's kept
    separate from load_data() and its result is cached to disk since it's slow
    (one HTTP request per row).
    """
    try:
        cached = pd.read_csv(cache_path)
        # Cache is only valid if it has the same rows as the current cleaned data
        if set(cached['order_id']) == set(df['order_id']):
            print("Using cached route-enriched data.")
            return cached
    except Exception:
        pass

    df = df.copy()
    df = df[df['delivery_duration'].notna()]
    df = df[(df['delivery_duration'] > 1) & (df['delivery_duration'] < 300)]

    df[['real_distance_km', 'real_duration_min']] = df.apply(
        lambda row: get_real_time_distance(
            row['start_lat'], row['start_lon'], row['end_lat'], row['end_lon']
        ),
        axis=1, result_type="expand"
    )

    # Drop rows where the routing API failed to return a distance/duration
    before = len(df)
    df = df.dropna(subset=['real_distance_km', 'real_duration_min'])
    dropped = before - len(df)
    if dropped:
        print(f"Routing enrichment: {dropped} rows dropped because OSRM could not "
              f"return a route for them (network issue, rate limit, or bad coordinates).")

    # Define delay as actual delivery duration exceeding the predicted duration by 20%
    df['is_delayed'] = (df['delivery_duration'] > df['real_duration_min'] * 1.2).astype(int)

    df.to_csv(cache_path, index=False)
    return df


FEATURE_COLUMNS = ['start_lat', 'start_lon', 'end_lat', 'end_lon', 'real_distance_km', 'real_duration_min']


def train_delay_model(df):
    """Train a RandomForest model to predict delays, and persist evaluation metrics for plotting"""
    X = df[FEATURE_COLUMNS]
    y = df['is_delayed']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # model.classes_ tells us which column of predict_proba corresponds to class "1" (delayed).
    # If the training data only contains one class, that column may not exist at index 1,
    # so we look it up safely instead of assuming index 1.
    proba = model.predict_proba(X_test)
    if 1 in model.classes_:
        positive_idx = list(model.classes_).index(1)
        y_proba = proba[:, positive_idx]
    else:
        # Model never saw a "delayed" example during training - probability is always 0
        y_proba = np.zeros(len(X_test))

    acc = accuracy_score(y_test, y_pred)
    print(f"Delay Prediction Model Trained. Accuracy: {acc:.2f}")

    # Save metrics needed to draw accuracy/performance graphs later, without needing to retrain
    metrics = {
        "accuracy": acc,
        "y_test": y_test.to_numpy(),
        "y_pred": y_pred,
        "y_proba": y_proba,
        "feature_names": FEATURE_COLUMNS,
        "feature_importances": model.feature_importances_,
    }
    joblib.dump(metrics, "delay_model_metrics.pkl")

    joblib.dump(model, "delay_prediction_model.pkl")
    return model


def plot_model_performance(metrics):
    """Render accuracy/performance graphs (confusion matrix, ROC curve, feature importance) in Streamlit"""
    st.subheader("Model Performance")
    st.write(f"**Test Accuracy:** {metrics['accuracy'] * 100:.2f}%")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Confusion Matrix**")
        fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
        cm = confusion_matrix(metrics["y_test"], metrics["y_pred"])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["On Time", "Delayed"])
        disp.plot(ax=ax_cm, cmap="Blues", colorbar=False)
        st.pyplot(fig_cm)

    with col2:
        st.markdown("**ROC Curve**")
        fpr, tpr, _ = roc_curve(metrics["y_test"], metrics["y_proba"])
        auc_score = roc_auc_score(metrics["y_test"], metrics["y_proba"])
        fig_roc, ax_roc = plt.subplots(figsize=(4, 4))
        ax_roc.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
        ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.legend(loc="lower right")
        st.pyplot(fig_roc)

    st.markdown("**Feature Importance**")
    fig_fi, ax_fi = plt.subplots(figsize=(6, 3))
    importances = pd.Series(metrics["feature_importances"], index=metrics["feature_names"]).sort_values()
    importances.plot(kind="barh", ax=ax_fi, color="teal")
    ax_fi.set_xlabel("Importance")
    st.pyplot(fig_fi)


# Load or Train Model
try:
    delay_model = joblib.load("delay_prediction_model.pkl")
    model_metrics = joblib.load("delay_model_metrics.pkl")
except Exception:
    df = load_data()
    df_enriched = enrich_with_routing(df)
    delay_model = train_delay_model(df_enriched)
    model_metrics = joblib.load("delay_model_metrics.pkl")

# Streamlit UI
st.title("Delivery Delay Prediction")

with st.expander("📊 Model Performance & Accuracy Graphs", expanded=False):
    plot_model_performance(model_metrics)

df = load_data()

order_id = st.text_input("Order ID").strip()

if order_id:
    existing_order = df[df['order_id'] == order_id]
    if not existing_order.empty:
        st.write("Existing Order Found:")
        start_lat = existing_order.iloc[0]['start_lat']
        start_lon = existing_order.iloc[0]['start_lon']
        end_lat = existing_order.iloc[0]['end_lat']
        end_lon = existing_order.iloc[0]['end_lon']
        actual_delivery_time = existing_order.iloc[0]['delivery_duration']
        if pd.isna(actual_delivery_time):
            st.warning("This order has missing or invalid delivery time in the dataset. Please enter it manually.")
            manual_delivery_time = st.number_input("Enter Actual Delivery Time (minutes)", min_value=1.0)
            actual_delivery_time = manual_delivery_time
    else:
        st.warning("Order ID not found. Please enter details for a new order.")
        start_lat = st.number_input("Pickup Latitude")
        start_lon = st.number_input("Pickup Longitude")
        end_lat = st.number_input("Drop Latitude")
        end_lon = st.number_input("Drop Longitude")
        manual_delivery_time = st.number_input("Enter Actual Delivery Time (minutes)", min_value=1.0)
        actual_delivery_time = manual_delivery_time
else:
    start_lat = st.number_input("Pickup Latitude")
    start_lon = st.number_input("Pickup Longitude")
    end_lat = st.number_input("Drop Latitude")
    end_lon = st.number_input("Drop Longitude")
    manual_delivery_time = st.number_input("Enter Actual Delivery Time (minutes)", min_value=1.0)
    actual_delivery_time = manual_delivery_time

if st.button("Predict Delay Probability"):
    real_distance, real_duration = get_real_time_distance(start_lat, start_lon, end_lat, end_lon)
    if real_distance is None or real_duration is None:
        st.error("Unable to fetch real-time distance and duration. Please check the coordinates.")
    else:
        X_new = np.array([[start_lat, start_lon, end_lat, end_lon, real_distance, real_duration]])
        delay_prob = delay_model.predict_proba(X_new)[0][1]  # Probability of being delayed
        estimated_delivery_time = real_duration * 1.2 if delay_prob > 0.5 else real_duration
        delay_status = "Delayed" if actual_delivery_time > estimated_delivery_time else "On Time"

        # Show results
        st.write(f"**Order ID:** {order_id}")
        st.write(f"**Probability of Delay:** {round(delay_prob * 100, 2)}%")
        st.write(f"**Real-time Distance:** {real_distance} km")
        st.write(f"**Estimated Delivery Time:** {estimated_delivery_time:.2f} minutes")
        st.write(f"**Actual Delivery Time:** {actual_delivery_time:.2f} minutes")
        st.write(f"**Delivery Status:** {delay_status}")

        # Show Map
        map_object = folium.Map(location=[start_lat, start_lon], zoom_start=12)
        folium.Marker([start_lat, start_lon], popup="Pickup",
                      icon=folium.Icon(color="blue")).add_to(map_object)
        folium.Marker([end_lat, end_lon], popup="Delivery",
                      icon=folium.Icon(color="red")).add_to(map_object)
        folium_static(map_object)
