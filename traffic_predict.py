# traffic_predict.py
# -------------------------
# Traffic Prediction Module
# -------------------------

import datetime


# 🚦 Function to Predict Traffic Condition
def traffic_status(vehicle_count):
    """
    Predict traffic level based on vehicle count.

    Parameters:
        vehicle_count (int): Total vehicles detected/counted

    Returns:
        str: Traffic condition message
    """

    if vehicle_count < 10:
        return "FREE ROAD 🟢"
    
    elif vehicle_count < 25:
        return "MODERATE TRAFFIC 🟡"
    
    else:
        return "HEAVY TRAFFIC 🔴"


# 📝 Function to Save Traffic Logs (Optional)
def save_traffic_log(vehicle_count, filename="traffic_log.csv"):
    """
    Save traffic data into a CSV file with timestamp.

    Parameters:
        vehicle_count (int): Number of vehicles
        filename (str): CSV log file name
    """

    status = traffic_status(vehicle_count)

    # Current date and time
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

    # Write to CSV
    with open(filename, "a") as file:
        file.write(f"{timestamp},{vehicle_count},{status}\n")

    print("✅ Traffic log saved!")


# 🔥 Main Test (Run this file separately)
if __name__ == "__main__":
    print("\n🚗 Traffic Prediction System\n")

    count = int(input("Enter Vehicle Count: "))

    result = traffic_status(count)
    print("\n📍 Traffic Condition:", result)

    # Save log
    save_traffic_log(count)
