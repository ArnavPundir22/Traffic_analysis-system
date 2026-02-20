import datetime


def traffic_status(vehicle_count):
    """
    Returns traffic condition based on number of vehicles.
    """

    if vehicle_count < 10:
        return "FREE ROAD"
    elif vehicle_count < 25:
        return "MODERATE TRAFFIC"
    else:
        return "HEAVY TRAFFIC"


def save_traffic_log(vehicle_count, filename="traffic_log.csv"):
    """
    Saves vehicle count and traffic status with timestamp.
    """

    status = traffic_status(vehicle_count)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(filename, "a") as f:
        f.write(f"{timestamp},{vehicle_count},{status}\n")

    print("Traffic log saved.")


# simple testing
if __name__ == "__main__":

    print("Traffic Prediction Test")

    try:
        count = int(input("Enter vehicle count: "))
        print("Traffic Condition:", traffic_status(count))
        save_traffic_log(count)
    except ValueError:
        print("Please enter a valid number.")
