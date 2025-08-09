# Sensor data from robot
string robot_id                    # Robot identifier
float64 battery_voltage            # Battery voltage (V)
float64 temperature                # Jetson CPU Temperature (°C)
float64 distance_front             # Front distance sensor (m)
float64 distance_left              # Left distance sensor (m)
float64 distance_right             # Right distance sensor (m)
float64 cpu_usage                  # CPU usage (%)
string camera_status               # Camera status
geometry_msgs/Vector3 imu_values   # IMU readings (x, y, z)
int64 timestamp_ns                 # Sensor reading timestamp
