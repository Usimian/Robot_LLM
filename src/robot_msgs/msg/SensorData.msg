# ROS2 message for robot sensor data
# Replaces HTTP-based sensor communication

string robot_id
float64 battery_voltage
float64 battery_percentage
float64 temperature
float64 humidity
float64 distance_front
float64 distance_left
float64 distance_right
int32 wifi_signal
float64 cpu_usage
float64 memory_usage
geometry_msgs/Vector3 imu_values
string camera_status
int64 timestamp_ns
