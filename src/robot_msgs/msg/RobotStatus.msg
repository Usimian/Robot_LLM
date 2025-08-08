# ROS2 message for robot status
# Replaces HTTP-based status communication

string robot_id
string name
int64 last_seen_ns
geometry_msgs/Pose position
float64 battery_level
string status  # active, idle, error, offline
string[] capabilities
string connection_type
SensorData sensor_data
string last_command
string[] command_history
