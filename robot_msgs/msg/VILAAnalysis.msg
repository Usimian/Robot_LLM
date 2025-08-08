# ROS2 message for VILA analysis results
# Replaces HTTP-based VILA communication

string robot_id
string prompt
sensor_msgs/Image image
string analysis_result
string navigation_commands_json  # JSON-encoded navigation commands
float64 confidence
int64 timestamp_ns
bool success
string error_message
