# ROS2 message for robot commands
# Replaces HTTP-based command communication

string robot_id
string command_type
string parameters_json  # JSON-encoded parameters dict
int64 timestamp_ns     # Nanosecond timestamp
int32 priority
bool safety_confirmed
bool gui_movement_enabled
string source          # Source of the command (GUI, API, etc.)
