# ROS2 message for robot commands
# Robot command message (flows through gateway only)

string robot_id                    # Target robot identifier
string command_type                # "move", "turn", "stop"
float64 linear_x                   # Forward/backward component (m/s)
float64 linear_y                   # Left/right strafe component (m/s)  
float64 angular_z                  # Rotational component (rad/s)
float64 duration                   # Command duration (seconds)
float64 distance                   # Linear distance to travel (meters)
float64 angular                    # Angular distance to turn (degrees)
float64 linear_speed               # Target linear speed (m/s)
float64 angular_speed              # Target angular speed (rad/s)
int64 timestamp_ns                 # Command timestamp
string source_node                 # Originating node name
