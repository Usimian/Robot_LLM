# Robot command message (flows through gateway only)
string robot_id                    # Target robot identifier
string command_type                 # "move", "rotate", "stop", "custom"
float64 linear_x                   # Forward/backward velocity (m/s)
float64 linear_y                   # Left/right velocity (m/s)  
float64 angular_z                  # Rotational velocity (rad/s)
float64 duration                   # Command duration (seconds)
string parameters_json             # Additional parameters as JSON
int64 timestamp_ns                 # Command timestamp
string source_node                 # Originating node name
