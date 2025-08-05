# Robot Control GUI System

A comprehensive tkinter-based graphical user interface for controlling and monitoring robots using the VILA vision model integration.

## Overview

This system consists of three main components:

1. **`robot_vila_server.py`** - The backend server that handles robot communication and VILA model integration
2. **`robot_gui.py`** - The main tkinter GUI application for robot control and monitoring  
3. **`robot_launcher.py`** - A convenient launcher utility to start/stop both components

## Features

### Server Status & Communication
- Real-time connection status monitoring
- WebSocket support for live updates
- HTTP REST API integration
- Server health monitoring

### Robot Management
- View all connected robots
- Register new robots
- Monitor robot status, battery levels, and positions
- Real-time robot information updates

### Robot Control
- Movement controls (Forward, Backward, Left, Right, Stop)
- Image analysis using VILA model
- Custom prompt support for vision analysis
- Command sending and status feedback

### Monitoring (Placeholder Areas)
- **SLAM Map Display** - Reserved space for future SLAM visualization
- **Sensor Readings** - Placeholder for live sensor data (Lidar, IMU, GPS, Camera, Battery, Motors)
- **Real-time Updates** - Framework ready for live data streaming

### Settings & Configuration
- Server connection settings (host, ports)
- Application logging
- Configuration persistence

## Installation

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify VILA Model Setup:**
   Make sure the VILA model is properly configured in `main_vila.py`

## Usage

### Option 1: Use the Launcher (Recommended)
```bash
python robot_launcher.py
```

The launcher provides:
- Easy start/stop of both server and GUI
- System status monitoring
- Integrated logging
- Process management

### Option 2: Manual Start

1. **Start the Server:**
   ```bash
   python robot_vila_server.py
   ```

2. **Start the GUI (in another terminal):**
   ```bash
   python robot_gui.py
   ```

## GUI Interface

### Tabs Overview

1. **Server Status**
   - Connection status indicator
   - Server health information
   - Connection management buttons

2. **Robots**
   - List of all connected robots
   - Robot details and status
   - Robot registration

3. **Robot Control**
   - Robot selection for control
   - Movement control buttons
   - Image analysis interface
   - VILA vision integration

4. **Monitoring**
   - SLAM map placeholder (ready for integration)
   - Sensor value displays (ready for live data)
   - Real-time status updates

5. **Settings**
   - Server configuration
   - Application logs
   - System preferences

## Server API Integration

The GUI communicates with the robot server using:

### HTTP REST Endpoints:
- `GET /health` - Server health check
- `GET /robots` - List all robots  
- `POST /robots/register` - Register new robot
- `GET /robots/{id}/status` - Get robot status
- `POST /robots/{id}/commands` - Send commands
- `POST /robots/{id}/analyze` - Analyze images

### WebSocket Events:
- `connect` / `disconnect` - Connection management
- `robot_registered` - New robot notifications
- `robot_analysis` - Live analysis results
- `join_monitors` - Subscribe to updates

## Configuration

### Server Settings
- **Host:** Server hostname/IP (default: localhost)
- **HTTP Port:** REST API port (default: 5000)  
- **TCP Port:** Direct robot communication (default: 9999)

### Single Robot System
The GUI is designed for a single robot system:
- Robots automatically register themselves when they connect to the server
- The GUI automatically selects and controls the connected robot
- Robot status and battery levels are monitored in real-time

## Architecture

```
┌─────────────────┐    HTTP/WS     ┌──────────────────┐
│   Robot GUI     │◄──────────────►│  VILA Server     │
│   (tkinter)     │                │  (Flask/SocketIO)│
└─────────────────┘                └──────────────────┘
                                            │
                                            │ VILA Model
                                            ▼
                                   ┌──────────────────┐
                                   │   main_vila.py   │
                                   │  (Vision Model)  │
                                   └──────────────────┘
                                            │
                                            │ TCP/HTTP
                                            ▼
                                   ┌──────────────────┐
                                   │     Robots       │
                                   │  (Physical/Sim)  │
                                   └──────────────────┘
```

## Future Enhancements

The GUI is designed with expansion in mind:

### Planned Features:
- **SLAM Integration:** Real-time map display and navigation planning
- **Live Sensor Data:** Integration with robot sensor streams
- **Advanced Controls:** Waypoint navigation, autonomous modes
- **Multi-Robot Support:** Coordinated multi-robot operations
- **Data Recording:** Session recording and playback
- **Custom Dashboards:** User-configurable monitoring layouts

### Integration Points:
- SLAM map rendering area is ready for map data
- Sensor display framework supports live data feeds
- Command system supports complex autonomous behaviors
- WebSocket infrastructure enables real-time updates

## Troubleshooting

### Common Issues:

1. **Connection Failed**
   - Verify server is running: `python robot_vila_server.py`
   - Check host/port settings in GUI Settings tab
   - Ensure firewall allows connections

2. **VILA Model Not Loading**
   - Check VILA model installation
   - Verify GPU availability if using CUDA
   - Check server logs for model loading errors

3. **Robot Not Appearing**
   - Verify robot registration via API
   - Check robot network connectivity
   - Review server logs for registration issues

### Debug Mode:
Enable detailed logging by checking the application log in the Settings tab.

## Development

### Adding New Features:
1. **Server Side:** Add endpoints to `robot_vila_server.py`
2. **GUI Side:** Add interface elements to relevant tabs
3. **Communication:** Use existing HTTP/WebSocket infrastructure

### Extending Monitoring:
1. Add new sensor types to the monitoring tab
2. Update WebSocket handlers for new data types
3. Implement real-time visualization components

## License

This project follows the same licensing as the VILA model and associated components.