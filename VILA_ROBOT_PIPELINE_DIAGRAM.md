# VILA → Robot Movement Pipeline Diagram

This diagram shows exactly how the VILA vision-language model controls robot movement through a 5-step process.

## Mermaid Diagram

```mermaid
graph TD
    A["📷 Camera Image"] --> B["🤖 VILA Model"]
    B --> C["📝 Text Response<br/>'I can see a clear path forward...'"]
    C --> D["🔍 Command Parser"]
    
    D --> E1["✓ 'advance', 'proceed', 'forward'<br/>→ move_forward = True"]
    D --> E2["✗ No 'stop' or hazard words<br/>→ stop = False"]  
    D --> E3["✗ No 'left'<br/>→ turn_left = False"]
    D --> E4["✗ No 'right'<br/>→ turn_right = False"]
    D --> E5["✓ No hazard keywords<br/>→ hazard_detected = False"]
    
    E1 --> F["⚙️ Command Generator"]
    E2 --> F
    E3 --> F
    E4 --> F
    E5 --> F
    
    F --> G["🤖 Robot Command<br/>{<br/>  'command_type': 'move',<br/>  'direction': 'forward',<br/>  'speed': 0.3,<br/>  'duration': 2.0<br/>}"]
    
    G --> H{"🛡️ Safety Check<br/>Movement Enabled?"}
    H -->|Yes| I["✅ Execute Command<br/>Robot moves forward"]
    H -->|No| J["🚫 Block Command<br/>Robot stays still"]
    
    K["🎛️ VILA Auto Nav Mode"] -.-> F
    L["🔧 Movement Toggle"] -.-> H
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style C fill:#fff3e0
    style D fill:#e8f5e8
    style F fill:#fff8e1
    style G fill:#fce4ec
    style H fill:#ffebee
    style I fill:#e8f5e8
    style J fill:#ffcdd2
```

## How to Use This File

1. **GitHub/GitLab**: This will render automatically in markdown files
2. **VS Code**: Install the "Markdown Preview Mermaid Support" extension
3. **Online**: Copy the code block to https://mermaid.live/
4. **Export**: Use mermaid-cli to convert to PNG/SVG: `mmdc -i VILA_ROBOT_PIPELINE_DIAGRAM.md -o pipeline.png`

## Pipeline Steps Explained

1. **📷 Camera Image**: Robot's camera captures environment
2. **🤖 VILA Model**: AI analyzes image and generates natural language response
3. **📝 Text Response**: VILA describes what it sees in human language
4. **🔍 Command Parser**: Uses flexible keyword detection (15+ forward patterns, hazard detection)
5. **⚙️ Command Generator**: Converts boolean flags to specific robot commands
6. **🛡️ Safety Check**: Multiple safety layers verify command is safe to execute
7. **🤖 Execute/Block**: Robot either moves or command is blocked for safety

This diagram shows the complete flow from vision to action in your robot system!