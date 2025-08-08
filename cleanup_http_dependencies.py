#!/usr/bin/env python3
"""
HTTP Dependencies Cleanup Script
Identifies and documents HTTP-related code that should be removed
after migration to ROS2 messaging
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Tuple

class HTTPDependencyAnalyzer:
    """Analyzes codebase for HTTP dependencies that can be removed"""
    
    def __init__(self, workspace_path: str):
        self.workspace_path = Path(workspace_path)
        self.http_patterns = [
            # Import patterns
            r'import\s+(requests|flask|aiohttp|httpx|websockets|socketio)',
            r'from\s+(requests|flask|aiohttp|httpx|websockets|socketio)',
            r'import\s+.*\.(get|post|put|delete|request)',
            
            # HTTP URL patterns
            r'http://|https://',
            r'localhost:\d+',
            r':\d+/',
            
            # HTTP method calls
            r'\.get\(.*http',
            r'\.post\(.*http',
            r'\.put\(.*http',
            r'\.delete\(.*http',
            r'requests\.',
            r'aiohttp\.',
            r'httpx\.',
            
            # Flask patterns
            r'@app\.route',
            r'Flask\(',
            r'app\.run\(',
            r'jsonify\(',
            
            # WebSocket patterns
            r'socketio\.',
            r'websocket',
            r'emit\(',
            r'@.*\.event',
        ]
        
        # Files that are now replaced by ROS2 versions
        self.deprecated_files = [
            'robot_vila_server.py',      # → robot_vila_server_ros2.py
            'vila_ros_node.py',          # → vila_ros2_node.py (ROS1 → ROS2)
            'robot_gui.py',              # → robot_gui_ros2.py
            'unified_robot_controller.py', # → Not needed (HTTP-based)
            'unified_robot_client.py',   # → robot_client_ros2.py
            'robot_client_examples.py',  # → Examples replaced
            'start_unified_system.sh',   # → launch_ros2_system.py
        ]
        
        # Files to keep (ROS2 versions and core files)
        self.keep_files = [
            'robot_vila_server_ros2.py',
            'vila_ros2_node.py', 
            'robot_gui_ros2.py',
            'robot_client_ros2.py',
            'launch_ros2_system.py',
            'build_ros2_system.sh',
            'ros2_command_gateway_validator.py',
            'main_vila.py',  # Core VILA functionality
        ]
    
    def scan_file(self, file_path: Path) -> Dict[str, List[Tuple[int, str]]]:
        """Scan a file for HTTP dependencies"""
        results = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for pattern in self.http_patterns:
                matches = []
                for line_num, line in enumerate(lines, 1):
                    if re.search(pattern, line, re.IGNORECASE):
                        matches.append((line_num, line.strip()))
                
                if matches:
                    results[pattern] = matches
        
        except Exception as e:
            print(f"Error scanning {file_path}: {e}")
        
        return results
    
    def analyze_workspace(self) -> Dict[str, Dict]:
        """Analyze entire workspace for HTTP dependencies"""
        results = {
            'deprecated_files': [],
            'files_with_http': {},
            'clean_files': [],
            'summary': {}
        }
        
        # Find all Python files
        python_files = list(self.workspace_path.glob('**/*.py'))
        
        for file_path in python_files:
            # Skip hidden files and directories
            if any(part.startswith('.') for part in file_path.parts):
                continue
            
            relative_path = file_path.relative_to(self.workspace_path)
            
            # Check if file is deprecated
            if relative_path.name in self.deprecated_files:
                results['deprecated_files'].append(str(relative_path))
                continue
            
            # Check if file should be kept
            if relative_path.name in self.keep_files:
                results['clean_files'].append(str(relative_path))
                continue
            
            # Scan for HTTP dependencies
            http_deps = self.scan_file(file_path)
            
            if http_deps:
                results['files_with_http'][str(relative_path)] = http_deps
            else:
                results['clean_files'].append(str(relative_path))
        
        # Generate summary
        results['summary'] = {
            'total_files': len(python_files),
            'deprecated_files': len(results['deprecated_files']),
            'files_with_http': len(results['files_with_http']),
            'clean_files': len(results['clean_files'])
        }
        
        return results
    
    def generate_report(self, results: Dict) -> str:
        """Generate a detailed cleanup report"""
        report = []
        report.append("# HTTP Dependencies Cleanup Report")
        report.append("=" * 50)
        report.append("")
        
        # Summary
        summary = results['summary']
        report.append("## Summary")
        report.append(f"- Total Python files: {summary['total_files']}")
        report.append(f"- Deprecated files (can be removed): {summary['deprecated_files']}")
        report.append(f"- Files with HTTP dependencies: {summary['files_with_http']}")
        report.append(f"- Clean ROS2 files: {summary['clean_files']}")
        report.append("")
        
        # Deprecated files
        if results['deprecated_files']:
            report.append("## 🗑️ Deprecated Files (Safe to Remove)")
            report.append("These files have been replaced by ROS2 versions:")
            report.append("")
            for file_path in results['deprecated_files']:
                report.append(f"- ❌ {file_path}")
            report.append("")
        
        # Files with HTTP dependencies
        if results['files_with_http']:
            report.append("## ⚠️ Files with HTTP Dependencies")
            report.append("These files still contain HTTP-related code:")
            report.append("")
            
            for file_path, dependencies in results['files_with_http'].items():
                report.append(f"### {file_path}")
                for pattern, matches in dependencies.items():
                    report.append(f"**Pattern: `{pattern}`**")
                    for line_num, line in matches[:5]:  # Show first 5 matches
                        report.append(f"  - Line {line_num}: `{line}`")
                    if len(matches) > 5:
                        report.append(f"  - ... and {len(matches) - 5} more matches")
                    report.append("")
        
        # Clean files
        if results['clean_files']:
            report.append("## ✅ Clean ROS2 Files")
            report.append("These files are already ROS2-compatible:")
            report.append("")
            for file_path in results['clean_files']:
                report.append(f"- ✅ {file_path}")
            report.append("")
        
        # Cleanup recommendations
        report.append("## 🧹 Cleanup Recommendations")
        report.append("")
        report.append("### 1. Remove Deprecated Files")
        report.append("```bash")
        for file_path in results['deprecated_files']:
            report.append(f"# mv {file_path} deprecated/  # or rm {file_path}")
        report.append("```")
        report.append("")
        
        report.append("### 2. Update Requirements")
        report.append("Remove HTTP dependencies from requirements.txt:")
        report.append("```")
        report.append("# Remove these packages:")
        report.append("# requests")
        report.append("# flask") 
        report.append("# flask-socketio")
        report.append("# python-socketio")
        report.append("# aiohttp")
        report.append("# websockets")
        report.append("```")
        report.append("")
        
        report.append("### 3. Use ROS2 Requirements")
        report.append("```bash")
        report.append("pip install -r requirements_ros2.txt")
        report.append("```")
        report.append("")
        
        return "\n".join(report)
    
    def create_deprecated_backup(self) -> None:
        """Create backup directory for deprecated files"""
        deprecated_dir = self.workspace_path / "deprecated_http_files"
        deprecated_dir.mkdir(exist_ok=True)
        
        print(f"📁 Created backup directory: {deprecated_dir}")
        
        # Create README in deprecated directory
        readme_content = """# Deprecated HTTP Files

This directory contains the original HTTP-based implementation files
that have been replaced by ROS2 versions.

## Migration Mapping
- robot_vila_server.py → robot_vila_server_ros2.py
- vila_ros_node.py (ROS1) → vila_ros2_node.py (ROS2)
- robot_gui.py → robot_gui_ros2.py
- unified_robot_controller.py → (functionality merged into ROS2 server)
- unified_robot_client.py → robot_client_ros2.py
- robot_client_examples.py → (examples updated for ROS2)
- start_unified_system.sh → launch_ros2_system.py

These files are kept for reference but should not be used in the ROS2 system.
"""
        
        with open(deprecated_dir / "README.md", "w") as f:
            f.write(readme_content)

def main():
    """Main function"""
    workspace = "/home/marc/Robot_LLM"
    analyzer = HTTPDependencyAnalyzer(workspace)
    
    print("🔍 Analyzing workspace for HTTP dependencies...")
    results = analyzer.analyze_workspace()
    
    print("📋 Generating cleanup report...")
    report = analyzer.generate_report(results)
    
    # Save report
    report_file = Path(workspace) / "http_cleanup_report.md"
    with open(report_file, "w") as f:
        f.write(report)
    
    print(f"📄 Report saved to: {report_file}")
    
    # Create deprecated backup directory
    analyzer.create_deprecated_backup()
    
    # Print summary
    print("\n📊 CLEANUP SUMMARY:")
    print(f"   └── Total files analyzed: {results['summary']['total_files']}")
    print(f"   └── Deprecated files: {results['summary']['deprecated_files']}")
    print(f"   └── Files with HTTP deps: {results['summary']['files_with_http']}")
    print(f"   └── Clean ROS2 files: {results['summary']['clean_files']}")
    
    if results['deprecated_files']:
        print(f"\n🗑️ Files ready for removal:")
        for file_path in results['deprecated_files']:
            print(f"   └── {file_path}")
    
    print(f"\n📋 Full report: {report_file}")
    print("🎉 Analysis complete!")

if __name__ == "__main__":
    main()
