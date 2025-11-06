#!/usr/bin/env python3
"""
Test script for NLP Command Parser

This script tests the NLP parser with various natural language commands
to verify it can handle different variations and edge cases.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'robot_vila_system'))

from robot_vila_system.nlp_command_parser import NLPCommandParser

def test_nlp_parser():
    """Test NLP parser with various commands"""

    print("=" * 80)
    print("NLP Command Parser Test")
    print("=" * 80)

    # Initialize parser
    print("\n[1/3] Initializing NLP parser...")
    parser = NLPCommandParser(device="cuda")

    print("[2/3] Loading model (this may take a minute)...")
    parser.load_model()

    print("[3/3] Running test commands...\n")

    # Test commands grouped by category
    test_commands = {
        "Simple Linear Movement": [
            "move forward 1 meter",
            "go ahead 2m",
            "move forward 1.5 meters",
            "back up 0.5 meters",
            "reverse 1m",
        ],
        "Rotational Movement": [
            "turn left 90 degrees",
            "rotate right 45 deg",
            "turn slowly clockwise until you see the door",
            "rotate left 180 degrees",
        ],
        "Complex Natural Language": [
            "move to 1m in front of the refrigerator",
            "go to the door",
            "navigate to the kitchen",
            "approach the table",
        ],
        "Lateral Movement": [
            "strafe left 1 meter",
            "strafe right 0.5m",
            "move sideways left",
        ],
        "Control Commands": [
            "stop",
            "halt",
            "emergency stop",
        ],
        "Variations & Edge Cases": [
            "advance 3 meters",
            "move ahead by 2.5m",
            "spin right 360 degrees",
            "go forward for 2 seconds",
        ]
    }

    # Run tests
    total_tests = 0
    passed_tests = 0

    for category, commands in test_commands.items():
        print(f"\n{'='*80}")
        print(f"Category: {category}")
        print(f"{'='*80}")

        for cmd in commands:
            total_tests += 1
            print(f"\n  Input: '{cmd}'")

            try:
                result = parser.parse_command(cmd)

                print(f"  ✓ Action:     {result.action}")
                if result.parameters:
                    print(f"    Parameters: {result.parameters}")
                print(f"    Vision:     {result.needs_vision}")
                print(f"    Confidence: {result.confidence:.2f}")

                # Debug: Show raw response if there's an error
                if result.confidence == 0.0:
                    print(f"    Raw Response: {result.raw_response[:200]}")

                # Basic validation
                if result.action and result.confidence > 0.5:
                    passed_tests += 1
                    status = "✓ PASS"
                else:
                    status = "✗ FAIL (low confidence or no action)"

                print(f"  Status: {status}")

            except Exception as e:
                print(f"  ✗ ERROR: {e}")

    # Summary
    print(f"\n{'='*80}")
    print("Test Summary")
    print(f"{'='*80}")
    print(f"Total Tests:  {total_tests}")
    print(f"Passed:       {passed_tests}")
    print(f"Failed:       {total_tests - passed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    print()

    return passed_tests == total_tests

if __name__ == "__main__":
    try:
        success = test_nlp_parser()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
