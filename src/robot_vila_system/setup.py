from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'robot_vila_system'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Launch files
        (os.path.join('share', package_name, 'launch'),
         glob('launch/*.launch.py')),
        # Config files
        (os.path.join('share', package_name, 'config'),
         glob('config/*.yaml') if os.path.exists('config') else []),
        # GUI component files
        (os.path.join('share', package_name, 'robot_vila_system'),
         ['robot_vila_system/gui_config.py',
          'robot_vila_system/gui_utils.py',
          'robot_vila_system/gui_components.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Robot Developer',
    maintainer_email='robot@example.com',
    description='ROS2 Local VLM Navigation Robot System',
    license='MIT',
    entry_points={
        'console_scripts': [
            'robot_gui_node.py = robot_vila_system.robot_gui_node:main',
            'gateway_validator_node.py = robot_vila_system.gateway_validator_node:main',
            'local_vlm_navigation_node.py = robot_vila_system.local_vlm_navigation_node:main',

        ],
    },
)
