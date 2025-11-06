cmake_minimum_required(VERSION 3.8)
project(robot_sim)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)

# Install launch files, worlds, config, and models
install(DIRECTORY
  launch
  worlds
  config
  models
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()

