cmake_minimum_required(VERSION 3.0.2)
project(formation_hallway)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  gazebo_ros
)

catkin_package()

include_directories(${catkin_INCLUDE_DIRS})

add_subdirectory(plugins)

catkin_install_python(PROGRAMS
  scripts/spawn_hallway_fleet.py
  scripts/teleop_hallway.py
  scripts/hallway_env_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
