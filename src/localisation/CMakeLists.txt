cmake_minimum_required(VERSION 3.5)
project(localisation_pkg)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Ofast)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/Localisation.srv"
  DEPENDENCIES std_msgs
)

add_executable(geneticLocaliserServer src/genetic.cpp)
ament_target_dependencies(geneticLocaliserServer rclcpp std_msgs OpenCV)

rosidl_get_typesupport_target(cpp_typesupport_target
  ${PROJECT_NAME} rosidl_typesupport_cpp)

target_link_libraries(geneticLocaliserServer "${cpp_typesupport_target}")

install(TARGETS
  geneticLocaliserServer
  DESTINATION lib/${PROJECT_NAME})

ament_package()