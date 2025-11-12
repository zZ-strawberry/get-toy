cmake_minimum_required(VERSION 3.8)
project(color_tracking_node)

# 设置C++标准
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 查找依赖包
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED)

# 包含目录
include_directories(include)

# 创建可执行文件
add_executable(color_tracking_node src/color_tracking_node.cpp)
add_executable(basket src/basket.cpp)

# 链接库
ament_target_dependencies(color_tracking_node
  rclcpp
  std_msgs
  geometry_msgs
  sensor_msgs
  cv_bridge
  image_transport
  OpenCV
)
ament_target_dependencies(basket
  rclcpp
  sensor_msgs
  geometry_msgs
  std_msgs
  cv_bridge
  OpenCV
)

# 安装可执行文件
install(TARGETS color_tracking_node
  DESTINATION lib/${PROJECT_NAME}
)
install(TARGETS basket DESTINATION lib/${PROJECT_NAME})

# 安装launch文件
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

# 安装配置文件
install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

# 导出依赖
ament_export_dependencies(rclcpp std_msgs geometry_msgs sensor_msgs cv_bridge image_transport OpenCV)

ament_package()
