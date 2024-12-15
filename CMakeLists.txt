cmake_minimum_required(VERSION 3.0.2)
project(TATOS_cpp_v1)



# 查找 catkin 包和所需的组件
find_package(catkin REQUIRED COMPONENTS
        roscpp
        rosbag
        sensor_msgs
        pcl_ros
        pcl_conversions
)

# 查找 PCL 库，并确保可视化组件可用
find_package(PCL REQUIRED COMPONENTS common io visualization)
find_package(Ceres REQUIRED)
find_package(NLopt REQUIRED)
set(NLopt_INCLUDE_DIRS /usr/local/include)
set(NLopt_LIBRARIES /usr/local/lib/libnlopt.so)

# 声明 catkin 包
catkin_package()

# 添加包含目录
include_directories(
        ${catkin_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}  # 包含 PCL 头文件
        ${NLopt_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/include  # 添加include
)

# 链接目录
link_directories(${PCL_LIBRARY_DIRS})

# 添加编译选项
add_definitions(${PCL_DEFINITIONS})


# 创建可执行文件
add_executable(tatos src/tatos.cpp
        include/utils.h
        include/segmentation.h
        include/visualizatioin.h
)

# 链接库
target_link_libraries(tatos
        ${catkin_LIBRARIES}
        ${PCL_LIBRARIES}  # 链接 PCL 库
        ${CERES_LIBRARIES}
        ${NLopt_LIBRARIES}
)
