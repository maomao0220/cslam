## cmake version required
cmake_minimum_required(VERSION 3.0.2)


## project name
project(cslam)


## Compile as C++11, supported in ROS Kinetic and newer
#set(CMAKE_BUILD_TYPE "Release")
#add_definitions("-DENABLE_SSE")
#set(CMAKE_CXX_FLAGS "-std=c++14 -O2 ${SSE_FLAGS} -msse4")
add_compile_options(-std=c++14)


## find_package catkin
find_package(catkin REQUIRED COMPONENTS
        roscpp
        rospy
        sensor_msgs
        std_msgs
        )


## find_package outside
# eigen3
find_package(Eigen3 3.3 REQUIRED)
include_directories(EIGEN3_INCLUDE_DIR)

message(STATUS "EIGEN3_INCLUDE_DIRS:" ${EIGEN3_INCLUDE_DIR})
message(STATUS "EIGEN3_LIBRARY_DIRS:" ${EIGEN3_LIBRARY_DIRS})

# pcl
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

message(STATUS "PCL_INCLUDE_DIRS:" ${PCL_INCLUDE_DIRS})
message(STATUS "PCL_LIBRARY_DIRS:" ${PCL_LIBRARY_DIRS})
message(STATUS "PCL_DEFINITIONS:" ${PCL_DEFINITIONS})

# g2o
list(APPEND CMAKE_MODULE_PATH /home/a2021-3/g2o/cmake_modules)
find_package(G2O REQUIRED)
include_directories(SYSTEM ${G2O_INCLUDE_DIR} ${G2O_INCLUDE_DIRS})
set(G2O_LIBRARY_DIRS "/usr/local/lib")
link_directories(${G2O_LIBRARY_DIRS})

message(STATUS "G2O_INCLUDE_DIRS:" ${G2O_INCLUDE_DIR})
message(STATUS "G2O_LIBRARY_DIRS:" ${G2O_LIBRARY_DIRS})
message(STATUS "G2O_LIBRARYS:" ${G2O_LIBRARIES})

# sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

message(STATUS "Sophus_INCLUDE_DIRS:" ${Sophus_INCLUDE_DIRS})
message(STATUS "Sophus_LIBRARYS:" ${Sophus_LIBRARIES})

# glog
find_package(glog 0.6.0 REQUIRED)


## catkin specific configuration
catkin_package(
        INCLUDE_DIRS include
        #  LIBRARIES cslam
        #  CATKIN_DEPENDS pcl_ros roscpp sensor_msgs
        #  DEPENDS system_lib
)


## catkin include 
include_directories(
        include
        ${catkin_INCLUDE_DIRS}
)


## add_executable
add_executable(test_algorithm src/cslam/test_algorithm.cpp)
target_link_libraries(test_algorithm
        ${catkin_LIBRARIES}
        ${PCL_LIBRARIES}
        g2o_core
        g2o_stuff
        g2o_types_slam3d glog::glog
        ${CHOLMOD_LIBRARIES}
        )

add_executable(test_algorithm_se3 src/cslam/test_algorithm_se3.cpp)
target_link_libraries(test_algorithm_se3
        ${catkin_LIBRARIES}
        ${PCL_LIBRARIES}
        g2o_core
        g2o_stuff
        g2o_types_slam3d glog::glog
        ${CHOLMOD_LIBRARIES}
        )

add_executable(test_algorithm_lie src/cslam/test_lie.cpp)
target_link_libraries(test_algorithm_lie
        ${catkin_LIBRARIES}
        ${PCL_LIBRARIES}
        g2o_core
        g2o_stuff
        g2o_types_slam3d glog::glog
        ${CHOLMOD_LIBRARIES}
        )

add_executable(test_algorithm_killian src/cslam/test_algorithm_killian.cpp)
target_link_libraries(test_algorithm_killian
        ${catkin_LIBRARIES}
        ${PCL_LIBRARIES}
        g2o_core
        g2o_stuff
        g2o_types_slam3d glog::glog
        ${CHOLMOD_LIBRARIES}
        )

