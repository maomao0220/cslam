/**
 * @file test_algorithm_killian.cpp
 * @author RJY (renjingyuan@whut.edu.cn)
 * @brief 测试多机器人图优化方法
 * @version 0.4
 * @date 2021-11-04
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <random>
#include <string>

#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <boost/thread/thread.hpp>

#include <Eigen/Core>

#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/slam3d/types_slam3d.h>

#include <sophus/se3.hpp>

using namespace std;
using namespace Eigen;
using namespace ros;

#define PI (3.1415926535897932346f)
typedef std::pair<Eigen::Vector3d, Eigen::Quaterniond> Pose;  // 位姿对


/**
 * @brief Get the pose object返回合适数据结构
 *
 * @param x
 * @param y
 * @param z
 * @param roll
 * @param pitch
 * @param yaw
 * @return Pose  逆时针旋转为正
 */
Pose get_pose(double x, double y, double z, double roll, double pitch, double yaw) {
    Eigen::AngleAxisd rollAngle(PI * roll, Vector3d(1, 0, 0));
    Eigen::AngleAxisd pitchAngle(PI * pitch, Vector3d(0, 1, 0));
    Eigen::AngleAxisd yawAngle(PI * yaw, Vector3d(0, 0, 1));
    Eigen::Quaterniond qd;  // 四元数
    qd = yawAngle * pitchAngle * rollAngle;
    qd.normalize();
    Eigen::Vector3d vd(x, y, z);  // 位置

    return Pose(vd, qd);
}


/**
 * @brief show_markers
 * 
 * @param markers 
 * @param nodes 
 */
void create_marker_array(visualization_msgs::MarkerArray &markers, const std::vector<g2o::VertexSE3 *> &nodes,
                         const std::vector<g2o::EdgeSE3 *> &edges) {
    // node markers
    visualization_msgs::Marker &traj_marker = markers.markers[0];
    traj_marker.header.frame_id = "map";
    traj_marker.header.stamp = ros::Time::now();
    traj_marker.ns = "nodes";
    traj_marker.id = 0;
    traj_marker.type = visualization_msgs::Marker::SPHERE_LIST;

    traj_marker.pose.orientation.w = 1.0;
    traj_marker.scale.x = traj_marker.scale.y = traj_marker.scale.z = 0.5;

    traj_marker.points.resize(nodes.size());
    traj_marker.colors.resize(nodes.size());

    for (int i = 0; i < nodes.size(); i++) {
        Eigen::Vector3d pos = nodes[i]->estimate().translation();
        traj_marker.points[i].x = pos.x();
        traj_marker.points[i].y = pos.y();
        traj_marker.points[i].z = pos.z();

        double p = static_cast<double>(i) / nodes.size();
        traj_marker.colors[i].r = 1.0 - p;
        traj_marker.colors[i].g = p;
        traj_marker.colors[i].b = 0.0;
        traj_marker.colors[i].a = 1.0;
    }

    // edge markers
    visualization_msgs::Marker &edge_marker = markers.markers[1];
    edge_marker.header.frame_id = "map";
    edge_marker.header.stamp = ros::Time::now();
    edge_marker.ns = "edges";
    edge_marker.id = 1;
    edge_marker.type = visualization_msgs::Marker::LINE_LIST;

    edge_marker.pose.orientation.w = 1.0;
    edge_marker.scale.x = 0.05;

    edge_marker.points.resize(edges.size() * 2);
    edge_marker.colors.resize(edges.size() * 2);

    int i = 0;
    for (auto &edge: edges) {
        // g2o::VertexSE3* v1 = dynamic_cast<g2o::VertexSE3*>(edge->vertices()[0]);
        // Eigen::Vector3d pt1 = v1->estimate().translation();
        // Eigen::Vector3d pt2 = Eigen::Vector3d::Zero();
        // pt2 = pt1 + edge->measurement().translation();
        g2o::VertexSE3 *v1 = dynamic_cast<g2o::VertexSE3 *>(edge->vertices()[0]);
        g2o::VertexSE3 *v2 = dynamic_cast<g2o::VertexSE3 *>(edge->vertices()[1]);
        if (v1 && v2) {
            Eigen::Vector3d pt1 = v1->estimate().translation();
            Eigen::Vector3d pt2 = v2->estimate().translation();

            edge_marker.points[i * 2].x = pt1.x();
            edge_marker.points[i * 2].y = pt1.y();
            edge_marker.points[i * 2].z = pt1.z() + 0.5;
            edge_marker.points[i * 2 + 1].x = pt2.x();
            edge_marker.points[i * 2 + 1].y = pt2.y();
            edge_marker.points[i * 2 + 1].z = pt2.z() + 0.5;

            edge_marker.colors[i * 2].r = 1.0;
            edge_marker.colors[i * 2].a = 1.0;
            edge_marker.colors[i * 2 + 1].r = 1.0;
            edge_marker.colors[i * 2 + 1].a = 1.0;
            i++;
        }
    }

    return;
}


/**
 * @brief main function
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, char **argv) {
    ros::init(argc, argv, "test_2rb_g2o");
    ros::NodeHandle nh("~");

    // init
    std::vector<g2o::VertexSE3 *> vectices;  // 顶点容器
    std::vector<g2o::EdgeSE3 *> edges;       // 边容器

    // publishers
    visualization_msgs::MarkerArray markers; // 显示标记
    markers.markers.resize(2);               // [0] robot node [1] edges
    ros::Publisher markers_pub = nh.advertise<visualization_msgs::MarkerArray>("/test_2rb_g2o/markers", 16);

    // 设定g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 6>> BlockSolverType;            // 每个误差项的优化变量维度为6,误差值维度为6
    typedef g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType> LinearSolverType;  // 线性求解器类型
    auto solver = new g2o::OptimizationAlgorithmLevenberg(                             // 梯度下降方法 可以选择高斯牛顿、LM、DogLeg
            g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;  // 图模型
    optimizer.setAlgorithm(solver);  // 设置求解器
    optimizer.setVerbose(true);      // 打开调试输出

    ROS_INFO_STREAM("loading data ...");

    // prepare
    std::string data_path = "/home/a2021-3/catkin_ws_cslam/src/cslam/src/cslam/killian-v.txt";
    ifstream fin(data_path);
    if (!fin) {
        cout << "file " << data_path << " does not exist." << endl;
        return 1;
    }

    // read data
    int vertexCnt = 0, edgeCnt = 0; // 顶点和边的数量
    double x, y, yaw;
    double inf_buf[6];
    while (!fin.eof()) {
        string name;
        fin >> name;
        if (name == "VERTEX2") {
            // SE3 顶点
            g2o::VertexSE3 *v = new g2o::VertexSE3();
            int index = 0;

            fin >> index >> x >> y >> yaw;

            Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
            auto temp = get_pose(x, y, 0, 0, 0, yaw);
            pose.prerotate(temp.second);
            pose.pretranslate(temp.first);

            v->setId(index);
            v->setEstimate(pose);
            optimizer.addVertex(v);
            vectices.push_back(v);
            vertexCnt++;
            if (index == 0)
                v->setFixed(true);
        } else if (name == "EDGE2") {
            // SE3-SE3 边
            g2o::EdgeSE3 *e = new g2o::EdgeSE3();
            int idx1, idx2;     // 关联的两个顶点

            fin >> idx1 >> idx2;
            fin >> x >> y >> yaw;
            for (int i = 0; i < 6; i++)
                fin >> inf_buf[i];

            Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
            auto temp = get_pose(x, y, 0, 0, 0, yaw);
            pose.prerotate(temp.second);
            pose.pretranslate(temp.first);

            e->setId(edgeCnt++);
            e->setVertex(0, optimizer.vertices()[idx1]);
            e->setVertex(1, optimizer.vertices()[idx2]);
            e->setMeasurement(pose);

            Eigen::Matrix<double, 6, 6> inf;    // 信息矩阵
            inf << inf_buf[0], 0, 0, 0, 0, 0,
                    0, inf_buf[2], 0, 0, 0, 0,
                    0, 0, inf_buf[0], 0, 0, 0,
                    0, 0, 0, inf_buf[3], 0, 0,
                    0, 0, 0, 0, inf_buf[3], 0,
                    0, 0, 0, 0, 0, inf_buf[3];

            e->setInformation(inf);

            optimizer.addEdge(e);
            edges.push_back(e);
        }
        if (!fin.good()) break;
    }

    // show
    ROS_INFO_STREAM("Finish " << vertexCnt << " vertices, " << edgeCnt << " edges.");
    ROS_INFO_STREAM("optimization start ...");
    ROS_INFO_STREAM("press g to optimize at most 512 times ...");
    ROS_INFO_STREAM("press q to quit ...");
    ROS_INFO_STREAM("press f to refresh res ...");

    // display marker
    optimizer.initializeOptimization();
    // optimizer.optimize(512);  // 优化三十次

    // spin
    char c;
    while (ros::ok()) {
        if (markers_pub.getNumSubscribers()) {
            create_marker_array(markers, vectices, edges);
            markers_pub.publish(markers);
        }
        ros::spinOnce();
        cin >> c;
        switch (c) {
            case 'q':
            case 'Q': {
                ROS_INFO_STREAM("quit ...");
                break;
            }
            case 'g':
            case 'G': {
                ROS_INFO_STREAM("start optimize at most 512 times ...");
                // optimizer.initializeOptimization();
                optimizer.optimize(30);  // 优化三十次
                break;
            }
            default:
                break;
        }
        if (c == 'q' || c == 'Q') break;

        ROS_INFO_STREAM("press g to optimize at most 512 times ...");
        ROS_INFO_STREAM("press q to quit ...");
    }

    return 0;
}