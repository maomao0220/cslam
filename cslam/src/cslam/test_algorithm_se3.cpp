/**
 * @file test_algorithm_se3.cpp
 * @author RJY (renjingyuan@whut.edu.cn)
 * @brief 测试多机器人图优化方法
 * @version 0.2
 * @date 2021-11-01
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

typedef Matrix<double, 6, 6> Matrix6d;
typedef std::pair<Eigen::Vector3d, Eigen::Quaterniond> Pose;  // 位姿对
typedef std::pair<int, int> SpecialNodes;                     // 特殊节点对
typedef Matrix<double, 6, 1> Vector6d;                        // 李代数顶点
#define random(a, b) (rand() % (b - a) + a)


/**
 * @brief Get the pose object返回合适数据结构
 * 
 * @param x 
 * @param y 
 * @param z 
 * @param roll 
 * @param pitch 
 * @param yaw 
 * @return Pose 
 */
Pose get_pose(double x, double y, double z, double roll, double pitch, double yaw) {
    Eigen::Vector3d eulerAngle(roll, pitch, yaw);  // 四元数转欧拉角
    Eigen::AngleAxisd rollAngle(AngleAxisd(eulerAngle(0), Vector3d::UnitX()));
    Eigen::AngleAxisd pitchAngle(AngleAxisd(eulerAngle(1), Vector3d::UnitY()));
    Eigen::AngleAxisd yawAngle(AngleAxisd(eulerAngle(2), Vector3d::UnitZ()));
    Eigen::Quaterniond qd;  // 四元数
    qd = yawAngle * pitchAngle * rollAngle;
    Eigen::Vector3d vd(x, y, z);  // 位置

    return Pose(vd, qd);
}

/**
 * @brief Get the pose data object
 * 
 * @param file_path 
 * @param truth 
 * @param real 
 * @param pc 
 * @param m 
 * @param n 
 * @return true 
 * @return false 
 */
bool get_pose_data(std::string &file_path, std::vector<Pose> &truth, std::vector<Pose> &real, std::vector<Pose> &pc,
                   double m, double n) {
    std::ifstream fin(file_path);
    if (!fin) {
        ROS_INFO_STREAM("file " << file_path << " does not exist.");
        return false;
    }

    unsigned int id = 0;
    double x = 0, y = 0, z = 0, roll = 0, pitch = 0, yaw = 0;
    double x_n = 0, y_n = 0, z_n = 0, roll_n = 0, pitch_n = 0, yaw_n = 0;
    double x_before = 0, y_before = 0, yaw_before = 0;        // 不带噪声的前一个xyz位姿 仿真时z r p一直为0，不考虑
    double x_n_before = 0, y_n_before = 0, yaw_n_before = 0;  // 带噪声的前一个xyz位姿 仿真时z r p一直为0，不考虑
    double delta_x = 0, delta_y = 0, delta_yaw = 0;           // 前后数据差 真实位姿变化
    double delta_x_n = 0, delta_y_n = 0, delta_yaw_n = 0;     // 前后数据差 带有噪声位姿变化

    id = 0;
    while (!fin.eof()) {
        fin >> x >> y >> z >> roll >> pitch >> yaw;

        roll = roll * PI;
        pitch = pitch * PI;
        yaw = yaw * PI;

        id++;  // 节点id
        auto temp = get_pose(x, y, z, roll, pitch, yaw);
        truth.push_back(temp);  // 添加机器人1的位置真值

        // 添加机器人1带噪声的位姿
        if (1 == id) {
            // 初始化 应该都为0
            x_n_before = x_before = x;
            y_n_before = y_before = y;
            yaw_n_before = yaw_before = yaw;

            real.push_back(temp);  // 初始点不带噪声
        } else {
            // 本次主要的位姿变化，真实变化
            delta_x = x - x_before;
            delta_y = y - y_before;
            delta_yaw = yaw - yaw_before;

            // 带有噪声的变化
            srand((int) time(NULL) + id);                     // 产生随机种子  把0换成NULL也行
            delta_x_n = delta_x + random(-50, 50) * m;  // -50 50 0.01 -> -0.5 0.5
            delta_y_n = delta_y + random(-50, 50) * m;
            delta_yaw_n = delta_yaw + random(-50, 50) * n;  // -50 50 0.005 -> -0.025 0.025
            auto temp_pc_n = get_pose(delta_x_n, delta_y_n, 0, 0, 0, delta_yaw_n);
            pc.push_back(temp_pc_n);

            // 加噪声 累计误差模拟，带误差的变化
            x_n = x_n_before + delta_x_n;  // -50 50 0.01 -> -0.5 0.5
            y_n = y_n_before + delta_y_n;
            yaw_n = yaw_n_before + delta_yaw_n;  // -50 50 0.005 -> -0.025 0.025
            auto temp_pose_n = get_pose(x_n, y_n, z, roll, pitch, yaw_n);
            real.push_back(temp_pose_n);

            // 更新存储过去量
            x_before = x;
            y_before = y;
            yaw_before = yaw;
            x_n_before = x_n;
            y_n_before = y_n;
            yaw_n_before = yaw_n;
        }
    }
    fin.close();

    return true;
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
        g2o::EdgeSE3 *v1 = dynamic_cast<g2o::EdgeSE3 *>(edge->vertices()[0]);
        g2o::EdgeSE3 *v2 = dynamic_cast<g2o::EdgeSE3 *>(edge->vertices()[1]);
        if (v1 && v2) {
            Eigen::Vector3d pt1 = v1->measurement().translation();
            Eigen::Vector3d pt2 = v2->measurement().translation();

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
    std::vector<Pose> pose_truth_rb1;  // ground truth rb1
    std::vector<Pose> pose_truth_rb2;  // ground truth rb2

    std::vector<Pose> pose_real_rb1;  // 实际带噪声的位姿
    std::vector<Pose> pose_real_rb2;  // 实际带噪声的位姿

    std::vector<Pose> pose_change_n_rb1;  // 带有噪声的位姿变化
    std::vector<Pose> pose_change_n_rb2;  // 带有噪声的位姿变化

    const unsigned int rb1_node_start = 0;
    const unsigned int rb2_node_start = 10000;
    const unsigned int sp_edge_start = 20000;

    std::vector<g2o::VertexSE3 *> vectices;  // 顶点容器
    std::vector<g2o::EdgeSE3 *> edges;       // 边容器

    std::vector<SpecialNodes> sp_nodes;  // 容器

    Eigen::Matrix<double, 6, 6> inf;     // 信息矩阵
    inf << 10000, 0, 0, 0, 0, 0,
            0, 10000, 0, 0, 0, 0,
            0, 0, 10000, 0, 0, 0,
            0, 0, 0, 40000, 0, 0,
            0, 0, 0, 0, 40000, 0,
            0, 0, 0, 0, 0, 40000;

    // publishers
    visualization_msgs::MarkerArray markers; // 显示标记
    markers.markers.resize(2);               // [0] robot node [1] edges
    ros::Publisher markers_pub = nh.advertise<visualization_msgs::MarkerArray>("/test_2rb_g2o/markers", 16);

    // 坐标系关系 坐标系变换 已知坐标系关系
    Eigen::Isometry3d rb2_2_rb1_trans = Eigen::Isometry3d::Identity();
    // c rb1 3, 1.5, 0, 0, 0, -pi
    // c rb2 4, 0, 0, 0, 0 ,0
    Eigen::Isometry3d rb2_c = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d rb1_c = Eigen::Isometry3d::Identity();
    auto rb1_pose = get_pose(3, 1.5, 0, 0, 0, 1 * PI);
    auto rb2_pose = get_pose(4, 0, 0, 0, 0, 0);
    rb1_c.prerotate(rb1_pose.second);
    rb1_c.pretranslate(rb1_pose.first);
    rb2_c.prerotate(rb2_pose.second);
    rb2_c.pretranslate(rb2_pose.first);
    rb2_2_rb1_trans = rb1_c * rb2_c.inverse();

    // 噪声符合的分布
    const double n_step_rb1 = 0.005, n_yaw_step_rb1 = 0.0005;
    const double n_step_rb2 = 0.005, n_yaw_step_rb2 = 0.0005;

    // read data robot 1 2
    std::string data_path_rb1 = "/home/a2021-3/catkin_ws_cslam/src/cslam/src/cslam/test_data_r1.txt";
    std::string data_path_rb2 = "/home/a2021-3/catkin_ws_cslam/src/cslam/src/cslam/test_data_r2.txt";
    if (!get_pose_data(data_path_rb1, pose_truth_rb1, pose_real_rb1, pose_change_n_rb1, n_step_rb1, n_yaw_step_rb1)) {
        ROS_INFO_STREAM("DATA 1 READ ERROR");
        return -1;
    }
    if (!get_pose_data(data_path_rb2, pose_truth_rb2, pose_real_rb2, pose_change_n_rb2, n_step_rb2, n_yaw_step_rb2)) {
        ROS_INFO_STREAM("DATA 2 READ ERROR");
        return -1;
    }

    // 设定g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 6>> BlockSolverType;            // 每个误差项的优化变量维度为6,误差值维度为6
    typedef g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType> LinearSolverType;  // 线性求解器类型
    auto solver = new g2o::OptimizationAlgorithmLevenberg(                             // 梯度下降方法 可以选择高斯牛顿、LM、DogLeg
            g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;  // 图模型
    optimizer.setAlgorithm(solver);  // 设置求解器
    optimizer.setVerbose(true);      // 打开调试输出

    ROS_INFO_STREAM("loading data ...");

    // robot1 设置节点
    int vertexCnt_rb1 = 0, edgeCnt_rb1 = 0;  // 顶点和边的数量
    unsigned int id = rb1_node_start;
    for (auto &p: pose_real_rb1) {
        // 顶点
        Eigen::Isometry3d temp = Eigen::Isometry3d::Identity();
        temp.prerotate(p.second);
        temp.pretranslate(p.first);

        g2o::VertexSE3 *v = new g2o::VertexSE3();
        v->setId(id);
        v->setEstimate(temp);

        optimizer.addVertex(v); // 优化器添加节点

        ROS_INFO_STREAM("add node, id: " << id);
        vertexCnt_rb1++;
        vectices.push_back(v);  // 节点存储

        // 设置边
        if (0 == id) {  // 以机器人一的坐标系为基准
            v->setFixed(true);
        } else {
            // SE3-SE3 边
            Eigen::Isometry3d temp = Eigen::Isometry3d::Identity();
            temp.prerotate(pose_change_n_rb1[id - 1].second);
            temp.pretranslate(pose_change_n_rb1[id - 1].first);

            g2o::EdgeSE3 *e = new g2o::EdgeSE3();
            int idx1 = id - 1, idx2 = id;           // 关联的前后两个顶点
            edgeCnt_rb1++;
            e->setId(id);                              // 设置边的id
            e->setVertex(0, optimizer.vertices()[idx1]);  // 设置边的两端节点 vertices是map类型
            e->setVertex(1, optimizer.vertices()[idx2]);
            e->setMeasurement(temp);
            e->setInformation(inf);

            optimizer.addEdge(e);  // 添加边

            ROS_INFO_STREAM("add edge, id: " << idx1 << " to " << " id : " << idx2);
            edges.push_back(e);
        }
        id++;
    }


    // robot2 设置节点 和边
    int vertexCnt_rb2 = 0, edgeCnt_rb2 = 0;  // 顶点和边的数量
    id = rb2_node_start;
    for (auto &p: pose_real_rb2) {
        // 顶点
        g2o::VertexSE3 *v = new g2o::VertexSE3();
        v->setId(id);

        // 坐标系转换
        Eigen::Isometry3d rb2_in_rb1 = Eigen::Isometry3d::Identity();
        rb2_in_rb1.prerotate(p.second);
        rb2_in_rb1.pretranslate(p.first);
        rb2_in_rb1 = rb2_2_rb1_trans * rb2_in_rb1; // rb2_2_rb1_trans

        // 设置节点
        v->setEstimate(rb2_in_rb1);

        optimizer.addVertex(v);

        ROS_INFO_STREAM("add node, id: " << id);
        vertexCnt_rb2++;
        vectices.push_back(v);

        // SE3-SE3 边 从10001开始
        if (rb2_node_start == id) {  // 以机器人一的坐标系为基准
            v->setFixed(true);
        } else {
            // SE3-SE3 边
            Eigen::Isometry3d temp = Eigen::Isometry3d::Identity();
            temp.prerotate(pose_change_n_rb2[id - rb2_node_start - 1].second);
            temp.pretranslate(pose_change_n_rb2[id - rb2_node_start - 1].first);

            g2o::EdgeSE3 *e = new g2o::EdgeSE3();
            int idx1 = id - 1, idx2 = id;           // 关联的前后两个顶点
            edgeCnt_rb2++;
            e->setId(id);                              // 设置边的id
            e->setVertex(0, optimizer.vertices()[idx1]);  // 设置边的两端节点
            e->setVertex(1, optimizer.vertices()[idx2]);
            e->setMeasurement(temp);
            e->setInformation(inf);

            optimizer.addEdge(e); // 添加边

            ROS_INFO_STREAM("add edge, id: " << idx1 << " to " << " id : " << idx2);
            edges.push_back(e);
        }
        id++;
    }


    // 回环节点
    std::string data_path_rb1_2_rb2 = "/home/a2021-3/catkin_ws_cslam/src/cslam/src/cslam/test_data_r1_2_r2.txt";
    std::ifstream fin(data_path_rb1_2_rb2);
    if (!fin) {
        ROS_INFO_STREAM("file " << data_path_rb1_2_rb2 << " does not exist.");
        return -1;
    }
    unsigned int idx1 = 0, idx2 = 0;
    double x = 0, y = 0, z = 0, roll = 0, pitch = 0, yaw = 0;
    id = sp_edge_start;
    while (!fin.eof()) {
        fin >> idx1 >> idx2;
        idx1 = idx1 + rb1_node_start;
        idx2 = idx2 + rb2_node_start;

        fin >> x >> y >> z >> roll >> pitch >> yaw;
        auto temp_pc = get_pose(x, y, z, roll, pitch, yaw);

        // SE3-SE3 边
        Eigen::Isometry3d temp = Eigen::Isometry3d::Identity();
        temp.prerotate(temp_pc.second);
        temp.pretranslate(temp_pc.first);

        g2o::EdgeSE3 *e = new g2o::EdgeSE3();
        e->setId(id);                                 // 设置边的id
        e->setVertex(0, optimizer.vertices()[idx1]);  // 设置边的两端节点
        e->setVertex(1, optimizer.vertices()[idx2]);
        e->setMeasurement(temp);
        e->setInformation(inf);

        optimizer.addEdge(e);  // 添加边

        ROS_INFO_STREAM("add edge, id: " << idx1 << " to " << " id : " << idx2);
        edges.push_back(e);
        id++;
    }

    int i = 1;
    for (auto &node: vectices) {
        cout << "node ID" << i << " DATA " << node->estimate().translation() << endl
             << node->estimate().rotation().eulerAngles(0, 1, 2) << endl;
        i++;
    }
    i = 1;
    for (auto &edge: edges) {
        cout << "edge ID" << i << " DATA " << edge->measurement().translation() << endl
             << edge->measurement().rotation().eulerAngles(0, 1, 2) << endl;
        i++;
    }


    // show
    ROS_INFO_STREAM("robot 1 " << vertexCnt_rb1 << " vertices, " << edgeCnt_rb1 << " edges.");
    ROS_INFO_STREAM("robot 2 " << vertexCnt_rb2 << " vertices, " << edgeCnt_rb2 << " edges.");
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
                optimizer.optimize(512);  // 优化三十次
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