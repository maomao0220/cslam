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
bool get_pose_data(std::string &file_path, std::vector<Eigen::Isometry3d> &truth, std::vector<Eigen::Isometry3d> &real,
                   std::vector<Eigen::Isometry3d> &pc,
                   std::vector<Eigen::Isometry3d> &noise, double m, double n) {
    std::ifstream fin(file_path);
    if (!fin) {
        ROS_INFO_STREAM("file " << file_path << " does not exist.");
        return false;
    }

    unsigned int id = 0;
    double x = 0, y = 0, z = 0, roll = 0, pitch = 0, yaw = 0;

    Eigen::Isometry3d node_pose_before = Eigen::Isometry3d::Identity(); // 之前的真实位姿
    Eigen::Isometry3d node_pose_now = Eigen::Isometry3d::Identity();    // 当前的真实位姿
    Eigen::Isometry3d node_pose_real = Eigen::Isometry3d::Identity();   // 测量位姿带噪声
    Eigen::Isometry3d node_pose_change = Eigen::Isometry3d::Identity(); // 真实位姿变化
    Eigen::Isometry3d node_pc_n = Eigen::Isometry3d::Identity();        // 测量位姿变化带噪声
    Eigen::Isometry3d pc_noise = Eigen::Isometry3d::Identity();         // 位姿噪声

    id = 0;
    while (!fin.eof()) {
        fin >> x >> y >> z >> roll >> pitch >> yaw;

        id++;  // 节点id
        auto temp = get_pose(x, y, z, roll, pitch, yaw);

        node_pose_now = Eigen::Isometry3d::Identity();

        node_pose_now.prerotate(temp.second);
        node_pose_now.pretranslate(temp.first);

        truth.push_back(node_pose_now);  // 添加机器人1的位置真值

        // 添加机器人1带噪声的位姿
        if (1 == id) {
            // 初始化 应该都为单位阵
            node_pose_before = node_pose_now;
            real.push_back(node_pose_now);  // 初始点不带噪声
        } else {
            // 本次主要的位姿变化，真实变化
            node_pose_change = Eigen::Isometry3d::Identity();
            node_pose_change = node_pose_now * (node_pose_before.inverse());
            // node_pose_change = node_pose_before * node_pose_now.inverse();

            // 噪声
            srand((int) time(NULL) + id);                     // 产生随机种子  把0换成NULL也行
            double noise_x = 0;
            double noise_y = 0;
            double noise_yaw = 0;
            if (node_pose_change.translation().x() != 0 || node_pose_change.translation().y() != 0) {
                // noise_x = random(-50, 50) * m;
                // noise_y = random(-50, 50) * m;
            }
            if (node_pose_change.rotation().eulerAngles(0, 1, 2).z() != 0) {
                noise_yaw = random(-50, 50) * n;
            }

            ROS_INFO_STREAM("noise set " << noise_x << " " << noise_y << " " << noise_yaw);
            auto temp_noise = get_pose(noise_x, noise_y, 0, 0, 0, noise_yaw);
            pc_noise = Eigen::Isometry3d::Identity();
            pc_noise.pretranslate(temp_noise.first);
            pc_noise.prerotate(temp_noise.second);
            noise.push_back(pc_noise);

            // 带有噪声的变化
//            ROS_INFO_STREAM("node_pc " << endl << node_pose_change.translation() << endl <<
//                                       node_pose_change.rotation().eulerAngles(0, 1, 2).z() / PI << endl);
            node_pc_n = pc_noise * node_pose_change;
            pc.push_back(node_pc_n);
//            ROS_INFO_STREAM("node_pc_n " << endl << node_pc_n.translation() << endl <<
//                                         node_pc_n.rotation().eulerAngles(0, 1, 2).z() / PI << endl);

            // 加噪声 累计误差模拟，带误差的变化
            node_pose_real = node_pc_n * node_pose_real;
            real.push_back(node_pose_real);

            node_pose_before = node_pose_now;
        }
    }
    fin.close();

    return true;
}


bool use_const_inf = false;
double const_stddev_x = 0.5;
double const_stddev_q = 0.1;

double var_gain_a = 20.0;

double min_stddev_x = 0.1;
double max_stddev_x = 5.0;
double min_stddev_q = 0.05;
double max_stddev_q = 0.2;
double fitness_score_thresh = 0.5;

/**
 *
 * @param a
 * @param max_x
 * @param min_y
 * @param max_y
 * @param x
 * @return
 */
double weight(double a, double max_x, double min_y, double max_y, double x) {
    double y = (1.0 - std::exp(-a * x)) / (1.0 - std::exp(-a * max_x));
    return min_y + (max_y - min_y) * y;
}


/**
 *
 * @param change
 * @return
 */
Eigen::MatrixXd calc_inf_matrix(Eigen::Isometry3d &noise) {
    if (use_const_inf) {
        Eigen::MatrixXd inf = Eigen::MatrixXd::Identity(6, 6);
        inf.topLeftCorner(3, 3).array() /= const_stddev_x;
        inf.bottomRightCorner(3, 3).array() /= const_stddev_q;
        return inf;
    }

    double fitness_score = 0.0f;

    auto tans = noise.translation();

    fitness_score = sqrt(pow(tans.x(), 2) + pow(tans.y(), 2));
    //abs(change.second.matrix().eulerAngles(0,1,2).z());

    double min_var_x = std::pow(min_stddev_x, 2);
    double max_var_x = std::pow(max_stddev_x, 2);
    double min_var_q = std::pow(min_stddev_q, 2);
    double max_var_q = std::pow(max_stddev_q, 2);

    double w_x = weight(var_gain_a, fitness_score_thresh, min_var_x, max_var_x, fitness_score);
    double w_q = weight(var_gain_a, fitness_score_thresh, min_var_q, max_var_q, fitness_score);

    Eigen::MatrixXd inf = Eigen::MatrixXd::Identity(6, 6);
    inf.topLeftCorner(3, 3).array() /= w_x;
    inf.bottomRightCorner(3, 3).array() /= w_q;
    return inf;
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
    std::vector<Eigen::Isometry3d> pose_truth_rb1;  // ground truth rb1
    std::vector<Eigen::Isometry3d> pose_truth_rb2;  // ground truth rb2

    std::vector<Eigen::Isometry3d> pose_real_rb1;  // 实际带噪声的位姿
    std::vector<Eigen::Isometry3d> pose_real_rb2;  // 实际带噪声的位姿

    std::vector<Eigen::Isometry3d> pose_change_n_rb1;  // 带有噪声的位姿变化
    std::vector<Eigen::Isometry3d> pose_change_n_rb2;  // 带有噪声的位姿变化

    std::vector<Eigen::Isometry3d> noise_rb1;      // 噪声 rb1
    std::vector<Eigen::Isometry3d> noise_rb2;      // 噪声 rb2

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
    // c rb1 3, 1.5, 0, 0, 0, 0
    // c rb2 4, 0, 0, 0, 0 ,1
    Eigen::Isometry3d rb2_c = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d rb1_c = Eigen::Isometry3d::Identity();
    auto rb1_pose = get_pose(3, 1.5, 0, 0, 0, 0);
    auto rb2_pose = get_pose(4, 0, 0, 0, 0, 1);
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
    if (!get_pose_data(data_path_rb1, pose_truth_rb1, pose_real_rb1, pose_change_n_rb1, noise_rb1, n_step_rb1, n_yaw_step_rb1)) {
        ROS_INFO_STREAM("DATA 1 READ ERROR");
        return -1;
    }
    if (!get_pose_data(data_path_rb2, pose_truth_rb2, pose_real_rb2, pose_change_n_rb2, noise_rb2, n_step_rb2, n_yaw_step_rb2)) {
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
        g2o::VertexSE3 *v = new g2o::VertexSE3();
        v->setId(id);
        v->setEstimate(p);

        optimizer.addVertex(v); // 优化器添加节点

        ROS_INFO_STREAM("add node, id: " << id);
        vertexCnt_rb1++;
        vectices.push_back(v);  // 节点存储

        // 设置边
        if (0 == id) {  // 以机器人一的坐标系为基准
            v->setFixed(true);
        } else {
            // SE3-SE3 边
            g2o::EdgeSE3 *e = new g2o::EdgeSE3();
            int idx1 = id - 1, idx2 = id;           // 关联的前后两个顶点
            edgeCnt_rb1++;
            e->setId(id);                              // 设置边的id
            e->setVertex(0, optimizer.vertices()[idx1]);  // 设置边的两端节点 vertices是map类型
            e->setVertex(1, optimizer.vertices()[idx2]);
            e->setMeasurement(pose_change_n_rb1[id - 1]);

            auto edge_inf = calc_inf_matrix(noise_rb1[id - 1]);

            e->setInformation(edge_inf);

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
        auto rb2_in_rb1 = rb2_2_rb1_trans * p; // rb2_2_rb1_trans

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
            g2o::EdgeSE3 *e = new g2o::EdgeSE3();
            int idx1 = id - 1, idx2 = id;           // 关联的前后两个顶点
            edgeCnt_rb2++;
            e->setId(id);                              // 设置边的id
            e->setVertex(0, optimizer.vertices()[idx1]);  // 设置边的两端节点
            e->setVertex(1, optimizer.vertices()[idx2]);
            e->setMeasurement(pose_change_n_rb2[id - rb2_node_start - 1]);

            auto edge_inf = calc_inf_matrix(noise_rb2[id - rb2_node_start - 1]);

            e->setInformation(edge_inf);

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

        auto temp_noise = Eigen::Isometry3d::Identity();
        auto edge_inf = calc_inf_matrix(temp_noise);

        e->setInformation(edge_inf);

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