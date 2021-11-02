/**
 * @file test_algorithm.cpp
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
using Sophus::SE3d;
using Sophus::SO3d;

#define PI (3.1415926535897932346f)

typedef Matrix<double, 6, 6> Matrix6d;
typedef std::pair<Eigen::Vector3d, Eigen::Quaterniond> Pose;  // 位姿对
typedef std::pair<int, int> SpecialNodes;                     // 特殊节点对
typedef Matrix<double, 6, 1> Vector6d;                        // 李代数顶点
#define random(a, b) (rand() % (b - a) + a)

// 给定误差求J_R^{-1}的近似
Matrix6d JRInv(const SE3d &e) {
    Matrix6d J;
    J.block(0, 0, 3, 3) = SO3d::hat(e.so3().log());
    J.block(0, 3, 3, 3) = SO3d::hat(e.translation());
    J.block(3, 0, 3, 3) = Matrix3d::Zero(3, 3);
    J.block(3, 3, 3, 3) = SO3d::hat(e.so3().log());
    // J = J * 0.5 + Matrix6d::Identity();
    J = Matrix6d::Identity();  // try Identity if you want
    return J;
}

/**
 * @brief g2o顶点
 * 
 */
class VertexSE3LieAlgebra : public g2o::BaseVertex<6, SE3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    virtual bool read(istream &is) override {
        double data[7];
        for (int i = 0; i < 7; i++)
            is >> data[i];
        setEstimate(SE3d(
                Quaterniond(data[6], data[3], data[4], data[5]),
                Vector3d(data[0], data[1], data[2])));
    }

    virtual bool write(ostream &os) const override {
        os << id() << " ";
        Quaterniond q = _estimate.unit_quaternion();
        os << _estimate.translation().transpose() << " ";
        os << q.coeffs()[0] << " " << q.coeffs()[1] << " " << q.coeffs()[2] << " " << q.coeffs()[3] << endl;
        return true;
    }

    virtual void setToOriginImpl() override {
        _estimate = SE3d();
    }

    // 左乘更新
    virtual void oplusImpl(const double *update) override {
        Vector6d upd;
        upd << update[0], update[1], update[2], update[3], update[4], update[5];
        _estimate = SE3d::exp(upd) * _estimate;
    }

    void setSE3d(const Eigen::Quaterniond &qd, const Eigen::Vector3d &vd) {
        setEstimate(SE3d(qd, vd));
    }
};

/**
 * @brief g2o边
 * 
 */
class EdgeSE3LieAlgebra : public g2o::BaseBinaryEdge<6, SE3d, VertexSE3LieAlgebra, VertexSE3LieAlgebra> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    virtual bool read(istream &is) override {
        double data[7];
        for (int i = 0; i < 7; i++)
            is >> data[i];
        Quaterniond q(data[6], data[3], data[4], data[5]);
        q.normalize();
        setMeasurement(SE3d(q, Vector3d(data[0], data[1], data[2])));
        for (int i = 0; i < information().rows() && is.good(); i++)
            for (int j = i; j < information().cols() && is.good(); j++) {
                is >> information()(i, j);
                if (i != j)
                    information()(j, i) = information()(i, j);
            }
        return true;
    }

    bool set_edge(const Eigen::Quaterniond &qd, const Eigen::Vector3d &vd, const Matrix6d &inf) {
        setMeasurement(SE3d(qd, vd));
        this->setInformation(inf);
        return true;
    }

    virtual bool write(ostream &os) const override {
        VertexSE3LieAlgebra *v1 = static_cast<VertexSE3LieAlgebra *>(_vertices[0]);
        VertexSE3LieAlgebra *v2 = static_cast<VertexSE3LieAlgebra *>(_vertices[1]);
        os << v1->id() << " " << v2->id() << " ";
        SE3d m = _measurement;
        Eigen::Quaterniond q = m.unit_quaternion();
        os << m.translation().transpose() << " ";
        os << q.coeffs()[0] << " " << q.coeffs()[1] << " " << q.coeffs()[2] << " " << q.coeffs()[3] << " ";

        // information matrix
        for (int i = 0; i < information().rows(); i++)
            for (int j = i; j < information().cols(); j++) {
                os << information()(i, j) << " ";
            }
        os << endl;
        return true;
    }

    // 误差计算与书中推导一致
    virtual void computeError() override {
        SE3d v1 = (static_cast<VertexSE3LieAlgebra *>(_vertices[0]))->estimate();
        SE3d v2 = (static_cast<VertexSE3LieAlgebra *>(_vertices[1]))->estimate();
        _error = (_measurement.inverse() * v1.inverse() * v2).log();
    }

    // 雅可比计算
    virtual void linearizeOplus() override {
        SE3d v1 = (static_cast<VertexSE3LieAlgebra *>(_vertices[0]))->estimate();
        SE3d v2 = (static_cast<VertexSE3LieAlgebra *>(_vertices[1]))->estimate();
        Matrix6d J = JRInv(SE3d::exp(_error));
        // 尝试把J近似为I？
        _jacobianOplusXi = -J * v2.inverse().Adj();
        _jacobianOplusXj = J * v2.inverse().Adj();
    }
};

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
            srand((int) time(NULL));                     // 产生随机种子  把0换成NULL也行
            delta_x_n = delta_x + random(-50, 50) * m;  // -50 50 0.01 -> -0.5 0.5
            delta_y_n = delta_y + random(-50, 50) * m;
            delta_yaw_n = delta_yaw + random(-50, 50) * n;  // -50 50 0.005 -> -0.025 0.025
            auto temp_pc_n = get_pose(delta_x_n, delta_y_n, 0, 0, 0, delta_yaw_n);
            pc.push_back(temp_pc_n);

            // 加噪声 累计误差模拟，带误差的变化
            x_n = x_n_before + delta_x;  // -50 50 0.01 -> -0.5 0.5
            y_n = y_n_before + delta_y;
            yaw_n = yaw_n_before + delta_yaw;  // -50 50 0.005 -> -0.025 0.025
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
void create_marker_array(visualization_msgs::MarkerArray &markers, const std::vector<VertexSE3LieAlgebra *> &nodes,
                         const std::vector<EdgeSE3LieAlgebra *> &edges) {
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
//    visualization_msgs::Marker& edge_marker = markers.markers[1];
//    edge_marker.header.frame_id = "map";
//    edge_marker.header.stamp = ros::Time::now();
//    edge_marker.ns = "edges";
//    edge_marker.id = 2;
//    edge_marker.type = visualization_msgs::Marker::LINE_LIST;
//
//    edge_marker.pose.orientation.w = 1.0;
//    edge_marker.scale.x = 0.05;
//
//    edge_marker.points.resize(edges.size() * 2);
//    edge_marker.colors.resize(edges.size() * 2);
//
//    int i = 0;
//    for(auto & edge : edges) {
//        // g2o::VertexSE3* v1 = dynamic_cast<g2o::VertexSE3*>(edge->vertices()[0]);
//        // Eigen::Vector3d pt1 = v1->estimate().translation();
//        // Eigen::Vector3d pt2 = Eigen::Vector3d::Zero();
//        // pt2 = pt1 + edge->measurement().translation();
//        g2o::VertexSE3* v1 = dynamic_cast<g2o::VertexSE3*>(edge->vertices()[0]);
//        g2o::VertexSE3* v2 = dynamic_cast<g2o::VertexSE3*>(edge->vertices()[1]);
//        Eigen::Vector3d pt1 = v1->estimate().translation();
//        Eigen::Vector3d pt2 = v2->estimate().translation();
//
//        edge_marker.points[i * 2].x = pt1.x();
//        edge_marker.points[i * 2].y = pt1.y();
//        edge_marker.points[i * 2].z = pt1.z() + 0.5;
//        edge_marker.points[i * 2 + 1].x = pt2.x();
//        edge_marker.points[i * 2 + 1].y = pt2.y();
//        edge_marker.points[i * 2 + 1].z = pt2.z();
//
//        edge_marker.colors[i * 2].r = 1.0;
//        edge_marker.colors[i * 2].a = 1.0;
//        edge_marker.colors[i * 2 + 1].r = 1.0;
//        edge_marker.colors[i * 2 + 1].a = 1.0;
//        i++;
//    }

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

    std::vector<VertexSE3LieAlgebra *> vectices;  // 顶点容器
    std::vector<EdgeSE3LieAlgebra *> edges;       // 边容器

    std::vector<SpecialNodes> sp_nodes;  // 容器

    Eigen::Matrix<double, 6, 6> inf;  // 信息矩阵
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

    // test
//    Eigen::Isometry3d test = Eigen::Isometry3d::Identity();
//    auto test_pose = get_pose(0, 0, 0, 0, 0, 0);
//    test.prerotate(test_pose.second);
//    test.pretranslate(test_pose.first);
//    auto res = rb2_2_rb1_trans * test;
//
//    Eigen::Vector3d rb1_av = rb1_c.rotation().eulerAngles(2, 1, 0);
//    Eigen::Vector3d rb2_av = rb2_c.rotation().eulerAngles(2, 1, 0);
//    Eigen::Vector3d test_av = res.rotation().eulerAngles(2, 1, 0);
//
//    cout << rb1_av << endl;
//    cout << rb2_av << endl;
//    cout << res.translation() << endl;
//    cout << test_av << endl;

    // 噪声符合的分布
    const double n_step_rb1 = 0.001, n_yaw_step_rb1 = 0.0005;
    const double n_step_rb2 = 0.001, n_yaw_step_rb2 = 0.0005;

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

    ROS_INFO_STREAM("optimizing ...");

    // robot1 设置节点
    int vertexCnt_rb1 = 0, edgeCnt_rb1 = 0;  // 顶点和边的数量
    unsigned int id = rb1_node_start;
    for (auto &p: pose_real_rb1) {
        // 顶点
        VertexSE3LieAlgebra *v = new VertexSE3LieAlgebra();
        v->setId(id);
        v->setSE3d(p.second, p.first);
        optimizer.addVertex(v); // 优化器添加节点
        vertexCnt_rb1++;
        vectices.push_back(v);  // 节点存储

        // 设置边
        if (0 == id) {  // 以机器人一的坐标系为基准
            v->setFixed(true);
        } else {
            // SE3-SE3 边
            EdgeSE3LieAlgebra *e = new EdgeSE3LieAlgebra();
            int idx1 = id - 1, idx2 = id;           // 关联的前后两个顶点
            edgeCnt_rb1++;
            e->setId(id);                              // 设置边的id
            e->setVertex(0, optimizer.vertices()[idx1]);  // 设置边的两端节点
            e->setVertex(1, optimizer.vertices()[idx2]);
            e->set_edge(pose_change_n_rb1[id - 1].second, pose_change_n_rb1[id - 1].first, inf);
            edges.push_back(e);
        }
        id++;
    }


    // robot2 设置节点 和边
    int vertexCnt_rb2 = 0, edgeCnt_rb2 = 0;  // 顶点和边的数量
    id = rb2_node_start;
    for (auto &p: pose_real_rb2) {
        // 顶点
        VertexSE3LieAlgebra *v = new VertexSE3LieAlgebra();
        v->setId(id);

        // 坐标系转换
        Eigen::Isometry3d rb2_in_rb1 = Eigen::Isometry3d::Identity();
        rb2_in_rb1.prerotate(p.second);
        rb2_in_rb1.pretranslate(p.first);
        rb2_in_rb1 = rb2_2_rb1_trans * rb2_in_rb1; // rb2_2_rb1_trans

        cout << "rb2_2_rb1_trans: " << id << endl;
        cout << rb2_in_rb1.translation() << endl;
        cout << rb2_in_rb1.rotation().eulerAngles(0, 1, 2) << endl;

        // 设置节点
        v->setSE3d(Eigen::Quaterniond(rb2_in_rb1.rotation()), rb2_in_rb1.translation());
        optimizer.addVertex(v);
        vertexCnt_rb2++;
        vectices.push_back(v);

        // SE3-SE3 边 从10001开始
        if (10000 != id) {
            EdgeSE3LieAlgebra *e = new EdgeSE3LieAlgebra();
            int idx1 = id - 1, idx2 = id;           // 关联的前后两个顶点
            edgeCnt_rb2++;
            e->setId(id);                              // 设置边的id
            e->setVertex(0, optimizer.vertices()[idx1]);  // 设置边的两端节点
            e->setVertex(1, optimizer.vertices()[idx2]);
            e->set_edge(pose_change_n_rb2[id - 1 - rb2_node_start].second,
                        pose_change_n_rb2[id - 1 - rb2_node_start].first, inf);
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

        EdgeSE3LieAlgebra *e = new EdgeSE3LieAlgebra();
        e->setId(id);                                 // 设置边的id
        e->setVertex(0, optimizer.vertices()[idx1]);  // 设置边的两端节点
        e->setVertex(1, optimizer.vertices()[idx2]);
        e->set_edge(temp_pc.second, temp_pc.first, inf);
        edges.push_back(e);
        id++;
    }

    // display marker
    optimizer.initializeOptimization();
    optimizer.optimize(30);  // 优化三十次

    // show
    ROS_INFO_STREAM("robot 1 " << vertexCnt_rb1 << " vertices, " << edgeCnt_rb1 << " edges.");
    ROS_INFO_STREAM("robot 2 " << vertexCnt_rb2 << " vertices, " << edgeCnt_rb2 << " edges.");
    ROS_INFO_STREAM("optimization finish ...");


    // spin
    while (ros::ok()) {
        if (markers_pub.getNumSubscribers()) {
            create_marker_array(markers, vectices, edges);
            markers_pub.publish(markers);
        }
        ros::spinOnce();
    }

    return 0;
}