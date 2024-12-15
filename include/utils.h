#ifndef UTILS_H
#define UTILS_H

#include <fstream>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/common/common.h>
#include <pcl/common/distances.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/registration/correspondence_estimation.h>
#include <Eigen/Dense>
#include <nlopt.hpp>
#include <cmath>
#include <algorithm>
#include <vector>
#include <array>
#include <chrono>
#include "dataset/TOSeg.h"
Eigen::Matrix4d g_K_dym;


// Visualization
int RED[3] = {237, 29, 33};
int BLUE[3] = {68, 114, 255};
int GREEN[3] = {150, 255, 10};
int GREY[3] = {200, 200, 200};

int GE45[3] = {100, 0, 0};
int GE35[3] = {255, 0, 0};
int GE25[3] = {237, 125, 49};
int GE15[3] = {237, 125, 255};
int GE0[3] = {232, 195, 255};
int LE15[3] = {169, 209, 142};
int LE25[3] = {0, 176, 240};
int LE35[3] = {112, 48, 160};
int LE45[3] = {32, 56, 100};


struct OptimizationData {
    std::vector<double> z;
};

struct ConstraintData {
    std::vector<double> constraint_jacobian;
};

ConstraintData data_cons;
std::vector<double> width(M * N);

void softmin(std::vector<double>& width) {
    double sum_exp = 0.0;
    std::vector<double> w(width.size());

    for (size_t i = 0; i < width.size(); ++i) {
        w[i] = std::exp(-width[i]);
        sum_exp += w[i];
    }

    for (size_t i = 0; i < width.size(); ++i) {
        width[i] = width.size() * w[i] / sum_exp;
    }
}

void global_init() {
    data_cons.constraint_jacobian.clear();
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            // horizontal constraints
            if (j != 0 && j != N - 1) {
                std::vector<double> grad_ge(M * N, 0.0);
                std::vector<double> grad_le(M * N, 0.0);

                // constraint_h_ge
                grad_ge[i * N + j] = -2;
                grad_ge[i * N + (j - 1)] = 1;
                grad_ge[i * N + (j + 1)] = 1;
                data_cons.constraint_jacobian.insert(data_cons.constraint_jacobian.end(), grad_ge.begin(), grad_ge.end());

                // constraint_h_le
                grad_le[i * N + j] = 2;
                grad_le[i * N + (j - 1)] = -1;
                grad_le[i * N + (j + 1)] = -1;
                data_cons.constraint_jacobian.insert(data_cons.constraint_jacobian.end(), grad_le.begin(), grad_le.end());
            }

            // vertical constraints
            if (i != 0 && i != M - 1) {
                std::vector<double> grad_ge(M * N, 0.0);
                std::vector<double> grad_le(M * N, 0.0);

                // constraint_v_ge
                grad_ge[i * N + j] = -2;
                grad_ge[(i - 1) * N + j] = 1;
                grad_ge[(i + 1) * N + j] = 1;
                data_cons.constraint_jacobian.insert(data_cons.constraint_jacobian.end(), grad_ge.begin(), grad_ge.end());

                // constraint_v_le
                grad_le[i * N + j] = 2;
                grad_le[(i - 1) * N + j] = -1;
                grad_le[(i + 1) * N + j] = -1;
                data_cons.constraint_jacobian.insert(data_cons.constraint_jacobian.end(), grad_le.begin(), grad_le.end());
            }
        }
    }
    // initial width
    width.resize(M * N);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            width[i * N + j] = std::abs(j - (N / 2));
        }
    }
    softmin(width);
}


inline void cropPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, std::vector<int>& indices, std::vector<int>& indices_neg) {
    // 定义 CropBox
    pcl::CropBox<pcl::PointXYZ> crop_box;
    Eigen::Vector4f min_point(XMIN, YMIN, ZMIN, 1.0);
    Eigen::Vector4f max_point(XMAX, YMAX, ZMAX, 1.0);
    crop_box.setMin(min_point);
    crop_box.setMax(max_point);
    crop_box.setInputCloud(cloud);
    // 提取ROI
    crop_box.filter(indices);

    // 进行第二次反向过滤，保留不在 crop_box 区域内的点
    crop_box.setNegative(true); // 启用反向过滤
    crop_box.filter(indices_neg);

    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud);

    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
    inliers->indices = indices;
    extract.setIndices(inliers);
    extract.filter(*cloud);
}


inline double countNonNan(const std::vector<double>& vec) {
    return static_cast<double>(std::count_if(vec.begin(), vec.end(), [](double value) {
        return !std::isnan(value);
    }));
}

void computeNormals(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_model,
                    pcl::PointCloud<pcl::Normal>::Ptr& normals) {
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(cloud_model);

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    ne.setSearchMethod(tree);

    ne.setKSearch(10);

    ne.compute(*normals);
}


bool judgeROI(std::array<double, 2> position) {
    double x = position[0];
    double y = position[1];

    bool if_roi = (x > XMIN + EDGE_LEN) && (x < XMAX - EDGE_LEN) &&
                 (y > YMIN + EDGE_WID) && (y < YMAX - EDGE_WID);

    return if_roi;
}

void printCurrentTime(const std::string& info) {
    auto now = std::chrono::high_resolution_clock::now();

    auto duration = now.time_since_epoch();
    double seconds = std::chrono::duration_cast<std::chrono::duration<double>>(duration).count();

    std::cout << std::setw(40) << std::left << std::setfill(' ')<< info << " "
    << std::setw(15) << std::left << "System time: "
    << std::fixed << std::setprecision(3) << seconds << std::endl;
}

double recordCurrentTime(const std::string& info) {
    auto now = std::chrono::high_resolution_clock::now();

    auto duration = now.time_since_epoch();
    double seconds = std::chrono::duration_cast<std::chrono::duration<double>>(duration).count();

    std::cout << std::setw(40) << std::left << std::setfill(' ')<< info << " "
    << std::setw(15) << std::left << "System time: "
    << std::fixed << std::setprecision(3) << seconds << std::endl;
    return seconds;
}

#endif // UTILS_H

