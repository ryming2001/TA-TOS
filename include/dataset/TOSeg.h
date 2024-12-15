#ifndef TOSEG_H
#define TOSEG_H

#define ROAD

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <boost/filesystem.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <regex>


#ifdef ROAD
std::string FOLDER = "../TOSeg-Road/";  // TOSeg-Road Dataset
std::string SAVE_FOLDER = "../results/";
// ROI setting
double XMIN = -6;
double XMAX = 20;
double YMIN = -7.5;
double YMAX = 8.5;
double ZMIN = -1000;
double ZMAX = 1000;
int EDGE_LEN = 0;
int EDGE_WID = 0;
// Segmentation Param Config
int DOWNSIZE = 2;
double DISTANCE = 0.05;
double DISTANCE_NEG = 0.05;
double NORMAL_ANGLE = 0;
double CUR_MAX = 0.05;
double G = 1;
double H = ZMAX;
int M = static_cast<int>((XMAX - XMIN) / DOWNSIZE);
int N = static_cast<int>((YMAX - YMIN) / DOWNSIZE);

// ERROR Tolerance Config
double X_TOL = 0.01;
#endif


struct Segmentation {
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> z;
    std::vector<double> dist;
    std::vector<int> indices;
    int seg;
};


class Metrics{
public:
    Metrics(): duration(-1) {}
    float duration;
};



bool customSort(const std::string& a, const std::string& b) {
    std::regex regex("([a-zA-Z0-9]+)-(\\d+)\\.pcd");
    std::smatch match_a, match_b;

    bool found_a = std::regex_search(a, match_a, regex);
    bool found_b = std::regex_search(b, match_b, regex);

    if (found_a && found_b) {
        std::string prefix_a = match_a[1].str();
        std::string prefix_b = match_b[1].str();

        int num_a = std::stoi(match_a[2].str());
        int num_b = std::stoi(match_b[2].str());

        if (prefix_a != prefix_b) {
            return prefix_a < prefix_b;
        } else {
            return num_a < num_b;
        }
    }

    return a < b;
}

void loadPcdFilesFromFolder(const std::string& folder_path, std::vector<std::string>& paths) {
    boost::filesystem::path dir(folder_path);
    if (!boost::filesystem::exists(dir) || !boost::filesystem::is_directory(dir)) {
        std::cerr << "Provided path is not a valid directory!" << std::endl;
        return;
    }

    boost::filesystem::directory_iterator end_iter;
    for (boost::filesystem::directory_iterator dir_iter(dir); dir_iter != end_iter; ++dir_iter) {
        if (boost::filesystem::is_directory(dir_iter->status())) {
            loadPcdFilesFromFolder(dir_iter->path().string(), paths);
        } else if (boost::filesystem::is_regular_file(dir_iter->status())) {
            std::string file_path = dir_iter->path().string();
            if (file_path.substr(file_path.find_last_of(".") + 1) == "pcd") {
                paths.push_back(file_path);
                std::cout << "Found .pcd file: " << file_path << std::endl;
            }
        }
    }

    std::sort(paths.begin(), paths.end(), customSort);
}


void preprocessPointCloud(std::string path, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    size_t pos = path.find(".pcd");
    if (pos != std::string::npos) {
        path.replace(pos, 4, ".txt");
    } else {
        std::cerr << "load file error" << std::endl;
        return;
    }

    Eigen::Matrix4f pose_matrix = Eigen::Matrix4f::Identity();

    std::ifstream file(path);
    if (!file) {
        std::cerr << "fail to load file: " << path << std::endl;
        return;
    }

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            file >> pose_matrix(i, j);
        }
    }

    file.close();

    Eigen::Affine3f transform(Eigen::Translation3f(pose_matrix.block<3, 1>(0, 3)) *
                              Eigen::Matrix3f(pose_matrix.block<3, 3>(0, 0)));

    pcl::transformPointCloud(*cloud, *cloud, transform);
}



void writeSegPCD(const std::string cloud_path, const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, std::vector<int>& indices_raw, std::vector<int>& indices_raw_neg, const Segmentation& seg_pos, const Segmentation& seg_neg, const Segmentation& seg_road) {
    std::vector<bool> if_repeated(cloud->points.size(), false);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_seg(new pcl::PointCloud<pcl::PointXYZRGB>);

    cloud_seg->resize(cloud->points.size());
    for (size_t i = 0; i < cloud->points.size(); ++i) {
        cloud_seg->points[i].x = cloud->points[i].x;
        cloud_seg->points[i].y = cloud->points[i].y;
        cloud_seg->points[i].z = cloud->points[i].z;
        cloud_seg->points[i].r = 237;
        cloud_seg->points[i].g = 29;
        cloud_seg->points[i].b = 33;
    }

    for (size_t k = 0; k < seg_road.indices.size(); ++k) {
        int index = indices_raw[seg_road.indices[k]];
        if (if_repeated[index]){
            continue;
        } else {
            if_repeated[index] = true;
        }

        cloud_seg->points[index].r = 150;
        cloud_seg->points[index].g = 255;
        cloud_seg->points[index].b = 10;
    }

    for (size_t k = 0; k < seg_neg.indices.size(); ++k) {
        int index = indices_raw[seg_neg.indices[k]];
        if (if_repeated[index]){
            continue;
        } else {
            if_repeated[index] = true;
        }
        cloud_seg->points[index].r = 68;
        cloud_seg->points[index].g = 114;
        cloud_seg->points[index].b = 255;
    }

    for (size_t k = 0; k < seg_pos.indices.size(); ++k) {
        int index = indices_raw[seg_pos.indices[k]];
        if (if_repeated[index]){
            continue;
        } else {
            if_repeated[index] = true;
        }
        cloud_seg->points[index].r = 237;
        cloud_seg->points[index].g = 29;
        cloud_seg->points[index].b = 33;
    }

    boost::filesystem::path p(cloud_path);
    std::string cloud_name = p.stem().string();

    size_t pos = cloud_name.find("raw");
    if (pos != std::string::npos) {
        cloud_name.replace(pos, 3, "seg");
    }

    std::string save_path = SAVE_FOLDER + cloud_name + ".pcd";

    std::cout << "save pcd:  " << save_path << std::endl << std::endl;
    pcl::io::savePCDFile(save_path, *cloud_seg);
}

void evalTime(const std::vector<Metrics>& result_stack) {
    Metrics results_all;

    float total_duration = 0.0f;
    for (const auto& result : result_stack) {
        total_duration += result.duration;
    }

    float duration_mean = result_stack.empty() ? -1.0f : total_duration / result_stack.size();
    results_all.duration = duration_mean;

    std::cout << "Average Duration: " << results_all.duration << " s" << std::endl;
}

#endif //TOSEG_H
