#include "utils.h"
#include "segmentation.h"
#include "visualizatioin.h"
#include <iostream>
#include <cstdint>  // For uint32_t
#include <fstream>



void mainProcessingModule(const std::string cloud_path, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_raw, const std::vector<uint16_t>& label_gt, std::vector<Metrics>& result_stack, std::vector<double>& initial_guess, const int& frame_num) {
    /* #########################################################################
    1. reading frames
    ######################################################################### */
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_input(new pcl::PointCloud<pcl::PointXYZ>(*cloud_raw));
    //visualizePointCloud(cloud_raw);

    std::cout << "Processing frame #" << frame_num << std::endl;
    double start = recordCurrentTime("Begin!");  

    /* #########################################################################
    2. cropping & denoising
    ######################################################################### */
    std::vector<int> indices_raw, indices_raw_neg;
    cropPointCloud(cloud_raw, indices_raw, indices_raw_neg);

    //visualizePointCloud(cloud_raw);

    /* #########################################################################
    3. tiny obstacle segmentation
    ######################################################################### */
    printCurrentTime("Begin tiny obstacle segmentation!");
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_model(new pcl::PointCloud<pcl::PointXYZ>);
    std::vector<std::vector<pcl::PointIndices::Ptr>> lattice_indices_stack(M, std::vector<pcl::PointIndices::Ptr>(N));

    cloud_model = getGroundHeight(cloud_raw, lattice_indices_stack);
    printCurrentTime("Initialization success!"); 

    refineGroundHeight(cloud_model, initial_guess);

    Segmentation seg_pos, seg_neg, seg_road;
    getSegmentation(cloud_raw, lattice_indices_stack, cloud_model, seg_pos, seg_neg, seg_road);

    double end = recordCurrentTime("Success!");
    double duration = end - start;
    std::cout << "**********Processing duration**********     " << std::fixed << std::setprecision(3) << duration << std::endl;

    Metrics result;
    result.duration = duration;
    writeSegPCD(cloud_path, cloud_input, indices_raw, indices_raw_neg, seg_pos, seg_neg, seg_road);
    result_stack.push_back(result);
    //visualizeSegmentation(seg_pos, seg_neg, seg_road);
}


void eval_TOSeg() {
    // initialize
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_raw(new pcl::PointCloud<pcl::PointXYZ>());
    std::vector<uint16_t> label_gt;
    std::vector<Metrics> result_stack;
    std::vector<double> initial_guess;
    global_init();

    std::vector<std::string> file_paths;
    loadPcdFilesFromFolder(FOLDER, file_paths);
    for (int i = 0; i < file_paths.size(); ++i) {
        std::string cloud_path = file_paths[i];
        std::cout << "Reading cloud: " << cloud_path << std::endl;

        // read pcd
        if (pcl::io::loadPCDFile<pcl::PointXYZ>(cloud_path, *cloud_raw) == -1) {
            PCL_ERROR("Couldn't read the PCD file\n");
        }
        preprocessPointCloud(cloud_path ,cloud_raw);
        // processing
        mainProcessingModule(cloud_path, cloud_raw, label_gt, result_stack, initial_guess, i);
    }
    std::cout << "###############################################################" << std::endl;
    evalTime(result_stack);
    std::cout << "###############################################################" << std::endl;
}

int main(int argc, char** argv) {
    eval_TOSeg();
    return 0;
}
