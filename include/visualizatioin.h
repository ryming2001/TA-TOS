#ifndef VISUALIZATIOIN_H
#define VISUALIZATIOIN_H

#include <pcl/visualization/pcl_visualizer.h>
#include "utils.h"


inline void visualizePointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    static pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(255, 255, 255);
    viewer->removeAllPointClouds();
    viewer->addPointCloud<pcl::PointXYZ>(cloud, "cloud input");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.7, 0.7, 0.7, "cloud input");
    viewer->spinOnce(100);
}

inline void visualizePointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud1,
                                  const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud2) {
    static pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(255, 255, 255);
    viewer->removeAllPointClouds();

    viewer->addPointCloud<pcl::PointXYZ>(cloud1, "cloud1");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.7, 0.7, 0.7, "cloud1"); // Red

    viewer->addPointCloud<pcl::PointXYZ>(cloud2, "cloud2");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "cloud2"); // Blue

    viewer->spinOnce(100);
}

inline void visualizePointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud1,
                                  const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud2,
                                  double r,
                                  double g,
                                  double b) {
    static pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(255, 255, 255);
    viewer->removeAllPointClouds();

    viewer->addPointCloud<pcl::PointXYZ>(cloud1, "cloud1");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.7, 0.7, 0.7, "cloud1"); // Red

    viewer->addPointCloud<pcl::PointXYZ>(cloud2, "cloud2");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, r, g, b, "cloud2"); // Blue

    viewer->spinOnce(100);
}


inline void visualizePointCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud) {
    static pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(255, 255, 255);
    viewer->removeAllPointClouds();
    viewer->addPointCloud<pcl::PointXYZRGB>(cloud, "cloud segmentation");
    viewer->spinOnce(10);
}

inline void visualizeSegmentation(const Segmentation& seg_pos, const Segmentation& seg_neg, const Segmentation& seg_road) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    for (size_t i = 0; i < seg_pos.x.size(); ++i) {
        pcl::PointXYZRGB point;
        point.x = seg_pos.x[i];
        point.y = seg_pos.y[i];
        point.z = seg_pos.z[i];
        point.r = RED[0];
        point.g = RED[1];
        point.b = RED[2];
        cloud->points.push_back(point);
    }

    for (size_t i = 0; i < seg_neg.x.size(); ++i) {
        pcl::PointXYZRGB point;
        point.x = seg_neg.x[i];
        point.y = seg_neg.y[i];
        point.z = seg_neg.z[i];
        point.r = BLUE[0];
        point.g = BLUE[1];
        point.b = BLUE[2];
        cloud->points.push_back(point);
    }

    for (size_t i = 0; i < seg_road.x.size(); ++i) {
        pcl::PointXYZRGB point;
        point.x = seg_road.x[i];
        point.y = seg_road.y[i];
        point.z = seg_road.z[i];
        point.r = GREEN[0];
        point.g = GREEN[1];
        point.b = GREEN[2];
        cloud->points.push_back(point);
    }

    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = true;

    visualizePointCloud(cloud);
}


#endif //VISUALIZATIOIN_H
