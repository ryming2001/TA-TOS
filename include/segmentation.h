#ifndef SEGMENTATION_H
#define SEGMENTATION_H

#include "utils.h"


pcl::PointCloud<pcl::PointXYZ>::Ptr getGroundHeight(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, std::vector<std::vector<pcl::PointIndices::Ptr>>& lattice_indices_stack) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr height_map(new pcl::PointCloud<pcl::PointXYZ>);
    height_map->resize(M * N);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double mid_x = XMIN + i * DOWNSIZE + DOWNSIZE / 2.0;
            double mid_y = YMIN + j * DOWNSIZE + DOWNSIZE / 2.0;

            height_map->points[i * N + j].x = mid_x;
            height_map->points[i * N + j].y = mid_y;
            pcl::CropBox<pcl::PointXYZ> crop_box;
            Eigen::Vector4f min_point(mid_x - DOWNSIZE / 2.0, mid_y - DOWNSIZE / 2.0, -H, 1.0);
            Eigen::Vector4f max_point(mid_x + DOWNSIZE / 2.0, mid_y + DOWNSIZE / 2.0, H, 1.0);
            crop_box.setMin(min_point);
            crop_box.setMax(max_point);
            crop_box.setInputCloud(cloud);

            std::vector<int> indices;
            crop_box.filter(indices);

            pcl::PointIndices::Ptr lattice_indices(new pcl::PointIndices());
            lattice_indices->indices = indices;

            lattice_indices_stack[i][j] = lattice_indices;

            if (indices.size() < 10) {
                height_map->points[i * N + j].z = std::numeric_limits<float>::quiet_NaN();
                continue;
            }

            // RANSAC
            pcl::ModelCoefficients::Ptr model(new pcl::ModelCoefficients);
            pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
            pcl::SACSegmentation<pcl::PointXYZ> seg;
            seg.setModelType(pcl::SACMODEL_PLANE);
            seg.setMethodType(pcl::SAC_RANSAC);
            seg.setDistanceThreshold(0.01);
            seg.setMaxIterations(100);

            seg.setInputCloud(cloud);
            seg.setIndices(lattice_indices);
            seg.segment(*inliers, *model);

            if (inliers->indices.size() > 0) {
                double a = model->values[0];
                double b = model->values[1];
                double c = model->values[2];
                double d = model->values[3];

                double z = -(a * mid_x + b * mid_y + d) / c;
                z = std::min(std::max(z, -H), H); // Clamp to [-H, H]

                height_map->points[i * N + j].z = z;
            }
            else height_map->points[i * N + j].z = std::numeric_limits<float>::quiet_NaN();
        }
    }

    height_map->width = N;
    height_map->height = M;
    height_map->is_dense = true;

    return height_map;
}

int iteration_count = 0;

double objective(const std::vector<double> &z_new, std::vector<double> &grad, void *data){
    OptimizationData *optData = static_cast<OptimizationData*>(data);
    std::vector<double> z = optData->z;

    // if (grad.size() != z_new.size()) {
    //     grad.resize(z_new.size());
    // }

    double obj = 0.0;
    for (size_t i = 0; i < z.size(); ++i) {
        if (!std::isnan(z[i])) {
            double res = std::abs(z_new[i] - z[i]);
            double gamma = width[i];
            obj -= std::exp(-res) * gamma;
            if (!grad.empty()) {
                grad[i] = -std::exp(-res) * gamma * (z[i] < z_new[i] ? -1 : 1); // 计算梯度
                iteration_count++;
            }
        }
    }
    return obj;
}


void constraint(unsigned m, double *result, unsigned n, const double* z_new, double* grad, void* c_data) {
    m = (M * (N - 2) + (M - 2) * N) * 2;
    int k = 0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            // horizontal constraints
            if (j != 0 && j != N - 1) {
                double z0 = z_new[i * N + j];
                double zl = z_new[i * N + (j - 1)];
                double zr = z_new[i * N + (j + 1)];
                double correlation = 2 * z0 - zl - zr;
                //ge
                result[k] = -CUR_MAX - correlation;
                k++;
                //le
                result[k] = -CUR_MAX + correlation;
                k++;
            }

            // vertical constraints
            if (i != 0 && i != M - 1) {
                double z0 = z_new[i * N + j];
                double zu = z_new[(i - 1) * N + j];
                double zp = z_new[(i + 1) * N + j];
                double correlation = 2 * z0 - zp - zu;
                //ge
                result[k] = -CUR_MAX - correlation;
                k++;
                //le
                result[k] = -CUR_MAX + correlation;
                k++;
            }
        }
    }
    if (k != m) {
        throw std::runtime_error("No Matching Constraint Size!");
    }


    if (grad) {
        std::memcpy(grad, data_cons.constraint_jacobian.data(), data_cons.constraint_jacobian.size() * sizeof(double));
    }
}

void refineGroundHeight(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, std::vector<double>& initial_guess) {
    // initialize OptimizationData
    iteration_count = 0;
    OptimizationData data;
    data.z.resize(M * N);
    std::transform(cloud->points.begin(), cloud->points.end(), data.z.begin(),
    [](const pcl::PointXYZ& point) {
        return point.z;
    });

    if (initial_guess.empty()) {
        std::vector<double> valid_values;
        std::copy_if(data.z.begin(), data.z.end(), std::back_inserter(valid_values),
                     [](double value) { return !std::isnan(value); });
        double z_mean = valid_values.empty() ? 0.0 : std::accumulate(valid_values.begin(), valid_values.end(), 0.0) / valid_values.size();
        initial_guess.resize(M * N, z_mean);
    }

    nlopt::opt optimizer(nlopt::LD_SLSQP, M * N);

    optimizer.set_min_objective(objective, &data);

    int constraint_num = static_cast<int>(data_cons.constraint_jacobian.size()/(M*N));
    std::vector<double> tol(constraint_num, 0.0);
    //std::vector<double> tol;
    optimizer.add_inequality_mconstraint(constraint, &data_cons, tol);

    optimizer.set_xtol_abs(X_TOL);
    optimizer.set_maxeval(1e5);

    std::vector<double> z_opt = initial_guess;
    double minf;
    nlopt::result result;
    result = optimizer.optimize(z_opt, minf);

    if (z_opt.size() != cloud->points.size()) {
        throw std::runtime_error("No Matching Size：z_opt and cloud do not have the same size！");
    }

    double sparse = 0;
    initial_guess = z_opt;
    for (size_t i = 0; i < cloud->points.size(); ++i) {
        sparse += std::abs(cloud->points[i].z - z_opt[i])<1e-2;
        cloud->points[i].z = z_opt[i];
    }
    double nonNan_num = countNonNan(data.z);
    sparse /= nonNan_num;
    //std::cout << "Optimization result: " << result << std::endl;
    //std::cout << "Iteration count: " << iteration_count << std::endl;
    std::cout << "Sparse Percentage: " << sparse << std::endl;
    printCurrentTime("MRF optimization success!");
}


void getSegmentation(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const std::vector<std::vector<pcl::PointIndices::Ptr>>& lattice_indices_stack, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_model, Segmentation& seg_pos, Segmentation& seg_neg, Segmentation& seg_road) {
    seg_pos.seg = 1;
    seg_neg.seg = -1;
    seg_road.seg = 0;

    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
    computeNormals(cloud_model, normals);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {

            pcl::PointIndices::Ptr lattice_indices(new pcl::PointIndices());
            lattice_indices =lattice_indices_stack[i][j];
            if (lattice_indices->indices.size() == 0) {
                continue;
            }
            Eigen::Vector3d n(normals->points[i * N + j].normal_x, normals->points[i * N + j].normal_y, normals->points[i * N + j].normal_z);
            n.normalize();
            if (n.z() < 0) {
                n = -n;
            }
            for (const int& index : lattice_indices->indices) {
                const pcl::PointXYZ& point = cloud->points[index];
                if(i == 0 || j == 0 || i == M - 1 || j == N - 1) {
                    if (point.x < XMIN + EDGE_LEN || point.x > XMAX - EDGE_LEN || point.y < YMIN + EDGE_WID || point.y > YMAX - EDGE_WID) {
                        continue;
                    }
                    //cropPointCloud(lattice_pcd, XMIN + EDGE_LEN, XMAX - EDGE_LEN, YMIN + EDGE_WID, YMAX - EDGE_WID, ZMIN, ZMAX);
                }
                Eigen::Vector3d v(point.x - cloud_model->points[i * N + j].x, point.y - cloud_model->points[i * N + j].y, point.z - cloud_model->points[i * N + j].z);
                double dist = v.dot(n);
                if(dist > DISTANCE) {
                    seg_pos.x.push_back(point.x);
                    seg_pos.y.push_back(point.y);
                    seg_pos.z.push_back(point.z);
                    seg_pos.dist.push_back(dist);
                    seg_pos.indices.push_back(index);
                }
                else if(dist < -DISTANCE_NEG) {
                    seg_neg.x.push_back(point.x);
                    seg_neg.y.push_back(point.y);
                    seg_neg.z.push_back(point.z);
                    seg_neg.dist.push_back(dist);
                    seg_neg.indices.push_back(index);
                }
                else {
                    seg_road.x.push_back(point.x);
                    seg_road.y.push_back(point.y);
                    seg_road.z.push_back(point.z);
                    seg_road.dist.push_back(dist);
                    seg_road.indices.push_back(index);
                }
            }
        }
    }
}

#endif //SEGMENTATION_H
