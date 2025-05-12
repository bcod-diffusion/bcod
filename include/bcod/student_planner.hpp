#pragma once

#include "belief_types.hpp"
#include "json_config.hpp"
#include <memory>
#include <string>
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <vector>
#include <array>
#include <cstdint>
#include <functional>
#include <deque>
#include <set>
#include <map>
#include <atomic>
#include <torch/torch.h>

namespace bcod {

struct Trajectory {
    std::vector<Eigen::Vector3d> waypoints;  // x, y, yaw increments
    std::vector<double> log_variances;
    double cvar_95;
    double max_variance;
    double mean_variance;
    std::vector<double> confidence;
    std::vector<double> risk_scores;
    int64_t timestamp;
    bool is_valid;
    double total_length;
    double max_curvature;
    double min_clearance;
    std::vector<double> feature_vector;
};

struct PlanningContext {
    cv::Mat belief_image;      // 5-channel belief raster
    cv::Mat semantic_map;      // Co-cropped map slice
    cv::Mat goal_mask;         // Binary goal mask
    std::vector<bool> active_sensors;
    Eigen::Vector3d current_pose;
    double max_velocity;
    double max_angular_velocity;
    double risk_threshold;
    int64_t timestamp;
    std::vector<double> sensor_uncertainties;
    std::vector<double> environment_features;
    std::vector<double> task_requirements;
    bool use_adaptive_horizon;
    int min_horizon;
    int max_horizon;
    double safety_margin;
    double energy_weight;
    double risk_weight;
    double goal_weight;
};

struct StudentParams {
    int input_channels;
    int hidden_dim;
    int num_layers;
    int num_heads;
    int trajectory_horizon;
    double dropout_rate;
    double learning_rate;
    double weight_decay;
    double kl_weight;
    double kl_ramp_steps;
    double min_kl_weight;
    double max_kl_weight;
    double cvar_percentile;
    double risk_threshold;
    double safety_margin;
    bool use_layer_norm;
    bool use_residual;
    bool use_attention;
    bool use_adaptive_horizon;
    std::string model_path;
    std::vector<int> encoder_channels;
    std::vector<int> decoder_channels;
    std::vector<int> attention_dims;
    std::vector<double> feature_weights;
    std::vector<double> risk_weights;
    std::vector<double> safety_weights;
};

class StudentPlanner {
public:
    StudentPlanner(const StudentParams& params);
    ~StudentPlanner();

    Trajectory plan(const PlanningContext& context);
    void load_model(const std::string& path);
    void save_model(const std::string& path);
    void set_params(const StudentParams& params);
    void set_device(const std::string& device);
    void set_batch_size(int batch_size);
    void set_sequence_length(int length);
    void set_risk_threshold(double threshold);
    void set_safety_margin(double margin);
    void set_energy_weight(double weight);
    void set_risk_weight(double weight);
    void set_goal_weight(double weight);
    void set_adaptive_horizon(bool use);
    void set_min_horizon(int horizon);
    void set_max_horizon(int horizon);
    void set_feature_weights(const std::vector<double>& weights);
    void set_risk_weights(const std::vector<double>& weights);
    void set_safety_weights(const std::vector<double>& weights);
    void set_debug(bool debug);
    void reset();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace bcod 