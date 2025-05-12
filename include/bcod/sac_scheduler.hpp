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

struct SchedulerState {
    cv::Mat belief_raster;     // Encoded belief state from B-COD
    double cvar_risk;          // Risk forecast from B-COD
    double goal_distance;      // Distance to goal
    std::vector<bool> prev_actions;  // Previous sensor mask
    int64_t timestamp;
    std::vector<double> sensor_powers;  // Current power consumption
    std::vector<double> sensor_uncertainties;  // Current uncertainties
    std::vector<double> environment_features;  // Environmental conditions
    std::vector<double> task_requirements;     // Task-specific requirements
};

struct SchedulerAction {
    std::vector<bool> sensor_mask;  // Binary sensor activation
    std::vector<double> probabilities;  // Raw probabilities before thresholding
    double total_power;  // Total power consumption
    double risk_violation;  // Risk constraint violation
    double energy_cost;  // Energy cost component
    double safety_cost;  // Safety cost component
    double total_cost;  // Combined cost
    int64_t timestamp;
};

struct SchedulerParams {
    // Network architecture
    int belief_dim;
    int hidden_dim;
    int num_layers;
    int num_heads;
    double dropout_rate;
    bool use_layer_norm;
    bool use_residual;
    bool use_attention;
    
    // Training parameters
    double learning_rate;
    double weight_decay;
    double temperature;
    double target_entropy;
    double tau;
    int batch_size;
    int buffer_size;
    int update_interval;
    int target_update_interval;
    int warmup_steps;
    int max_steps;
    bool use_mixed_precision;
    
    // Optimization parameters
    double risk_threshold;
    double violation_rate;
    double lambda_init;
    double lambda_lr;
    double lambda_min;
    double lambda_max;
    double energy_weight;
    double safety_weight;
    double goal_weight;
    
    // Sensor parameters
    std::vector<double> power_coefficients;
    std::vector<double> uncertainty_coefficients;
    std::vector<double> feature_weights;
    std::vector<double> risk_weights;
    std::vector<double> safety_weights;
    
    // Hardware parameters
    std::string device;
    int num_threads;
    int64_t memory_limit;
    bool use_tensor_cores;
    bool use_graph_optimization;
    bool use_dynamic_shapes;
    bool use_fp16;
    
    // Logging parameters
    std::string log_level;
    std::string log_file;
    bool use_tensorboard;
    bool use_wandb;
    std::vector<std::string> metrics;
};

class SACScheduler {
public:
    SACScheduler(const SchedulerParams& params);
    ~SACScheduler();

    SchedulerAction schedule(const SchedulerState& state);
    void update(const SchedulerState& state, const SchedulerAction& action, double reward, const SchedulerState& next_state);
    void load_model(const std::string& path);
    void save_model(const std::string& path);
    void set_params(const SchedulerParams& params);
    void set_device(const std::string& device);
    void set_batch_size(int batch_size);
    void set_risk_threshold(double threshold);
    void set_violation_rate(double rate);
    void set_lambda(double lambda);
    void set_energy_weight(double weight);
    void set_safety_weight(double weight);
    void set_goal_weight(double weight);
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