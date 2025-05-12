#include <bcod/student_planner.hpp>
#include <bcod/utils.hpp>
#include <torch/torch.h>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <vector>
#include <array>
#include <memory>
#include <mutex>
#include <atomic>
#include <chrono>
#include <thread>
#include <queue>
#include <unordered_map>

namespace bcod {

struct StudentPlanner::Impl {
    struct StudentNetwork : torch::nn::Module {
        struct Encoder : torch::nn::Module {
            torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
            torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn3{nullptr};
            torch::nn::Linear fc1{nullptr}, fc2{nullptr};
            torch::nn::LayerNorm ln1{nullptr}, ln2{nullptr};
            torch::nn::Dropout dropout{nullptr};
            int hidden_dim;

            Encoder(int in_channels, int hidden_dim) : hidden_dim(hidden_dim) {
                conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, 64, 3).stride(1).padding(1)));
                bn1 = register_module("bn1", torch::nn::BatchNorm2d(64));
                conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).stride(2).padding(1)));
                bn2 = register_module("bn2", torch::nn::BatchNorm2d(128));
                conv3 = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, hidden_dim, 3).stride(2).padding(1)));
                bn3 = register_module("bn3", torch::nn::BatchNorm2d(hidden_dim));
                fc1 = register_module("fc1", torch::nn::Linear(hidden_dim * 16 * 16, hidden_dim));
                fc2 = register_module("fc2", torch::nn::Linear(hidden_dim, hidden_dim));
                ln1 = register_module("ln1", torch::nn::LayerNorm(hidden_dim));
                ln2 = register_module("ln2", torch::nn::LayerNorm(hidden_dim));
                dropout = register_module("dropout", torch::nn::Dropout(0.1));
            }

            torch::Tensor forward(torch::Tensor x) {
                x = torch::relu(bn1(conv1(x)));
                x = torch::relu(bn2(conv2(x)));
                x = torch::relu(bn3(conv3(x)));
                x = x.view({-1, hidden_dim * 16 * 16});
                x = torch::relu(ln1(fc1(x)));
                x = dropout(x);
                x = torch::relu(ln2(fc2(x)));
                return x;
            }
        };

        struct Decoder : torch::nn::Module {
            torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
            torch::nn::LayerNorm ln1{nullptr}, ln2{nullptr}, ln3{nullptr};
            torch::nn::Dropout dropout{nullptr};
            int hidden_dim, horizon;

            Decoder(int hidden_dim, int horizon) : hidden_dim(hidden_dim), horizon(horizon) {
                fc1 = register_module("fc1", torch::nn::Linear(hidden_dim, hidden_dim * 2));
                fc2 = register_module("fc2", torch::nn::Linear(hidden_dim * 2, hidden_dim * 4));
                fc3 = register_module("fc3", torch::nn::Linear(hidden_dim * 4, horizon * 6));  // 3 for mean, 3 for log_var
                ln1 = register_module("ln1", torch::nn::LayerNorm(hidden_dim * 2));
                ln2 = register_module("ln2", torch::nn::LayerNorm(hidden_dim * 4));
                ln3 = register_module("ln3", torch::nn::LayerNorm(horizon * 6));
                dropout = register_module("dropout", torch::nn::Dropout(0.1));
            }

            std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
                x = torch::relu(ln1(fc1(x)));
                x = dropout(x);
                x = torch::relu(ln2(fc2(x)));
                x = dropout(x);
                x = ln3(fc3(x));
                auto mean = x.slice(1, 0, -horizon * 3).view({-1, horizon, 3});
                auto log_var = x.slice(1, -horizon * 3).view({-1, horizon, 3});
                return {mean, log_var};
            }
        };

        Encoder encoder{nullptr};
        Decoder decoder{nullptr};
        torch::nn::Linear sensor_embedding{nullptr};
        torch::nn::MultiheadAttention attention{nullptr};
        int hidden_dim, horizon;

        StudentNetwork(const StudentParams& params) : hidden_dim(params.hidden_dim), horizon(params.trajectory_horizon) {
            encoder = register_module("encoder", Encoder(params.input_channels, hidden_dim));
            decoder = register_module("decoder", Decoder(hidden_dim, horizon));
            sensor_embedding = register_module("sensor_embedding", torch::nn::Linear(params.active_sensors.size(), hidden_dim));
            attention = register_module("attention", torch::nn::MultiheadAttention(hidden_dim, params.num_heads));
        }

        std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x, torch::Tensor sensor_mask) {
            auto features = encoder(x);
            auto sensor_features = sensor_embedding(sensor_mask);
            features = features + sensor_features;
            features = attention(features, features, features).output;
            return decoder(features);
        }
    };

    StudentParams params;
    std::unique_ptr<StudentNetwork> network;
    torch::Device device;
    std::mt19937 rng;
    std::mutex mtx;
    std::atomic<bool> debug;
    std::vector<double> feature_weights;
    std::vector<double> risk_weights;
    std::vector<double> safety_weights;
    double current_kl_weight;
    int64_t training_steps;

    Impl(const StudentParams& p) : params(p), device(torch::kCPU), rng(std::random_device{}()), debug(false), current_kl_weight(p.min_kl_weight), training_steps(0) {
        network = std::make_unique<StudentNetwork>(params);
        network->to(device);
    }

    Trajectory plan(const PlanningContext& context) {
        std::lock_guard<std::mutex> lock(mtx);
        network->eval();
        torch::NoGradGuard no_grad;

        auto belief_tensor = torch::from_blob(context.belief_image.data, {1, 5, 64, 64}, torch::kFloat32).to(device);
        auto map_tensor = torch::from_blob(context.semantic_map.data, {1, 3, 64, 64}, torch::kFloat32).to(device);
        auto goal_tensor = torch::from_blob(context.goal_mask.data, {1, 1, 64, 64}, torch::kFloat32).to(device);
        auto sensor_tensor = torch::zeros({1, context.active_sensors.size()}, torch::kFloat32).to(device);
        for (size_t i = 0; i < context.active_sensors.size(); ++i) {
            sensor_tensor[0][i] = context.active_sensors[i] ? 1.0f : 0.0f;
        }

        auto input = torch::cat({belief_tensor, map_tensor, goal_tensor}, 1);
        auto [mean, log_var] = network->forward(input, sensor_tensor);

        auto mean_cpu = mean.cpu();
        auto log_var_cpu = log_var.cpu();
        auto mean_data = mean_cpu.data_ptr<float>();
        auto log_var_data = log_var_cpu.data_ptr<float>();

        Trajectory traj;
        traj.waypoints.resize(params.trajectory_horizon);
        traj.log_variances.resize(params.trajectory_horizon);
        traj.confidence.resize(params.trajectory_horizon);
        traj.risk_scores.resize(params.trajectory_horizon);

        for (int i = 0; i < params.trajectory_horizon; ++i) {
            traj.waypoints[i] = Eigen::Vector3d(mean_data[i*3], mean_data[i*3+1], mean_data[i*3+2]);
            traj.log_variances[i] = log_var_data[i*3];
            traj.confidence[i] = std::exp(-log_var_data[i*3]);
            traj.risk_scores[i] = std::sqrt(std::exp(log_var_data[i*3]));
        }

        compute_trajectory_metrics(traj);
        return traj;
    }

    void compute_trajectory_metrics(Trajectory& traj) {
        std::vector<double> sorted_variances = traj.log_variances;
        std::sort(sorted_variances.begin(), sorted_variances.end());
        int cvar_idx = static_cast<int>(params.trajectory_horizon * (1.0 - params.cvar_percentile));
        traj.cvar_95 = 0.0;
        for (int i = cvar_idx; i < params.trajectory_horizon; ++i) {
            traj.cvar_95 += std::sqrt(std::exp(sorted_variances[i]));
        }
        traj.cvar_95 /= (params.trajectory_horizon - cvar_idx);

        traj.max_variance = std::sqrt(std::exp(*std::max_element(traj.log_variances.begin(), traj.log_variances.end())));
        traj.mean_variance = 0.0;
        for (double var : traj.log_variances) {
            traj.mean_variance += std::sqrt(std::exp(var));
        }
        traj.mean_variance /= params.trajectory_horizon;

        traj.total_length = 0.0;
        traj.max_curvature = 0.0;
        for (int i = 1; i < params.trajectory_horizon; ++i) {
            double dx = traj.waypoints[i].x() - traj.waypoints[i-1].x();
            double dy = traj.waypoints[i].y() - traj.waypoints[i-1].y();
            double ds = std::sqrt(dx*dx + dy*dy);
            traj.total_length += ds;

            if (i > 1) {
                double dx1 = traj.waypoints[i].x() - traj.waypoints[i-1].x();
                double dy1 = traj.waypoints[i].y() - traj.waypoints[i-1].y();
                double dx2 = traj.waypoints[i-1].x() - traj.waypoints[i-2].x();
                double dy2 = traj.waypoints[i-1].y() - traj.waypoints[i-2].y();
                double cross = dx1*dy2 - dy1*dx2;
                double dot = dx1*dx2 + dy1*dy2;
                double curvature = std::abs(cross) / (std::pow(dx1*dx1 + dy1*dy1, 1.5) + 1e-6);
                traj.max_curvature = std::max(traj.max_curvature, curvature);
            }
        }
    }

    void load_model(const std::string& path) {
        torch::load(network, path);
    }

    void save_model(const std::string& path) {
        torch::save(network, path);
    }

    void set_params(const StudentParams& p) { params = p; }
    void set_device(const std::string& dev) { device = torch::Device(dev); network->to(device); }
    void set_batch_size(int bs) { }
    void set_sequence_length(int len) { }
    void set_risk_threshold(double thresh) { params.risk_threshold = thresh; }
    void set_safety_margin(double margin) { params.safety_margin = margin; }
    void set_energy_weight(double weight) { params.energy_weight = weight; }
    void set_risk_weight(double weight) { params.risk_weight = weight; }
    void set_goal_weight(double weight) { params.goal_weight = weight; }
    void set_adaptive_horizon(bool use) { params.use_adaptive_horizon = use; }
    void set_min_horizon(int h) { params.min_horizon = h; }
    void set_max_horizon(int h) { params.max_horizon = h; }
    void set_feature_weights(const std::vector<double>& w) { feature_weights = w; }
    void set_risk_weights(const std::vector<double>& w) { risk_weights = w; }
    void set_safety_weights(const std::vector<double>& w) { safety_weights = w; }
    void set_debug(bool d) { debug = d; }
    void reset() { }
};

StudentPlanner::StudentPlanner(const StudentParams& params) : impl_(std::make_unique<Impl>(params)) {}
StudentPlanner::~StudentPlanner() = default;

Trajectory StudentPlanner::plan(const PlanningContext& context) { return impl_->plan(context); }
void StudentPlanner::load_model(const std::string& path) { impl_->load_model(path); }
void StudentPlanner::save_model(const std::string& path) { impl_->save_model(path); }
void StudentPlanner::set_params(const StudentParams& params) { impl_->set_params(params); }
void StudentPlanner::set_device(const std::string& device) { impl_->set_device(device); }
void StudentPlanner::set_batch_size(int batch_size) { impl_->set_batch_size(batch_size); }
void StudentPlanner::set_sequence_length(int length) { impl_->set_sequence_length(length); }
void StudentPlanner::set_risk_threshold(double threshold) { impl_->set_risk_threshold(threshold); }
void StudentPlanner::set_safety_margin(double margin) { impl_->set_safety_margin(margin); }
void StudentPlanner::set_energy_weight(double weight) { impl_->set_energy_weight(weight); }
void StudentPlanner::set_risk_weight(double weight) { impl_->set_risk_weight(weight); }
void StudentPlanner::set_goal_weight(double weight) { impl_->set_goal_weight(weight); }
void StudentPlanner::set_adaptive_horizon(bool use) { impl_->set_adaptive_horizon(use); }
void StudentPlanner::set_min_horizon(int horizon) { impl_->set_min_horizon(horizon); }
void StudentPlanner::set_max_horizon(int horizon) { impl_->set_max_horizon(horizon); }
void StudentPlanner::set_feature_weights(const std::vector<double>& weights) { impl_->set_feature_weights(weights); }
void StudentPlanner::set_risk_weights(const std::vector<double>& weights) { impl_->set_risk_weights(weights); }
void StudentPlanner::set_safety_weights(const std::vector<double>& weights) { impl_->set_safety_weights(weights); }
void StudentPlanner::set_debug(bool debug) { impl_->set_debug(debug); }
void StudentPlanner::reset() { impl_->reset(); }

} // namespace bcod 