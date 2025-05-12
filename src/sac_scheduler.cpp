#include <bcod/sac_scheduler.hpp>
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

struct SACScheduler::Impl {
    struct ActorNetwork : torch::nn::Module {
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

        struct PolicyHead : torch::nn::Module {
            torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
            torch::nn::LayerNorm ln1{nullptr}, ln2{nullptr}, ln3{nullptr};
            torch::nn::Dropout dropout{nullptr};
            int hidden_dim, num_sensors;

            PolicyHead(int hidden_dim, int num_sensors) : hidden_dim(hidden_dim), num_sensors(num_sensors) {
                fc1 = register_module("fc1", torch::nn::Linear(hidden_dim + 3, hidden_dim));  // +3 for risk, distance, prev_action
                fc2 = register_module("fc2", torch::nn::Linear(hidden_dim, hidden_dim));
                fc3 = register_module("fc3", torch::nn::Linear(hidden_dim, num_sensors));
                ln1 = register_module("ln1", torch::nn::LayerNorm(hidden_dim));
                ln2 = register_module("ln2", torch::nn::LayerNorm(hidden_dim));
                ln3 = register_module("ln3", torch::nn::LayerNorm(num_sensors));
                dropout = register_module("dropout", torch::nn::Dropout(0.1));
            }

            std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x, torch::Tensor context) {
                x = torch::cat({x, context}, 1);
                x = torch::relu(ln1(fc1(x)));
                x = dropout(x);
                x = torch::relu(ln2(fc2(x)));
                x = dropout(x);
                x = ln3(fc3(x));
                auto mean = torch::sigmoid(x);
                auto log_std = torch::ones_like(mean) * -2.0;  // Fixed log_std for simplicity
                return {mean, log_std};
            }
        };

        Encoder encoder{nullptr};
        PolicyHead policy_head{nullptr};
        int hidden_dim, num_sensors;

        ActorNetwork(const SchedulerParams& params) : hidden_dim(params.hidden_dim), num_sensors(params.power_coefficients.size()) {
            encoder = register_module("encoder", Encoder(5, hidden_dim));  // 5 channels from belief raster
            policy_head = register_module("policy_head", PolicyHead(hidden_dim, num_sensors));
        }

        std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x, torch::Tensor context) {
            auto features = encoder(x);
            return policy_head(features, context);
        }
    };

    struct CriticNetwork : torch::nn::Module {
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

        struct QHead : torch::nn::Module {
            torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
            torch::nn::LayerNorm ln1{nullptr}, ln2{nullptr}, ln3{nullptr};
            torch::nn::Dropout dropout{nullptr};
            int hidden_dim, num_sensors;

            QHead(int hidden_dim, int num_sensors) : hidden_dim(hidden_dim), num_sensors(num_sensors) {
                fc1 = register_module("fc1", torch::nn::Linear(hidden_dim + 3 + num_sensors, hidden_dim));
                fc2 = register_module("fc2", torch::nn::Linear(hidden_dim, hidden_dim));
                fc3 = register_module("fc3", torch::nn::Linear(hidden_dim, 1));
                ln1 = register_module("ln1", torch::nn::LayerNorm(hidden_dim));
                ln2 = register_module("ln2", torch::nn::LayerNorm(hidden_dim));
                ln3 = register_module("ln3", torch::nn::LayerNorm(1));
                dropout = register_module("dropout", torch::nn::Dropout(0.1));
            }

            torch::Tensor forward(torch::Tensor x, torch::Tensor context, torch::Tensor action) {
                x = torch::cat({x, context, action}, 1);
                x = torch::relu(ln1(fc1(x)));
                x = dropout(x);
                x = torch::relu(ln2(fc2(x)));
                x = dropout(x);
                x = ln3(fc3(x));
                return x;
            }
        };

        Encoder encoder{nullptr};
        QHead q_head{nullptr};
        int hidden_dim, num_sensors;

        CriticNetwork(const SchedulerParams& params) : hidden_dim(params.hidden_dim), num_sensors(params.power_coefficients.size()) {
            encoder = register_module("encoder", Encoder(5, hidden_dim));
            q_head = register_module("q_head", QHead(hidden_dim, num_sensors));
        }

        torch::Tensor forward(torch::Tensor x, torch::Tensor context, torch::Tensor action) {
            auto features = encoder(x);
            return q_head(features, context, action);
        }
    };

    struct ReplayBuffer {
        struct Transition {
            torch::Tensor state;
            torch::Tensor context;
            torch::Tensor action;
            double reward;
            torch::Tensor next_state;
            torch::Tensor next_context;
            bool done;
        };

        std::deque<Transition> buffer;
        int capacity;
        std::mt19937 rng;

        ReplayBuffer(int capacity) : capacity(capacity), rng(std::random_device{}()) {}

        void push(const Transition& transition) {
            if (buffer.size() >= capacity) {
                buffer.pop_front();
            }
            buffer.push_back(transition);
        }

        std::vector<Transition> sample(int batch_size) {
            std::uniform_int_distribution<int> dist(0, buffer.size() - 1);
            std::vector<Transition> batch;
            for (int i = 0; i < batch_size; ++i) {
                batch.push_back(buffer[dist(rng)]);
            }
            return batch;
        }

        int size() const { return buffer.size(); }
    };

    SchedulerParams params;
    std::unique_ptr<ActorNetwork> actor;
    std::unique_ptr<CriticNetwork> critic1;
    std::unique_ptr<CriticNetwork> critic2;
    std::unique_ptr<CriticNetwork> target_critic1;
    std::unique_ptr<CriticNetwork> target_critic2;
    torch::optim::Adam actor_optimizer{nullptr};
    torch::optim::Adam critic1_optimizer{nullptr};
    torch::optim::Adam critic2_optimizer{nullptr};
    torch::Device device;
    std::mt19937 rng;
    std::mutex mtx;
    std::atomic<bool> debug;
    ReplayBuffer replay_buffer;
    double lambda;
    int64_t training_steps;

    Impl(const SchedulerParams& p) : params(p), device(torch::kCPU), rng(std::random_device{}()), debug(false), 
        replay_buffer(p.buffer_size), lambda(p.lambda_init), training_steps(0) {
        actor = std::make_unique<ActorNetwork>(params);
        critic1 = std::make_unique<CriticNetwork>(params);
        critic2 = std::make_unique<CriticNetwork>(params);
        target_critic1 = std::make_unique<CriticNetwork>(params);
        target_critic2 = std::make_unique<CriticNetwork>(params);
        
        actor->to(device);
        critic1->to(device);
        critic2->to(device);
        target_critic1->to(device);
        target_critic2->to(device);
        
        actor_optimizer = torch::optim::Adam(actor->parameters(), torch::optim::AdamOptions(params.learning_rate));
        critic1_optimizer = torch::optim::Adam(critic1->parameters(), torch::optim::AdamOptions(params.learning_rate));
        critic2_optimizer = torch::optim::Adam(critic2->parameters(), torch::optim::AdamOptions(params.learning_rate));
        
        target_critic1->load_state_dict(critic1->state_dict());
        target_critic2->load_state_dict(critic2->state_dict());
    }

    SchedulerAction schedule(const SchedulerState& state) {
        std::lock_guard<std::mutex> lock(mtx);
        actor->eval();
        torch::NoGradGuard no_grad;

        auto belief_tensor = torch::from_blob(state.belief_raster.data, {1, 5, 64, 64}, torch::kFloat32).to(device);
        auto context_tensor = torch::zeros({1, 3}, torch::kFloat32).to(device);
        context_tensor[0][0] = state.cvar_risk;
        context_tensor[0][1] = state.goal_distance;
        for (size_t i = 0; i < state.prev_actions.size(); ++i) {
            context_tensor[0][2] = state.prev_actions[i] ? 1.0f : 0.0f;
        }

        auto [mean, log_std] = actor->forward(belief_tensor, context_tensor);
        auto action = mean + torch::randn_like(mean) * torch::exp(log_std);
        action = torch::sigmoid(action);

        auto action_cpu = action.cpu();
        auto action_data = action_cpu.data_ptr<float>();

        SchedulerAction scheduler_action;
        scheduler_action.sensor_mask.resize(params.power_coefficients.size());
        scheduler_action.probabilities.resize(params.power_coefficients.size());
        scheduler_action.total_power = 0.0;
        scheduler_action.risk_violation = 0.0;
        scheduler_action.energy_cost = 0.0;
        scheduler_action.safety_cost = 0.0;
        scheduler_action.total_cost = 0.0;
        scheduler_action.timestamp = state.timestamp;

        for (size_t i = 0; i < params.power_coefficients.size(); ++i) {
            scheduler_action.probabilities[i] = action_data[i];
            scheduler_action.sensor_mask[i] = action_data[i] > 0.5;
            scheduler_action.total_power += params.power_coefficients[i] * (scheduler_action.sensor_mask[i] ? 1.0 : 0.0);
        }

        scheduler_action.risk_violation = state.cvar_risk > params.risk_threshold ? 1.0 : 0.0;
        scheduler_action.energy_cost = -scheduler_action.total_power;
        scheduler_action.safety_cost = scheduler_action.risk_violation;
        scheduler_action.total_cost = params.energy_weight * scheduler_action.energy_cost + 
                                    params.safety_weight * scheduler_action.safety_cost;

        return scheduler_action;
    }

    void update(const SchedulerState& state, const SchedulerAction& action, double reward, const SchedulerState& next_state) {
        std::lock_guard<std::mutex> lock(mtx);
        
        auto belief_tensor = torch::from_blob(state.belief_raster.data, {1, 5, 64, 64}, torch::kFloat32).to(device);
        auto next_belief_tensor = torch::from_blob(next_state.belief_raster.data, {1, 5, 64, 64}, torch::kFloat32).to(device);
        
        auto context_tensor = torch::zeros({1, 3}, torch::kFloat32).to(device);
        context_tensor[0][0] = state.cvar_risk;
        context_tensor[0][1] = state.goal_distance;
        for (size_t i = 0; i < state.prev_actions.size(); ++i) {
            context_tensor[0][2] = state.prev_actions[i] ? 1.0f : 0.0f;
        }
        
        auto next_context_tensor = torch::zeros({1, 3}, torch::kFloat32).to(device);
        next_context_tensor[0][0] = next_state.cvar_risk;
        next_context_tensor[0][1] = next_state.goal_distance;
        for (size_t i = 0; i < next_state.prev_actions.size(); ++i) {
            next_context_tensor[0][2] = next_state.prev_actions[i] ? 1.0f : 0.0f;
        }
        
        auto action_tensor = torch::zeros({1, action.sensor_mask.size()}, torch::kFloat32).to(device);
        for (size_t i = 0; i < action.sensor_mask.size(); ++i) {
            action_tensor[0][i] = action.sensor_mask[i] ? 1.0f : 0.0f;
        }

        ReplayBuffer::Transition transition{
            belief_tensor, context_tensor, action_tensor, reward,
            next_belief_tensor, next_context_tensor, false
        };
        replay_buffer.push(transition);

        if (replay_buffer.size() < params.batch_size) {
            return;
        }

        auto batch = replay_buffer.sample(params.batch_size);
        update_critics(batch);
        update_actor(batch);
        update_lambda(batch);
        
        if (training_steps % params.target_update_interval == 0) {
            target_critic1->load_state_dict(critic1->state_dict());
            target_critic2->load_state_dict(critic2->state_dict());
        }
        
        training_steps++;
    }

    void update_critics(const std::vector<ReplayBuffer::Transition>& batch) {
        critic1->train();
        critic2->train();
        
        auto states = torch::cat(std::vector<torch::Tensor>(batch.size(), batch[0].state));
        auto contexts = torch::cat(std::vector<torch::Tensor>(batch.size(), batch[0].context));
        auto actions = torch::cat(std::vector<torch::Tensor>(batch.size(), batch[0].action));
        auto rewards = torch::zeros({batch.size(), 1}, torch::kFloat32).to(device);
        auto next_states = torch::cat(std::vector<torch::Tensor>(batch.size(), batch[0].next_state));
        auto next_contexts = torch::cat(std::vector<torch::Tensor>(batch.size(), batch[0].next_context));
        auto dones = torch::zeros({batch.size(), 1}, torch::kFloat32).to(device);
        
        for (size_t i = 0; i < batch.size(); ++i) {
            rewards[i][0] = batch[i].reward;
            dones[i][0] = batch[i].done ? 1.0f : 0.0f;
        }

        auto [next_actions, next_log_probs] = actor->forward(next_states, next_contexts);
        auto target_q1 = target_critic1->forward(next_states, next_contexts, next_actions);
        auto target_q2 = target_critic2->forward(next_states, next_contexts, next_actions);
        auto target_q = torch::min(target_q1, target_q2);
        auto target = rewards + (1.0 - dones) * params.tau * (target_q - params.temperature * next_log_probs);

        auto current_q1 = critic1->forward(states, contexts, actions);
        auto current_q2 = critic2->forward(states, contexts, actions);

        auto critic1_loss = torch::mse_loss(current_q1, target);
        auto critic2_loss = torch::mse_loss(current_q2, target);

        critic1_optimizer.zero_grad();
        critic1_loss.backward();
        critic1_optimizer.step();

        critic2_optimizer.zero_grad();
        critic2_loss.backward();
        critic2_optimizer.step();
    }

    void update_actor(const std::vector<ReplayBuffer::Transition>& batch) {
        actor->train();
        
        auto states = torch::cat(std::vector<torch::Tensor>(batch.size(), batch[0].state));
        auto contexts = torch::cat(std::vector<torch::Tensor>(batch.size(), batch[0].context));
        
        auto [actions, log_probs] = actor->forward(states, contexts);
        auto q1 = critic1->forward(states, contexts, actions);
        auto q2 = critic2->forward(states, contexts, actions);
        auto q = torch::min(q1, q2);
        
        auto actor_loss = (params.temperature * log_probs - q).mean();
        
        actor_optimizer.zero_grad();
        actor_loss.backward();
        actor_optimizer.step();
    }

    void update_lambda(const std::vector<ReplayBuffer::Transition>& batch) {
        auto states = torch::cat(std::vector<torch::Tensor>(batch.size(), batch[0].state));
        auto contexts = torch::cat(std::vector<torch::Tensor>(batch.size(), batch[0].context));
        
        auto [actions, _] = actor->forward(states, contexts);
        auto risk_violations = torch::zeros({batch.size(), 1}, torch::kFloat32).to(device);
        
        for (size_t i = 0; i < batch.size(); ++i) {
            risk_violations[i][0] = batch[i].context[0][0] > params.risk_threshold ? 1.0f : 0.0f;
        }
        
        auto constraint_violation = risk_violations.mean() - params.violation_rate;
        lambda = std::max(params.lambda_min, std::min(params.lambda_max, lambda + params.lambda_lr * constraint_violation.item<float>()));
    }

    void load_model(const std::string& path) {
        torch::load(actor, path + "_actor.pt");
        torch::load(critic1, path + "_critic1.pt");
        torch::load(critic2, path + "_critic2.pt");
        target_critic1->load_state_dict(critic1->state_dict());
        target_critic2->load_state_dict(critic2->state_dict());
    }

    void save_model(const std::string& path) {
        torch::save(actor, path + "_actor.pt");
        torch::save(critic1, path + "_critic1.pt");
        torch::save(critic2, path + "_critic2.pt");
    }

    void set_params(const SchedulerParams& p) { params = p; }
    void set_device(const std::string& dev) { device = torch::Device(dev); 
        actor->to(device); critic1->to(device); critic2->to(device);
        target_critic1->to(device); target_critic2->to(device); }
    void set_batch_size(int bs) { }
    void set_risk_threshold(double thresh) { params.risk_threshold = thresh; }
    void set_violation_rate(double rate) { params.violation_rate = rate; }
    void set_lambda(double l) { lambda = l; }
    void set_energy_weight(double weight) { params.energy_weight = weight; }
    void set_safety_weight(double weight) { params.safety_weight = weight; }
    void set_goal_weight(double weight) { params.goal_weight = weight; }
    void set_feature_weights(const std::vector<double>& w) { params.feature_weights = w; }
    void set_risk_weights(const std::vector<double>& w) { params.risk_weights = w; }
    void set_safety_weights(const std::vector<double>& w) { params.safety_weights = w; }
    void set_debug(bool d) { debug = d; }
    void reset() { }
};

SACScheduler::SACScheduler(const SchedulerParams& params) : impl_(std::make_unique<Impl>(params)) {}
SACScheduler::~SACScheduler() = default;

SchedulerAction SACScheduler::schedule(const SchedulerState& state) { return impl_->schedule(state); }
void SACScheduler::update(const SchedulerState& state, const SchedulerAction& action, double reward, const SchedulerState& next_state) { 
    impl_->update(state, action, reward, next_state); }
void SACScheduler::load_model(const std::string& path) { impl_->load_model(path); }
void SACScheduler::save_model(const std::string& path) { impl_->save_model(path); }
void SACScheduler::set_params(const SchedulerParams& params) { impl_->set_params(params); }
void SACScheduler::set_device(const std::string& device) { impl_->set_device(device); }
void SACScheduler::set_batch_size(int batch_size) { impl_->set_batch_size(batch_size); }
void SACScheduler::set_risk_threshold(double threshold) { impl_->set_risk_threshold(threshold); }
void SACScheduler::set_violation_rate(double rate) { impl_->set_violation_rate(rate); }
void SACScheduler::set_lambda(double lambda) { impl_->set_lambda(lambda); }
void SACScheduler::set_energy_weight(double weight) { impl_->set_energy_weight(weight); }
void SACScheduler::set_safety_weight(double weight) { impl_->set_safety_weight(weight); }
void SACScheduler::set_goal_weight(double weight) { impl_->set_goal_weight(weight); }
void SACScheduler::set_feature_weights(const std::vector<double>& weights) { impl_->set_feature_weights(weights); }
void SACScheduler::set_risk_weights(const std::vector<double>& weights) { impl_->set_risk_weights(weights); }
void SACScheduler::set_safety_weights(const std::vector<double>& weights) { impl_->set_safety_weights(weights); }
void SACScheduler::set_debug(bool debug) { impl_->set_debug(debug); }
void SACScheduler::reset() { impl_->reset(); }

} // namespace bcod 