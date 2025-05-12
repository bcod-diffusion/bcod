#include <gtest/gtest.h>
#include <bcod/sac_scheduler.hpp>
#include <opencv2/opencv.hpp>

class SACSchedulerTest : public ::testing::Test {
protected:
    void SetUp() override {
        params = bcod::SACScheduler::SchedulerParams{
            .input_dim = 64,
            .hidden_dim = 256,
            .num_layers = 3,
            .dropout_rate = 0.1f,
            .layer_norm = true,
            .learning_rate = 0.0003f,
            .weight_decay = 0.0001f,
            .gradient_clip = 1.0f,
            .warmup_steps = 1000,
            .max_steps = 100000,
            .eval_interval = 1000,
            .save_interval = 5000,
            .mixed_precision = true,
            .num_workers = 4,
            .pin_memory = true,
            .temperature = 0.2f,
            .lambda = 0.5f,
            .target_risk = 0.2f,
            .discount_factor = 0.99f,
            .entropy_coef = 0.01f,
            .value_coef = 0.5f,
            .policy_coef = 1.0f,
            .max_grad_norm = 1.0f,
            .batch_size = 256,
            .buffer_size = 1000000,
            .update_interval = 1,
            .target_update_interval = 1000
        };
        scheduler = std::make_unique<bcod::SACScheduler>(params);
    }

    bcod::SACScheduler::SchedulerParams params;
    std::unique_ptr<bcod::SACScheduler> scheduler;
};

TEST_F(SACSchedulerTest, Initialization) {
    EXPECT_EQ(scheduler->get_params().input_dim, params.input_dim);
    EXPECT_EQ(scheduler->get_params().hidden_dim, params.hidden_dim);
    EXPECT_EQ(scheduler->get_params().num_layers, params.num_layers);
}

TEST_F(SACSchedulerTest, Scheduling) {
    // Create test state
    bcod::SchedulerState state;
    state.belief_raster = cv::Mat::zeros(100, 100, CV_32FC5);
    state.cvar_risk = 0.1f;
    state.goal_distance = 10.0f;
    state.prev_actions = std::vector<bool>(6, true);

    // Test scheduling
    auto action = scheduler->schedule(state);
    EXPECT_EQ(action.sensor_mask.size(), 6);
    EXPECT_GE(action.energy_cost, 0.0f);
    EXPECT_GE(action.risk_estimate, 0.0f);
}

TEST_F(SACSchedulerTest, ModelLoading) {
    // Test model loading
    EXPECT_NO_THROW(scheduler->load_model("models/sac_policy.pt"));
}

TEST_F(SACSchedulerTest, DeviceManagement) {
    // Test device management
    EXPECT_NO_THROW(scheduler->set_device("cuda:0"));
    EXPECT_NO_THROW(scheduler->set_batch_size(32));
} 