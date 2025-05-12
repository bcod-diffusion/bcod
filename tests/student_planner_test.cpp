#include <gtest/gtest.h>
#include <bcod/student_planner.hpp>
#include <opencv2/opencv.hpp>

class StudentPlannerTest : public ::testing::Test {
protected:
    void SetUp() override {
        params = bcod::StudentPlanner::StudentParams{
            .input_channels = 5,
            .hidden_dim = 256,
            .num_layers = 3,
            .dropout_rate = 0.1f,
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
        planner = std::make_unique<bcod::StudentPlanner>(params);
    }

    bcod::StudentPlanner::StudentParams params;
    std::unique_ptr<bcod::StudentPlanner> planner;
};

TEST_F(StudentPlannerTest, Initialization) {
    EXPECT_EQ(planner->get_params().input_channels, params.input_channels);
    EXPECT_EQ(planner->get_params().hidden_dim, params.hidden_dim);
    EXPECT_EQ(planner->get_params().num_layers, params.num_layers);
}

TEST_F(StudentPlannerTest, Planning) {
    // Create test context
    bcod::PlanningContext context;
    context.belief_image = cv::Mat::zeros(100, 100, CV_32FC5);
    context.semantic_map = cv::Mat::zeros(100, 100, CV_8UC1);
    context.current_pose = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
    context.goal_pose = Eigen::Vector3f(10.0f, 10.0f, 0.0f);
    context.risk_threshold = 0.2f;
    context.energy_threshold = 0.5f;
    context.max_planning_time = 0.1f;
    context.goal_mask = cv::Mat::zeros(100, 100, CV_8UC1);
    context.active_sensors = std::vector<bool>(6, true);

    // Test planning
    auto trajectory = planner->plan(context);
    EXPECT_FALSE(trajectory.waypoints.empty());
    EXPECT_FALSE(trajectory.log_variances.empty());
    EXPECT_FALSE(trajectory.confidence_scores.empty());
    EXPECT_FALSE(trajectory.risk_scores.empty());
    EXPECT_GE(trajectory.cvar_95, 0.0f);
    EXPECT_GE(trajectory.energy_cost, 0.0f);
}

TEST_F(StudentPlannerTest, ModelLoading) {
    // Test model loading
    EXPECT_NO_THROW(planner->load_model("models/student_policy.pt"));
}

TEST_F(StudentPlannerTest, DeviceManagement) {
    // Test device management
    EXPECT_NO_THROW(planner->set_device("cuda:0"));
    EXPECT_NO_THROW(planner->set_batch_size(32));
} 