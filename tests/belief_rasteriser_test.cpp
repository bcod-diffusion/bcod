#include <gtest/gtest.h>
#include <bcod/belief_rasteriser.hpp>
#include <opencv2/opencv.hpp>

class BeliefRasteriserTest : public ::testing::Test {
protected:
    void SetUp() override {
        params = bcod::BeliefRasteriser::Params{
            .resolution = 0.1f,
            .window_size = 100,
            .num_particles = 1000,
            .max_uncertainty = 5.0f,
            .risk_threshold = 0.2f,
            .energy_threshold = 0.5f
        };
        rasteriser = std::make_unique<bcod::BeliefRasteriser>(params);
    }

    bcod::BeliefRasteriser::Params params;
    std::unique_ptr<bcod::BeliefRasteriser> rasteriser;
};

TEST_F(BeliefRasteriserTest, Initialization) {
    EXPECT_EQ(rasteriser->get_params().resolution, params.resolution);
    EXPECT_EQ(rasteriser->get_params().window_size, params.window_size);
}

TEST_F(BeliefRasteriserTest, Rasterization) {
    // Create test particles
    std::vector<bcod::Particle> particles;
    for (int i = 0; i < 100; ++i) {
        particles.push_back({
            .position = Eigen::Vector2f(i * 0.1f, i * 0.1f),
            .orientation = i * 0.1f,
            .weight = 1.0f / 100.0f
        });
    }

    // Test rasterization
    auto raster = rasteriser->rasterise(particles);
    EXPECT_EQ(raster.channels, 5);
    EXPECT_EQ(raster.rows, params.window_size);
    EXPECT_EQ(raster.cols, params.window_size);
}

TEST_F(BeliefRasteriserTest, WindowComputation) {
    std::vector<bcod::Particle> particles;
    for (int i = 0; i < 100; ++i) {
        particles.push_back({
            .position = Eigen::Vector2f(i * 0.1f, i * 0.1f),
            .orientation = i * 0.1f,
            .weight = 1.0f / 100.0f
        });
    }

    auto window = rasteriser->compute_raster_window(particles);
    EXPECT_GE(window.center.x(), 0.0f);
    EXPECT_GE(window.center.y(), 0.0f);
    EXPECT_GT(window.size, 0.0f);
}

TEST_F(BeliefRasteriserTest, StatisticsComputation) {
    std::vector<bcod::Particle> particles;
    for (int i = 0; i < 100; ++i) {
        particles.push_back({
            .position = Eigen::Vector2f(i * 0.1f, i * 0.1f),
            .orientation = i * 0.1f,
            .weight = 1.0f / 100.0f
        });
    }

    auto stats = rasteriser->compute_cell_statistics(particles);
    EXPECT_GE(stats.mean_position.x(), 0.0f);
    EXPECT_GE(stats.mean_position.y(), 0.0f);
    EXPECT_GE(stats.mean_orientation, 0.0f);
    EXPECT_GE(stats.position_covariance.determinant(), 0.0f);
    EXPECT_GE(stats.orientation_variance, 0.0f);
    EXPECT_GE(stats.risk_score, 0.0f);
    EXPECT_GE(stats.energy_cost, 0.0f);
} 