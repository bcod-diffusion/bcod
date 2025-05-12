#pragma once

#include <Eigen/Dense>
#include <array>
#include <vector>
#include <memory>

namespace bcod {

struct Particle {
    Eigen::Vector3d pose;
    Eigen::Matrix3d cov;
    double weight;
};

using ParticleSet = std::vector<Particle>;

struct BeliefRaster {
    static constexpr int WIDTH = 64;
    static constexpr int HEIGHT = 64;
    static constexpr int CHANNELS = 5;
    
    std::array<float, WIDTH * HEIGHT * CHANNELS> data;
    
    float& at(int x, int y, int c) {
        return data[(y * WIDTH + x) * CHANNELS + c];
    }
    
    const float& at(int x, int y, int c) const {
        return data[(y * WIDTH + x) * CHANNELS + c];
    }
};

struct Trajectory {
    std::vector<Eigen::Vector3d> waypoints;
    std::vector<Eigen::Matrix3d> covariances;
};

struct SensorMask {
    uint32_t active_sensors;
    
    bool is_active(int sensor_idx) const {
        return (active_sensors & (1 << sensor_idx)) != 0;
    }
    
    void set_active(int sensor_idx, bool active) {
        if (active) {
            active_sensors |= (1 << sensor_idx);
        } else {
            active_sensors &= ~(1 << sensor_idx);
        }
    }
};

struct PlanningContext {
    BeliefRaster belief;
    Eigen::Vector2d goal;
    SensorMask current_sensors;
    double time_horizon;
};

} // namespace bcod 