#include "bcod/rasteriser.hpp"
#include "bcod/student_planner.hpp"
#include "bcod/sac_scheduler.hpp"
#include "bcod/json_config.hpp"
#include "bcod/logging.hpp"
#include <opencv2/opencv.hpp>
#include <thread>
#include <chrono>
#include <fstream>
#include <iomanip>

using namespace bcod;

struct SimulatedSensor {
    SimulatedSensor(const std::string& name, double power, double warmup)
        : name(name), power(power), warmup(warmup) {}
        
    std::string name;
    double power;
    double warmup;
    bool is_active = false;
    std::chrono::steady_clock::time_point last_activation;
};

class SensorSimulator {
public:
    SensorSimulator() {
        sensors_.emplace_back("LIDAR", 45.0, 5.0);
        sensors_.emplace_back("RGB", 12.0, 1.0);
        sensors_.emplace_back("THERMAL", 8.0, 2.0);
        sensors_.emplace_back("GNSS", 2.5, 0.1);
        sensors_.emplace_back("IMU", 1.2, 0.05);
        sensors_.emplace_back("EXO2", 35.0, 3.0);
    }
    
    void update(const SensorMask& mask) {
        auto now = std::chrono::steady_clock::now();
        
        for (size_t i = 0; i < sensors_.size(); ++i) {
            bool should_be_active = mask.is_active(i);
            
            if (should_be_active && !sensors_[i].is_active) {
                sensors_[i].last_activation = now;
                sensors_[i].is_active = true;
            } else if (!should_be_active) {
                sensors_[i].is_active = false;
            }
        }
    }
    
    double get_total_power() const {
        double total = 0.0;
        auto now = std::chrono::steady_clock::now();
        
        for (const auto& sensor : sensors_) {
            if (sensor.is_active) {
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                    now - sensor.last_activation).count();
                if (elapsed >= sensor.warmup) {
                    total += sensor.power;
                }
            }
        }
        
        return total;
    }
    
    void log_status(std::ofstream& log) const {
        auto now = std::chrono::steady_clock::now();
        
        for (const auto& sensor : sensors_) {
            if (sensor.is_active) {
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                    now - sensor.last_activation).count();
                log << sensor.name << ": " 
                    << (elapsed >= sensor.warmup ? "ON" : "WARMING") 
                    << " (" << elapsed << "s)\n";
            } else {
                log << sensor.name << ": OFF\n";
            }
        }
        
        log << "Total power: " << get_total_power() << "W\n";
    }
    
private:
    std::vector<SimulatedSensor> sensors_;
};

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <config_file>" << std::endl;
        return 1;
    }
    
    try {
        JsonConfig config(argv[1]);
        
        BeliefRasteriser rasteriser(config);
        OneStepPlanner planner(config);
        SacScheduler scheduler(config);
        
        if (!planner.load_engine("assets/student_engine.plan")) {
            BCOD_FATAL("Failed to load planner engine");
            return 1;
        }
        
        if (!scheduler.load_policy("assets/sac_policy.onnx")) {
            BCOD_FATAL("Failed to load scheduler policy");
            return 1;
        }
        
        cv::Mat map = cv::imread("demo/lake_map.png", cv::IMREAD_GRAYSCALE);
        if (map.empty()) {
            BCOD_FATAL("Failed to load map");
            return 1;
        }
        
        std::ofstream log_file("demo/run_log.txt");
        if (!log_file) {
            BCOD_FATAL("Failed to open log file");
            return 1;
        }
        
        SensorSimulator sensor_sim;
        ParticleSet particles;
        
        for (int i = 0; i < 1000; ++i) {
            Particle p;
            p.pose = Eigen::Vector3d(
                std::rand() % map.cols,
                std::rand() % map.rows,
                std::rand() * 2.0 * M_PI / RAND_MAX
            );
            p.cov = Eigen::Matrix3d::Identity() * 0.1;
            p.weight = 1.0 / 1000.0;
            particles.push_back(p);
        }
        
        BeliefRaster raster;
        PlanningContext ctx;
        ctx.goal = Eigen::Vector2d(map.cols / 2, map.rows / 2);
        ctx.time_horizon = 5.0;
        
        for (int step = 0; step < 100; ++step) {
            rasteriser.generate(particles, raster);
            ctx.belief = raster;
            
            Trajectory traj;
            std::vector<double> variances;
            double risk;
            
            if (!planner.infer(ctx, traj, variances, risk)) {
                BCOD_ERROR("Planner inference failed");
                break;
            }
            
            std::vector<float> observation(128);
            for (size_t i = 0; i < observation.size(); ++i) {
                observation[i] = static_cast<float>(std::rand()) / RAND_MAX;
            }
            
            SensorMask sensor_mask;
            scheduler.step(observation, sensor_mask);
            sensor_sim.update(sensor_mask);
            
            log_file << "Step " << step << ":\n";
            log_file << "Risk: " << risk << "\n";
            log_file << "Variances: ";
            for (double v : variances) {
                log_file << v << " ";
            }
            log_file << "\n";
            sensor_sim.log_status(log_file);
            log_file << "\n";
            
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
    } catch (const std::exception& e) {
        BCOD_FATAL("Fatal error: ", e.what());
        return 1;
    }
    
    return 0;
} 