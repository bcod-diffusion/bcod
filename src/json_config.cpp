#include "bcod/json_config.hpp"
#include "bcod/logging.hpp"
#include <fstream>
#include <stdexcept>

namespace bcod {

JsonConfig::JsonConfig(const std::string& config_path) {
    try {
        std::ifstream file(config_path);
        if (!file) {
            BCOD_ERROR("Failed to open config file: ", config_path);
            throw std::runtime_error("Failed to open config file");
        }
        
        file >> config_;
        validate_config();
        set_defaults();
    } catch (const nlohmann::json::exception& e) {
        BCOD_ERROR("Failed to parse config file: ", e.what());
        throw;
    }
}

void JsonConfig::save(const std::string& path) const {
    try {
        std::ofstream file(path);
        if (!file) {
            BCOD_ERROR("Failed to open output file: ", path);
            throw std::runtime_error("Failed to open output file");
        }
        
        file << std::setw(4) << config_ << std::endl;
    } catch (const nlohmann::json::exception& e) {
        BCOD_ERROR("Failed to write config file: ", e.what());
        throw;
    }
}

void JsonConfig::validate_config() const {
    const std::vector<std::string> required_keys = {
        "rasteriser.resolution",
        "rasteriser.max_range",
        "planner.batch_size",
        "planner.sequence_length",
        "scheduler.observation_dim",
        "scheduler.action_dim"
    };
    
    for (const auto& key : required_keys) {
        if (!config_.contains(key)) {
            BCOD_ERROR("Missing required config key: ", key);
            throw std::runtime_error("Missing required config key");
        }
    }
}

void JsonConfig::set_defaults() {
    if (!config_.contains("rasteriser.min_weight")) {
        config_["rasteriser.min_weight"] = 0.01;
    }
    
    if (!config_.contains("rasteriser.origin_x")) {
        config_["rasteriser.origin_x"] = 0.0;
    }
    
    if (!config_.contains("rasteriser.origin_y")) {
        config_["rasteriser.origin_y"] = 0.0;
    }
    
    if (!config_.contains("planner.max_velocity")) {
        config_["planner.max_velocity"] = 2.0;
    }
    
    if (!config_.contains("planner.max_angular_velocity")) {
        config_["planner.max_angular_velocity"] = 1.0;
    }
    
    if (!config_.contains("planner.risk_threshold")) {
        config_["planner.risk_threshold"] = 0.1;
    }
    
    if (!config_.contains("scheduler.temperature")) {
        config_["scheduler.temperature"] = 0.1;
    }
    
    if (!config_.contains("scheduler.lambda")) {
        config_["scheduler.lambda"] = 0.1;
    }
    
    if (!config_.contains("scheduler.target_risk")) {
        config_["scheduler.target_risk"] = 0.05;
    }
    
    if (!config_.contains("scheduler.learning_rate")) {
        config_["scheduler.learning_rate"] = 0.001;
    }
    
    if (!config_.contains("scheduler.discount_factor")) {
        config_["scheduler.discount_factor"] = 0.99;
    }
}

} // namespace bcod 