#pragma once

#include <array>
#include <chrono>

namespace bcod {

enum class SensorType : uint8_t {
    LIDAR = 0,
    RGB = 1,
    THERMAL = 2,
    GNSS = 3,
    IMU = 4,
    EXO2 = 5,
    SENSOR_COUNT = 6
};

struct SensorConfig {
    static constexpr std::array<double, static_cast<size_t>(SensorType::SENSOR_COUNT)> POWER_CONSUMPTION = {
        45.0,  
        12.0,  
        8.0,   
        2.5,   
        1.2,   
        35.0  
    };
    
    static constexpr std::array<std::chrono::milliseconds, static_cast<size_t>(SensorType::SENSOR_COUNT)> WARMUP_TIME = {
        std::chrono::milliseconds(5000),   
        std::chrono::milliseconds(1000),   
        std::chrono::milliseconds(2000),   
        std::chrono::milliseconds(100),    
        std::chrono::milliseconds(50),     
        std::chrono::milliseconds(3000)    
    };
    
    static constexpr std::array<double, static_cast<size_t>(SensorType::SENSOR_COUNT)> MEASUREMENT_NOISE = {
        0.05,  
        0.10,  
        0.15,  
        0.02,  
        0.01,  
        0.08   
    };
};

} // namespace bcod 