#include "bcod/sensor_defs.hpp"
#include "bcod/logging.hpp"
#include <fstream>
#include <vector>
#include <map>
#include <chrono>
#include <iomanip>

using namespace bcod;

struct PowerEvent {
    std::chrono::system_clock::time_point timestamp;
    SensorType sensor;
    bool activated;
};

class PowerAnalyzer {
public:
    PowerAnalyzer(const std::string& log_path) {
        std::ifstream file(log_path);
        if (!file) {
            throw std::runtime_error("Failed to open log file");
        }
        
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string timestamp_str;
            iss >> timestamp_str;
            
            PowerEvent event;
            event.timestamp = parse_timestamp(timestamp_str);
            
            int sensor_idx;
            iss >> sensor_idx;
            event.sensor = static_cast<SensorType>(sensor_idx);
            
            std::string action;
            iss >> action;
            event.activated = (action == "ON");
            
            events_.push_back(event);
        }
    }
    
    void analyze() {
        std::map<SensorType, double> total_energy;
        std::map<SensorType, std::chrono::system_clock::duration> total_time;
        std::map<SensorType, int> activation_count;
        
        for (size_t i = 0; i < events_.size(); ++i) {
            const auto& event = events_[i];
            
            if (event.activated) {
                activation_count[event.sensor]++;
                
                if (i + 1 < events_.size()) {
                    auto duration = events_[i + 1].timestamp - event.timestamp;
                    total_time[event.sensor] += duration;
                    
                    double energy = SensorConfig::POWER_CONSUMPTION[
                        static_cast<size_t>(event.sensor)] *
                        std::chrono::duration<double>(duration).count() / 3600.0;
                    total_energy[event.sensor] += energy;
                }
            }
        }
        
        std::ofstream report("power_report.txt");
        report << std::fixed << std::setprecision(2);
        
        report << "Power Consumption Report\n";
        report << "=======================\n\n";
        
        double total_energy_wh = 0.0;
        for (int i = 0; i < static_cast<int>(SensorType::SENSOR_COUNT); ++i) {
            SensorType sensor = static_cast<SensorType>(i);
            report << get_sensor_name(sensor) << ":\n";
            report << "  Activations: " << activation_count[sensor] << "\n";
            report << "  Total time: " << 
                std::chrono::duration<double>(total_time[sensor]).count() << 
                " seconds\n";
            report << "  Energy: " << total_energy[sensor] << " Wh\n";
            report << "  Average power: " << 
                SensorConfig::POWER_CONSUMPTION[i] << " W\n\n";
                
            total_energy_wh += total_energy[sensor];
        }
        
        report << "Total energy consumption: " << total_energy_wh << " Wh\n";
    }
    
private:
    std::vector<PowerEvent> events_;
    
    std::chrono::system_clock::time_point parse_timestamp(
        const std::string& str) {
        std::tm tm = {};
        std::istringstream ss(str);
        ss >> std::get_time(&tm, "%Y-%m-%dT%H:%M:%S");
        return std::chrono::system_clock::from_time_t(std::mktime(&tm));
    }
    
    const char* get_sensor_name(SensorType sensor) {
        switch (sensor) {
            case SensorType::LIDAR:   return "LIDAR";
            case SensorType::RGB:     return "RGB Camera";
            case SensorType::THERMAL: return "Thermal Camera";
            case SensorType::GNSS:    return "GNSS";
            case SensorType::IMU:     return "IMU";
            case SensorType::EXO2:    return "EXO2";
            default:                  return "Unknown";
        }
    }
};

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <log_file>" << std::endl;
        return 1;
    }
    
    try {
        PowerAnalyzer analyzer(argv[1]);
        analyzer.analyze();
        
    } catch (const std::exception& e) {
        BCOD_FATAL("Fatal error: ", e.what());
        return 1;
    }
    
    return 0;
} 