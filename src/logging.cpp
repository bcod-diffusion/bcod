#include "bcod/logging.hpp"
#include <iostream>
#include <fstream>
#include <mutex>
#include <chrono>
#include <iomanip>

namespace bcod {

namespace {
    std::mutex log_mutex;
    std::ofstream log_file;
    bool use_colors = true;
    
    const char* get_color_code(LogLevel level) {
        if (!use_colors) return "";
        
        switch (level) {
            case LogLevel::DEBUG:   return "\033[36m";  // Cyan
            case LogLevel::INFO:    return "\033[32m";  // Green
            case LogLevel::WARNING: return "\033[33m";  // Yellow
            case LogLevel::ERROR:   return "\033[31m";  // Red
            case LogLevel::FATAL:   return "\033[35m";  // Magenta
            default:               return "";
        }
    }
    
    const char* get_reset_code() {
        return use_colors ? "\033[0m" : "";
    }
}

void Logger::write_log(LogLevel level, const std::string& message) {
    std::lock_guard<std::mutex> lock(log_mutex);
    
    const char* color = get_color_code(level);
    const char* reset = get_reset_code();
    
    std::cerr << color << message << reset << std::endl;
    
    if (log_file.is_open()) {
        log_file << message << std::endl;
    }
}

void Logger::set_log_file(const std::string& path) {
    std::lock_guard<std::mutex> lock(log_mutex);
    
    if (log_file.is_open()) {
        log_file.close();
    }
    
    log_file.open(path, std::ios::app);
    if (!log_file) {
        std::cerr << "Failed to open log file: " << path << std::endl;
    }
}

void Logger::set_use_colors(bool use) {
    std::lock_guard<std::mutex> lock(log_mutex);
    use_colors = use;
}

} // namespace bcod 