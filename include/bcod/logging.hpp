#pragma once

#include <string>
#include <chrono>
#include <sstream>
#include <iomanip>

namespace bcod {

enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR,
    FATAL
};

class Logger {
public:
    static Logger& instance() {
        static Logger instance;
        return instance;
    }
    
    void set_level(LogLevel level) {
        current_level_ = level;
    }
    
    template<typename... Args>
    void log(LogLevel level, const char* file, int line, Args&&... args) {
        if (level < current_level_) return;
        
        std::stringstream ss;
        ss << get_timestamp() << " [" << get_level_str(level) << "] "
           << file << ":" << line << " - ";
        (ss << ... << std::forward<Args>(args));
        
        write_log(level, ss.str());
    }
    
private:
    Logger() = default;
    ~Logger() = default;
    
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    
    std::string get_timestamp() const {
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;
            
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S")
           << '.' << std::setfill('0') << std::setw(3) << ms.count();
        return ss.str();
    }
    
    const char* get_level_str(LogLevel level) const {
        switch (level) {
            case LogLevel::DEBUG:   return "DEBUG";
            case LogLevel::INFO:    return "INFO";
            case LogLevel::WARNING: return "WARNING";
            case LogLevel::ERROR:   return "ERROR";
            case LogLevel::FATAL:   return "FATAL";
            default:               return "UNKNOWN";
        }
    }
    
    void write_log(LogLevel level, const std::string& message);
    
    LogLevel current_level_ = LogLevel::INFO;
};

#define BCOD_LOG(level, ...) \
    bcod::Logger::instance().log(level, __FILE__, __LINE__, __VA_ARGS__)

#define BCOD_DEBUG(...) BCOD_LOG(bcod::LogLevel::DEBUG, __VA_ARGS__)
#define BCOD_INFO(...)  BCOD_LOG(bcod::LogLevel::INFO, __VA_ARGS__)
#define BCOD_WARN(...)  BCOD_LOG(bcod::LogLevel::WARNING, __VA_ARGS__)
#define BCOD_ERROR(...) BCOD_LOG(bcod::LogLevel::ERROR, __VA_ARGS__)
#define BCOD_FATAL(...) BCOD_LOG(bcod::LogLevel::FATAL, __VA_ARGS__)

} // namespace bcod 