#pragma once

#include <string>
#include <memory>
#include <nlohmann/json.hpp>

namespace bcod {

class JsonConfig {
public:
    explicit JsonConfig(const std::string& config_path);
    
    template<typename T>
    T get(const std::string& key, const T& default_value = T()) const {
        try {
            return config_.at(key).get<T>();
        } catch (const nlohmann::json::exception&) {
            return default_value;
        }
    }
    
    template<typename T>
    std::vector<T> get_array(const std::string& key) const {
        try {
            return config_.at(key).get<std::vector<T>>();
        } catch (const nlohmann::json::exception&) {
            return {};
        }
    }
    
    bool has(const std::string& key) const {
        return config_.contains(key);
    }
    
    void set(const std::string& key, const nlohmann::json& value) {
        config_[key] = value;
    }
    
    void save(const std::string& path) const;
    
private:
    nlohmann::json config_;
    
    void validate_config() const;
    void set_defaults();
};

} // namespace bcod 