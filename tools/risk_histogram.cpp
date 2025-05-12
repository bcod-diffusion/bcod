#include "bcod/student_planner.hpp"
#include "bcod/json_config.hpp"
#include "bcod/logging.hpp"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

using namespace bcod;

struct RiskData {
    double predicted_risk;
    double actual_error;
};

class RiskAnalyzer {
public:
    RiskAnalyzer(const std::string& log_path) {
        std::ifstream file(log_path);
        if (!file) {
            throw std::runtime_error("Failed to open log file");
        }
        
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            RiskData data;
            iss >> data.predicted_risk >> data.actual_error;
            data_.push_back(data);
        }
    }
    
    void compute_statistics() {
        std::vector<double> errors;
        for (const auto& data : data_) {
            errors.push_back(std::abs(data.predicted_risk - data.actual_error));
        }
        
        std::sort(errors.begin(), errors.end());
        
        mean_error_ = std::accumulate(errors.begin(), errors.end(), 0.0) / 
                     errors.size();
                     
        median_error_ = errors[errors.size() / 2];
        
        double sum_sq = 0.0;
        for (double e : errors) {
            sum_sq += (e - mean_error_) * (e - mean_error_);
        }
        std_error_ = std::sqrt(sum_sq / errors.size());
        
        max_error_ = errors.back();
    }
    
    void generate_histogram(const std::string& output_path) {
        const int num_bins = 50;
        std::vector<int> histogram(num_bins, 0);
        
        double min_risk = std::numeric_limits<double>::max();
        double max_risk = std::numeric_limits<double>::lowest();
        
        for (const auto& data : data_) {
            min_risk = std::min(min_risk, data.predicted_risk);
            max_risk = std::max(max_risk, data.predicted_risk);
        }
        
        double bin_width = (max_risk - min_risk) / num_bins;
        
        for (const auto& data : data_) {
            int bin = static_cast<int>((data.predicted_risk - min_risk) / 
                                     bin_width);
            if (bin >= 0 && bin < num_bins) {
                histogram[bin]++;
            }
        }
        
        int max_count = *std::max_element(histogram.begin(), histogram.end());
        
        cv::Mat vis(400, 800, CV_8UC3, cv::Scalar(255, 255, 255));
        
        for (int i = 0; i < num_bins; ++i) {
            int height = static_cast<int>(histogram[i] * 350.0 / max_count);
            cv::rectangle(vis,
                         cv::Point(i * 16, 399),
                         cv::Point((i + 1) * 16 - 1, 399 - height),
                         cv::Scalar(0, 0, 255),
                         -1);
        }
        
        std::string stats = "Mean: " + std::to_string(mean_error_) + 
                          "  Median: " + std::to_string(median_error_) +
                          "  Std: " + std::to_string(std_error_) +
                          "  Max: " + std::to_string(max_error_);
                          
        cv::putText(vis, stats, cv::Point(10, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
                   
        cv::imwrite(output_path, vis);
    }
    
private:
    std::vector<RiskData> data_;
    double mean_error_;
    double median_error_;
    double std_error_;
    double max_error_;
};

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <log_file> <output_image>" 
                  << std::endl;
        return 1;
    }
    
    try {
        RiskAnalyzer analyzer(argv[1]);
        analyzer.compute_statistics();
        analyzer.generate_histogram(argv[2]);
        
    } catch (const std::exception& e) {
        BCOD_FATAL("Fatal error: ", e.what());
        return 1;
    }
    
    return 0;
} 