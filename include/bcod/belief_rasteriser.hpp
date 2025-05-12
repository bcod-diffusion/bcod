#pragma once
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <vector>
#include <array>
#include <memory>
#include <cstdint>
#include <utility>
#include <string>
#include <functional>
#include <deque>
#include <set>
#include <map>
#include <atomic>

namespace bcod {

struct Particle {
    Eigen::Vector2d position;
    double yaw;
    double weight;
    std::array<double, 4> covariance;
    double confidence;
    int64_t timestamp;
    std::vector<double> features;
};

struct RasterWindow {
    Eigen::Vector2d center;
    double size;
    int grid_size;
    double scale;
    Eigen::Matrix2d axes;
    double max_eigen;
    double min_eigen;
    double angle;
};

struct RasterCell {
    double mass;
    double mean_sin;
    double mean_cos;
    double logdet_cov;
    double circ_var;
    int count;
    double max_weight;
    double min_weight;
    double sum_yaw;
    double sum_yaw2;
    double sum_x;
    double sum_y;
    double sum_x2;
    double sum_y2;
    double sum_xy;
};

struct BeliefRaster {
    cv::Mat data;
    RasterWindow window;
    std::vector<RasterCell> cells;
    int H, W, C;
    double normalization[5];
    std::vector<double> channel_min;
    std::vector<double> channel_max;
};

class BeliefRasteriser {
public:
    struct Params {
        int raster_H;
        int raster_W;
        int raster_C;
        double min_window;
        double max_window;
        double sigma_scale;
        double min_mass;
        double max_mass;
        double min_logdet;
        double max_logdet;
        double min_cvar;
        double max_cvar;
        double min_sin;
        double max_sin;
        double min_cos;
        double max_cos;
        double min_weight;
        double max_weight;
        double min_conf;
        double max_conf;
        int min_particles;
        int max_particles;
        double crop_margin;
        double resample_eps;
        int window_stride;
        int window_pad;
        bool align_axes;
        bool normalize;
        bool use_adaptive_window;
        std::vector<double> normalization;
    };

    BeliefRasteriser(const Params& params);
    ~BeliefRasteriser();

    BeliefRaster rasterise(const std::vector<Particle>& particles);
    RasterWindow compute_window(const std::vector<Particle>& particles) const;
    void fill_cells(const std::vector<Particle>& particles, RasterWindow& window, std::vector<RasterCell>& cells) const;
    void normalize_raster(BeliefRaster& raster) const;
    cv::Mat to_image(const BeliefRaster& raster) const;
    std::vector<double> compute_channel_stats(const BeliefRaster& raster, int channel) const;
    void resample_window(BeliefRaster& raster, int target_H, int target_W) const;
    void update_normalization(BeliefRaster& raster) const;
    void set_normalization(const std::vector<double>& norm);
    void set_window_params(double minw, double maxw, double sigmas);
    void set_alignment(bool align);
    void set_adaptive(bool adaptive);
    void set_crop_margin(double margin);
    void set_resample_eps(double eps);
    void set_window_stride(int stride);
    void set_window_pad(int pad);
    void set_particle_limits(int minp, int maxp);
    void set_channel_limits(const std::vector<double>& minv, const std::vector<double>& maxv);
    void set_confidence_limits(double minc, double maxc);
    void set_weight_limits(double minw, double maxw);
    void set_logdet_limits(double minl, double maxl);
    void set_cvar_limits(double minc, double maxc);
    void set_sin_limits(double mins, double maxs);
    void set_cos_limits(double minc, double maxc);
    void set_mass_limits(double minm, double maxm);
    void set_debug(bool debug);
    void reset();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace bcod 