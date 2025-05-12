#include <bcod/belief_rasteriser.hpp>
#include <Eigen/Eigenvalues>
#include <opencv2/imgproc.hpp>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <vector>
#include <array>
#include <memory>
#include <mutex>
#include <atomic>

namespace bcod {

struct BeliefRasteriser::Impl {
    Params params;
    std::vector<double> normalization;
    std::atomic<bool> debug;
    std::mutex mtx;
    Impl(const Params& p) : params(p), normalization(p.normalization), debug(false) {}

    RasterWindow compute_window(const std::vector<Particle>& particles) const {
        Eigen::Vector2d mean = Eigen::Vector2d::Zero();
        double total_w = 0;
        for (const auto& p : particles) {
            mean += p.position * p.weight;
            total_w += p.weight;
        }
        if (total_w > 0) mean /= total_w;
        Eigen::Matrix2d cov = Eigen::Matrix2d::Zero();
        for (const auto& p : particles) {
            Eigen::Vector2d d = p.position - mean;
            cov += p.weight * (d * d.transpose());
        }
        if (total_w > 0) cov /= total_w;
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eig(cov);
        double max_eig = std::sqrt(std::max(1e-8, eig.eigenvalues()(1)));
        double min_eig = std::sqrt(std::max(1e-8, eig.eigenvalues()(0)));
        double win_size = std::clamp(params.sigma_scale * max_eig, params.min_window, params.max_window);
        int grid_size = params.raster_H;
        double scale = win_size / grid_size;
        double angle = std::atan2(eig.eigenvectors()(1,0), eig.eigenvectors()(0,0));
        RasterWindow w{mean, win_size, grid_size, scale, eig.eigenvectors(), max_eig, min_eig, angle};
        return w;
    }

    void fill_cells(const std::vector<Particle>& particles, RasterWindow& window, std::vector<RasterCell>& cells) const {
        int H = window.grid_size, W = window.grid_size;
        cells.resize(H*W);
        for (auto& c : cells) {
            c.mass = 0; c.mean_sin = 0; c.mean_cos = 0; c.logdet_cov = 0; c.circ_var = 0; c.count = 0;
            c.max_weight = -1e9; c.min_weight = 1e9; c.sum_yaw = 0; c.sum_yaw2 = 0;
            c.sum_x = 0; c.sum_y = 0; c.sum_x2 = 0; c.sum_y2 = 0; c.sum_xy = 0;
        }
        for (const auto& p : particles) {
            Eigen::Vector2d rel = p.position - window.center;
            int u = static_cast<int>(std::floor((rel.x() + window.size/2) / window.scale));
            int v = static_cast<int>(std::floor((rel.y() + window.size/2) / window.scale));
            if (u < 0 || u >= W || v < 0 || v >= H) continue;
            int idx = v*W + u;
            auto& c = cells[idx];
            c.mass += p.weight;
            c.mean_sin += std::sin(p.yaw) * p.weight;
            c.mean_cos += std::cos(p.yaw) * p.weight;
            c.count++;
            c.max_weight = std::max(c.max_weight, p.weight);
            c.min_weight = std::min(c.min_weight, p.weight);
            c.sum_yaw += p.yaw * p.weight;
            c.sum_yaw2 += p.yaw * p.yaw * p.weight;
            c.sum_x += p.position.x() * p.weight;
            c.sum_y += p.position.y() * p.weight;
            c.sum_x2 += p.position.x()*p.position.x()*p.weight;
            c.sum_y2 += p.position.y()*p.position.y()*p.weight;
            c.sum_xy += p.position.x()*p.position.y()*p.weight;
        }
        for (auto& c : cells) {
            if (c.mass > 0) {
                c.mean_sin /= c.mass;
                c.mean_cos /= c.mass;
                double R = std::sqrt(c.mean_sin*c.mean_sin + c.mean_cos*c.mean_cos);
                c.circ_var = 1.0 - R;
            } else {
                c.mean_sin = 0; c.mean_cos = 0; c.circ_var = 1.0;
            }
            double sx2 = c.sum_x2/c.mass - std::pow(c.sum_x/c.mass,2);
            double sy2 = c.sum_y2/c.mass - std::pow(c.sum_y/c.mass,2);
            double sxy = c.sum_xy/c.mass - (c.sum_x/c.mass)*(c.sum_y/c.mass);
            double det = sx2*sy2 - sxy*sxy;
            c.logdet_cov = (c.mass > 0 && det > 1e-12) ? std::log(det) : 0.0;
        }
    }

    void normalize_raster(BeliefRaster& raster) const {
        int H = raster.H, W = raster.W, C = raster.C;
        for (int c = 0; c < C; ++c) {
            double minv = params.normalization.size() > c ? params.normalization[c] : 0.0;
            double maxv = params.normalization.size() > c ? params.normalization[c] : 1.0;
            for (int i = 0; i < H*W; ++i) {
                double v = raster.data.at<cv::Vec<float,5>>(i)[c];
                v = (v - minv) / (maxv - minv + 1e-8);
                v = std::clamp(v, 0.0, 1.0);
                raster.data.at<cv::Vec<float,5>>(i)[c] = static_cast<float>(v);
            }
        }
    }

    cv::Mat to_image(const BeliefRaster& raster) const {
        int H = raster.H, W = raster.W;
        cv::Mat img(H, W, CV_8UC3);
        for (int v = 0; v < H; ++v) {
            for (int u = 0; u < W; ++u) {
                int idx = v*W + u;
                float mass = raster.data.at<cv::Vec<float,5>>(v,W)[0];
                float hue = (std::atan2(raster.data.at<cv::Vec<float,5>>(v,W)[1], raster.data.at<cv::Vec<float,5>>(v,W)[2]) + M_PI) / (2*M_PI);
                float sat = 1.0f - raster.data.at<cv::Vec<float,5>>(v,W)[4];
                float val = mass;
                cv::Vec3b hsv(static_cast<uchar>(180*hue), static_cast<uchar>(255*sat), static_cast<uchar>(255*val));
                cv::Vec3b bgr;
                cv::cvtColor(cv::Mat(1,1,CV_8UC3,&hsv), cv::Mat(1,1,CV_8UC3,&bgr), cv::COLOR_HSV2BGR);
                img.at<cv::Vec3b>(v,u) = bgr;
            }
        }
        return img;
    }

    std::vector<double> compute_channel_stats(const BeliefRaster& raster, int channel) const {
        int H = raster.H, W = raster.W;
        std::vector<double> vals;
        for (int i = 0; i < H*W; ++i) {
            vals.push_back(raster.data.at<cv::Vec<float,5>>(i)[channel]);
        }
        std::sort(vals.begin(), vals.end());
        double mean = std::accumulate(vals.begin(), vals.end(), 0.0) / vals.size();
        double median = vals[vals.size()/2];
        double minv = vals.front();
        double maxv = vals.back();
        return {mean, median, minv, maxv};
    }

    void resample_window(BeliefRaster& raster, int target_H, int target_W) const {
        if (raster.H == target_H && raster.W == target_W) return;
        cv::Mat resized;
        cv::resize(raster.data, resized, cv::Size(target_W, target_H), 0, 0, cv::INTER_LINEAR);
        raster.data = resized;
        raster.H = target_H;
        raster.W = target_W;
    }

    void update_normalization(BeliefRaster& raster) const {
        int C = raster.C;
        for (int c = 0; c < C; ++c) {
            double minv = 1e9, maxv = -1e9;
            for (int i = 0; i < raster.H*raster.W; ++i) {
                double v = raster.data.at<cv::Vec<float,5>>(i)[c];
                minv = std::min(minv, v);
                maxv = std::max(maxv, v);
            }
            raster.channel_min.push_back(minv);
            raster.channel_max.push_back(maxv);
        }
    }

    void set_normalization(const std::vector<double>& norm) { normalization = norm; }
    void set_window_params(double minw, double maxw, double sigmas) { params.min_window = minw; params.max_window = maxw; params.sigma_scale = sigmas; }
    void set_alignment(bool align) { params.align_axes = align; }
    void set_adaptive(bool adaptive) { params.use_adaptive_window = adaptive; }
    void set_crop_margin(double margin) { params.crop_margin = margin; }
    void set_resample_eps(double eps) { params.resample_eps = eps; }
    void set_window_stride(int stride) { params.window_stride = stride; }
    void set_window_pad(int pad) { params.window_pad = pad; }
    void set_particle_limits(int minp, int maxp) { params.min_particles = minp; params.max_particles = maxp; }
    void set_channel_limits(const std::vector<double>& minv, const std::vector<double>& maxv) { }
    void set_confidence_limits(double minc, double maxc) { }
    void set_weight_limits(double minw, double maxw) { }
    void set_logdet_limits(double minl, double maxl) { }
    void set_cvar_limits(double minc, double maxc) { }
    void set_sin_limits(double mins, double maxs) { }
    void set_cos_limits(double minc, double maxc) { }
    void set_mass_limits(double minm, double maxm) { }
    void set_debug(bool d) { debug = d; }
    void reset() {}
};

BeliefRasteriser::BeliefRasteriser(const Params& params) : impl_(std::make_unique<Impl>(params)) {}
BeliefRasteriser::~BeliefRasteriser() = default;

BeliefRaster BeliefRasteriser::rasterise(const std::vector<Particle>& particles) {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    RasterWindow window = impl_->compute_window(particles);
    std::vector<RasterCell> cells;
    impl_->fill_cells(particles, window, cells);
    int H = window.grid_size, W = window.grid_size, C = 5;
    cv::Mat data(H, W, CV_32FC(C));
    for (int v = 0; v < H; ++v) {
        for (int u = 0; u < W; ++u) {
            int idx = v*W + u;
            const auto& c = cells[idx];
            data.at<cv::Vec<float,5>>(v,u)[0] = c.mass;
            data.at<cv::Vec<float,5>>(v,u)[1] = c.mean_sin;
            data.at<cv::Vec<float,5>>(v,u)[2] = c.mean_cos;
            data.at<cv::Vec<float,5>>(v,u)[3] = c.logdet_cov;
            data.at<cv::Vec<float,5>>(v,u)[4] = c.circ_var;
        }
    }
    BeliefRaster raster{data, window, cells, H, W, C, {0,0,0,0,0}, {}, {}};
    if (impl_->params.normalize) impl_->normalize_raster(raster);
    return raster;
}

RasterWindow BeliefRasteriser::compute_window(const std::vector<Particle>& particles) const {
    return impl_->compute_window(particles);
}

void BeliefRasteriser::fill_cells(const std::vector<Particle>& particles, RasterWindow& window, std::vector<RasterCell>& cells) const {
    impl_->fill_cells(particles, window, cells);
}

void BeliefRasteriser::normalize_raster(BeliefRaster& raster) const {
    impl_->normalize_raster(raster);
}

cv::Mat BeliefRasteriser::to_image(const BeliefRaster& raster) const {
    return impl_->to_image(raster);
}

std::vector<double> BeliefRasteriser::compute_channel_stats(const BeliefRaster& raster, int channel) const {
    return impl_->compute_channel_stats(raster, channel);
}

void BeliefRasteriser::resample_window(BeliefRaster& raster, int target_H, int target_W) const {
    impl_->resample_window(raster, target_H, target_W);
}

void BeliefRasteriser::update_normalization(BeliefRaster& raster) const {
    impl_->update_normalization(raster);
}

void BeliefRasteriser::set_normalization(const std::vector<double>& norm) { impl_->set_normalization(norm); }
void BeliefRasteriser::set_window_params(double minw, double maxw, double sigmas) { impl_->set_window_params(minw, maxw, sigmas); }
void BeliefRasteriser::set_alignment(bool align) { impl_->set_alignment(align); }
void BeliefRasteriser::set_adaptive(bool adaptive) { impl_->set_adaptive(adaptive); }
void BeliefRasteriser::set_crop_margin(double margin) { impl_->set_crop_margin(margin); }
void BeliefRasteriser::set_resample_eps(double eps) { impl_->set_resample_eps(eps); }
void BeliefRasteriser::set_window_stride(int stride) { impl_->set_window_stride(stride); }
void BeliefRasteriser::set_window_pad(int pad) { impl_->set_window_pad(pad); }
void BeliefRasteriser::set_particle_limits(int minp, int maxp) { impl_->set_particle_limits(minp, maxp); }
void BeliefRasteriser::set_channel_limits(const std::vector<double>& minv, const std::vector<double>& maxv) { impl_->set_channel_limits(minv, maxv); }
void BeliefRasteriser::set_confidence_limits(double minc, double maxc) { impl_->set_confidence_limits(minc, maxc); }
void BeliefRasteriser::set_weight_limits(double minw, double maxw) { impl_->set_weight_limits(minw, maxw); }
void BeliefRasteriser::set_logdet_limits(double minl, double maxl) { impl_->set_logdet_limits(minl, maxl); }
void BeliefRasteriser::set_cvar_limits(double minc, double maxc) { impl_->set_cvar_limits(minc, maxc); }
void BeliefRasteriser::set_sin_limits(double mins, double maxs) { impl_->set_sin_limits(mins, maxs); }
void BeliefRasteriser::set_cos_limits(double minc, double maxc) { impl_->set_cos_limits(minc, maxc); }
void BeliefRasteriser::set_mass_limits(double minm, double maxm) { impl_->set_mass_limits(minm, maxm); }
void BeliefRasteriser::set_debug(bool debug) { impl_->set_debug(debug); }
void BeliefRasteriser::reset() { impl_->reset(); }

} // namespace bcod 