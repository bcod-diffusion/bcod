#include "bcod/rasteriser.hpp"
#include "bcod/logging.hpp"
#include <cmath>
#include <algorithm>

namespace bcod {

struct BeliefRasteriser::Impl {
    Impl(const JsonConfig& config) {
        resolution_ = config.get<double>("rasteriser.resolution", 0.1);
        max_range_ = config.get<double>("rasteriser.max_range", 10.0);
        min_weight_ = config.get<double>("rasteriser.min_weight", 0.01);
        
        origin_ = Eigen::Vector2d(
            config.get<double>("rasteriser.origin_x", 0.0),
            config.get<double>("rasteriser.origin_y", 0.0)
        );
    }
    
    double resolution_;
    double max_range_;
    double min_weight_;
    Eigen::Vector2d origin_;
};

BeliefRasteriser::BeliefRasteriser(const JsonConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

BeliefRasteriser::~BeliefRasteriser() = default;

void BeliefRasteriser::generate(const ParticleSet& particles, BeliefRaster& raster) const {
    std::fill(raster.data.begin(), raster.data.end(), 0.0f);
    
    for (const auto& particle : particles) {
        if (particle.weight < pimpl_->min_weight_) continue;
        
        Eigen::Vector2d pos = particle.pose.head<2>();
        Eigen::Vector2d cell_pos = (pos - pimpl_->origin_) / pimpl_->resolution_;
        
        int cell_x = static_cast<int>(std::floor(cell_pos.x()));
        int cell_y = static_cast<int>(std::floor(cell_pos.y()));
        
        if (cell_x < 0 || cell_x >= BeliefRaster::WIDTH ||
            cell_y < 0 || cell_y >= BeliefRaster::HEIGHT) {
            continue;
        }
        
        float mass, orientation;
        Eigen::Matrix2f cov;
        compute_cell_stats(particles, cell_x, cell_y, mass, orientation, cov);
        
        raster.at(cell_x, cell_y, 0) = mass;
        raster.at(cell_x, cell_y, 1) = orientation;
        raster.at(cell_x, cell_y, 2) = cov(0, 0);
        raster.at(cell_x, cell_y, 3) = cov(0, 1);
        raster.at(cell_x, cell_y, 4) = cov(1, 1);
    }
    
    normalize_raster(raster);
}

void BeliefRasteriser::compute_cell_stats(const ParticleSet& particles,
                                        int cell_x, int cell_y,
                                        float& mass, float& orientation,
                                        Eigen::Matrix2f& cov) const {
    mass = 0.0f;
    orientation = 0.0f;
    cov.setZero();
    
    Eigen::Vector2d cell_center = pimpl_->origin_ + 
        Eigen::Vector2d((cell_x + 0.5) * pimpl_->resolution_,
                       (cell_y + 0.5) * pimpl_->resolution_);
                       
    for (const auto& particle : particles) {
        if (particle.weight < pimpl_->min_weight_) continue;
        
        Eigen::Vector2d pos = particle.pose.head<2>();
        Eigen::Vector2d diff = pos - cell_center;
        
        if (diff.norm() > pimpl_->max_range_) continue;
        
        mass += static_cast<float>(particle.weight);
        orientation += static_cast<float>(particle.pose.z() * particle.weight);
        
        Eigen::Matrix2d particle_cov = particle.cov.block<2,2>(0,0);
        cov += static_cast<float>(particle.weight) * particle_cov.cast<float>();
    }
    
    if (mass > 0.0f) {
        orientation /= mass;
        cov /= mass;
    }
}

void BeliefRasteriser::normalize_raster(BeliefRaster& raster) const {
    float max_mass = 0.0f;
    for (int y = 0; y < BeliefRaster::HEIGHT; ++y) {
        for (int x = 0; x < BeliefRaster::WIDTH; ++x) {
            max_mass = std::max(max_mass, raster.at(x, y, 0));
        }
    }
    
    if (max_mass > 0.0f) {
        for (int y = 0; y < BeliefRaster::HEIGHT; ++y) {
            for (int x = 0; x < BeliefRaster::WIDTH; ++x) {
                raster.at(x, y, 0) /= max_mass;
            }
        }
    }
}

} // namespace bcod 