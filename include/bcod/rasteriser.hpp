#pragma once

#include "belief_types.hpp"
#include "json_config.hpp"
#include <memory>

namespace bcod {

class BeliefRasteriser {
public:
    explicit BeliefRasteriser(const JsonConfig& config);
    
    void generate(const ParticleSet& particles, BeliefRaster& raster) const;
    
private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
    
    void compute_cell_stats(const ParticleSet& particles, 
                          int cell_x, int cell_y,
                          float& mass, float& orientation,
                          Eigen::Matrix2f& cov) const;
                          
    void normalize_raster(BeliefRaster& raster) const;
    
    double resolution_;
    Eigen::Vector2d origin_;
    double max_range_;
    double min_weight_;
};

} // namespace bcod 