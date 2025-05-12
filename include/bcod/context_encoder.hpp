#pragma once

#include "belief_types.hpp"
#include <memory>
#include <vector>

namespace bcod {

class ContextEncoder {
public:
    explicit ContextEncoder(const std::string& weights_path);
    ~ContextEncoder();
    
    void encode(const BeliefRaster& raster,
               const Eigen::Vector2d& goal,
               std::vector<float>& context) const;
               
private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
    
    void load_weights(const std::string& path);
    
    void apply_film(const float* input,
                   const float* gamma,
                   const float* beta,
                   float* output,
                   int batch_size,
                   int channels) const;
                   
    int hidden_dim_;
    int num_layers_;
    float dropout_rate_;
    bool use_layer_norm_;
};

} // namespace bcod 