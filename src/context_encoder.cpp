#include "bcod/context_encoder.hpp"
#include "bcod/logging.hpp"
#include <fstream>
#include <cstring>

namespace bcod {

struct ContextEncoder::Impl {
    Impl(const std::string& weights_path) {
        load_weights(weights_path);
        
        hidden_dim_ = 128;
        num_layers_ = 3;
        dropout_rate_ = 0.1f;
        use_layer_norm_ = true;
    }
    
    void load_weights(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            BCOD_ERROR("Failed to open weights file: ", path);
            return;
        }
        
        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
        
        weights_.resize(size / sizeof(float));
        file.read(reinterpret_cast<char*>(weights_.data()), size);
    }
    
    std::vector<float> weights_;
    int hidden_dim_;
    int num_layers_;
    float dropout_rate_;
    bool use_layer_norm_;
};

ContextEncoder::ContextEncoder(const std::string& weights_path)
    : pimpl_(std::make_unique<Impl>(weights_path)) {}

ContextEncoder::~ContextEncoder() = default;

void ContextEncoder::encode(const BeliefRaster& raster,
                          const Eigen::Vector2d& goal,
                          std::vector<float>& context) const {
    const int input_size = BeliefRaster::WIDTH * BeliefRaster::HEIGHT * 
                          BeliefRaster::CHANNELS + 2;
    const int output_size = pimpl_->hidden_dim_;
    
    std::vector<float> input(input_size);
    int idx = 0;
    
    for (int y = 0; y < BeliefRaster::HEIGHT; ++y) {
        for (int x = 0; x < BeliefRaster::WIDTH; ++x) {
            for (int c = 0; c < BeliefRaster::CHANNELS; ++c) {
                input[idx++] = raster.at(x, y, c);
            }
        }
    }
    
    input[idx++] = static_cast<float>(goal.x());
    input[idx++] = static_cast<float>(goal.y());
    
    context.resize(output_size);
    std::fill(context.begin(), context.end(), 0.0f);
    
    for (int layer = 0; layer < pimpl_->num_layers_; ++layer) {
        const float* layer_weights = pimpl_->weights_.data() + 
            layer * (input_size * output_size + 2 * output_size);
            
        const float* gamma = layer_weights + input_size * output_size;
        const float* beta = gamma + output_size;
        
        apply_film(input.data(), gamma, beta, context.data(),
                 1, output_size);
                 
        input = context;
    }
}

void ContextEncoder::apply_film(const float* input,
                              const float* gamma,
                              const float* beta,
                              float* output,
                              int batch_size,
                              int channels) const {
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < channels; ++c) {
            float sum = 0.0f;
            for (int i = 0; i < BeliefRaster::WIDTH * BeliefRaster::HEIGHT * 
                            BeliefRaster::CHANNELS + 2; ++i) {
                sum += input[b * (BeliefRaster::WIDTH * BeliefRaster::HEIGHT * 
                                BeliefRaster::CHANNELS + 2) + i] *
                       gamma[c * (BeliefRaster::WIDTH * BeliefRaster::HEIGHT * 
                                BeliefRaster::CHANNELS + 2) + i];
            }
            
            if (pimpl_->use_layer_norm_) {
                float mean = sum / (BeliefRaster::WIDTH * BeliefRaster::HEIGHT * 
                                  BeliefRaster::CHANNELS + 2);
                float var = 0.0f;
                for (int i = 0; i < BeliefRaster::WIDTH * BeliefRaster::HEIGHT * 
                                BeliefRaster::CHANNELS + 2; ++i) {
                    float diff = input[b * (BeliefRaster::WIDTH * BeliefRaster::HEIGHT * 
                                          BeliefRaster::CHANNELS + 2) + i] - mean;
                    var += diff * diff;
                }
                var /= (BeliefRaster::WIDTH * BeliefRaster::HEIGHT * 
                       BeliefRaster::CHANNELS + 2);
                
                output[b * channels + c] = (sum - mean) / std::sqrt(var + 1e-5f) *
                                         gamma[c] + beta[c];
            } else {
                output[b * channels + c] = sum + beta[c];
            }
        }
    }
}

} // namespace bcod 