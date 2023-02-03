#include "deeds/deedsBCV.h"

#include "tbb/tbb.h"

#include <memory>
#include <algorithm>
#include <numeric>
#include <stdexcept>

DeedsBCV::DeedsBCV(const Parameters &params) : m_parameters(new Parameters(params)) {}

void DeedsBCV::setParameters(const Parameters &params) { m_parameters.reset(new Parameters(params)); }

void DeedsBCV::execute() {
    if (!m_parameters) {
        throw std::runtime_error("Registration parameters are not set!");
    }

    // Read nifti images
    m_fixedImage.reset(nifti_image_read(m_parameters->fixed_image.c_str(), true));
    m_movingImage.reset(nifti_image_read(m_parameters->moving_image.c_str(), true));

    // Check for float types
    if (m_fixedImage->datatype != DT_FLOAT || m_movingImage->datatype != DT_FLOAT) {
        throw std::runtime_error("Only float image type is currently supported!");
    }
    float *fixedImageBuffer = static_cast<float *>(m_fixedImage->data);
    float *movingImageBuffer = static_cast<float *>(m_movingImage->data);

    // Check for same dimensions
    if (m_fixedImage->nx != m_movingImage->nx || m_fixedImage->ny != m_movingImage->ny ||
        m_fixedImage->nz != m_movingImage->nz) {
        throw std::runtime_error("Inconsistent image sizes!");
    }

    std::size_t cols = m_fixedImage->nx;
    std::size_t rows = m_fixedImage->ny;
    std::size_t slices = m_fixedImage->nz;
    std::size_t size = cols * rows * slices;

    // Assumption: We're working with CT scans. 1024HU are added
    tbb::parallel_for(tbb::blocked_range<std::size_t>(0, size), [&](tbb::blocked_range<std::size_t> &r) {
        for (auto idx = r.begin(); idx < r.end(); idx++) {
            fixedImageBuffer[idx] += 1024.0f;
            movingImageBuffer[idx] += 1024.0f;
        }
    });

    // Allocate warped image
    m_deformedImage.reset(nifti_copy_nim_info(m_fixedImage.get()));
    m_deformedImage->data = new float[size];

    // Patch-radius for MIND-SSC descriptors
    std::vector<int> mind_steps(m_parameters->quantisation.size(), 0);
    std::transform(m_parameters->quantisation.begin(), m_parameters->quantisation.end(), mind_steps.begin(),
                   [](int q) { return static_cast<int>(std::floor(0.5f * static_cast<float>(q) + 1.0f)); });

    // Set initial flow-fields to 0; i indicates inverse transform
    // TODO

    // Allocate MIND descriptors -> 12 elements * 5 bits per element quantization.
    // See: http://www.mpheinrich.de/pub/miccai2013_943_mheinrich.pdf
    std::vector<std::uint64_t> fixed_mind(size);
    std::vector<std::uint64_t> fixedb_mind(size);
    std::vector<std::uint64_t> warped_mind(size);

    for (int level = 0; level < m_parameters->levels; level++) {
        float quant1 = m_parameters->quantisation.at(level);

        float prev_mind_step = mind_steps.at(std::max(level - 1, 0));
        float curr_mind_step = mind_steps.at(level);

        // If needed, recalculate MIND descriptors
        if (level == 0 || prev_mind_step != curr_mind_step) {
        }
    }
}

void DeedsBCV::descriptor(uint64_t *mindq, float *image, int cols, int rows, int slices, int step) {
    // Image neighbour patch offset. 3D image -> 6 neighbours
    int dx[6] = {+step, +step, -step, +0, +step, +0};
    int dy[6] = {+step, -step, +0, -step, +0, +step};
    int dz[6] = {0, +0, +step, +step, +step, +step};

    // SSC connections. See the MIND paper for more info.
    int sx[12] = {-step, +0, -step, +0, +0, +step, +0, +0, +0, -step, +0, +0};
    int sy[12] = {+0, -step, +0, +step, +0, +0, +0, +step, +0, +0, +0, -step};
    int sz[12] = {+0, +0, +0, +0, -step, +0, -step, +0, -step, +0, -step, +0};

    int index[12] = {0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5};

    // Allocate 6 new images, which will be shifted for more efficient calculation.
    std::vector<float> distances(rows * cols * slices * 6);
}

void DeedsBCV::distances(float *image, float *distances, int cols, int rows, int slices, int step, int index) {
    std::size_t size = cols * rows * slices;
    auto convolved = std::vector<float>(size);
    auto tmp1 = std::vector<float>(size);
    auto tmp2 = std::vector<float>(size);

    int dx[6] = {+step, +step, -step, +0, +step, +0};
    int dy[6] = {+step, -step, +0, -step, +0, +step};
    int dz[6] = {0, +0, +step, +step, +step, +step};

    imshift(image, convolved.data(), dx[index], dy[index], dz[index], cols, rows, slices);

    tbb::parallel_for(tbb::blocked_range<std::size_t>(0, size), [&](auto &r) {
        for (auto i = r.begin(); i < r.end(); i++) {
            convolved[i] = (convolved[i] - image[i]) * (convolved[i] - image[i]);
        }
    });

    boxfilter(convolved.data(), tmp1.data(), tmp2.data(), step, cols, rows, slices);

    tbb::parallel_for(tbb::blocked_range<std::size_t>(0, size), [&](auto &r) {
        for (auto i = r.begin(); i < r.end(); i++) {
            distances[i + index * size] = convolved[i];
        }
    });
}

void DeedsBCV::imshift(const float *iImage, float *oImage, const int dx, int dy, int dz, int cols, int rows,
                       int slices) {
    auto execute = [&](tbb::blocked_range3d<std::size_t> &br) {
        for (auto s = br.pages().begin(); s < br.pages().end(); s++) {
            for (auto r = br.rows().begin(); r < br.rows().end(); r++) {
                for (auto c = br.cols().begin(); c < br.cols().end(); c++) {
                    if (c + dx > 0 && c + dx < cols && r + dy > 0 && r + dy < rows && s + dz > 0 && s + dz < slices) {
                        oImage[c + r * cols + s * rows * cols] =
                            iImage[c + dx + (r + dy) * cols + (s + dz) * rows * cols];
                    } else {
                        oImage[c + r * cols + s * rows * cols] = iImage[c + r * cols + s * rows * cols];
                    }
                }
            }
        }
    };

    tbb::parallel_for(tbb::blocked_range3d<std::size_t>(0, slices, 0, rows, 0, cols), execute);
}

void DeedsBCV::boxfilter(float *input, float *temp1, float *temp2, int step, int cols, int rows, int slices) {
    std::size_t size = cols * rows * slices;
    std::copy(input, input + size, temp1);

    // Convolution with box filter
    auto execute1 = [&](tbb::blocked_range3d<std::size_t> &br) {
        for (auto s = br.pages().begin(); s < br.pages().end(); s++) {
            for (auto r = br.rows().begin(); r < br.rows().end(); r++) {
                for (auto c = br.cols().begin(); c < br.cols().end(); c++) {
                    temp1[c + r * rows + s * rows * cols] += temp1[(c - 1) + r * cols + s * rows * cols];
                }
            }
        }
    };

    tbb::parallel_for(tbb::blocked_range3d<std::size_t>(0, slices, 0, rows, 1, cols), execute1);

    auto execute2 = [&](tbb::blocked_range2d<std::size_t> &br) {
        for (auto s = br.rows().begin(); s < br.rows().end(); s++) {
            for (auto r = br.cols().begin(); r < br.cols().end(); r++) {
                for (std::size_t c = 0; c < (step + 1); c++) {
                    temp2[c + r * cols + s * rows * cols] = temp1[(c + step) + r * cols + s * rows * cols];
                }
                for (std::size_t c = (step + 1); c < (cols - step); c++) {
                    temp2[c + r * cols + s * rows * cols] = temp1[(c + step) + r * cols + s * rows * cols] -
                                                            temp1[(c - step - 1) + r * cols + s * rows * cols];
                }
                for (std::size_t c = (cols - step); c < cols; c++) {
                    temp2[c + r * cols + s * rows * cols] = temp1[(cols - 1) + r * cols + s * rows * cols] -
                                                            temp1[(c - step - 1) + r * cols + s * rows * cols];
                }
            }
        }
    };
    tbb::parallel_for(tbb::blocked_range2d<std::size_t>(0, slices, 0, rows), execute2);

    auto execute3 = [&](tbb::blocked_range3d<std::size_t> &br) {
        for (auto s = br.pages().begin(); s < br.pages().end(); s++) {
            for (auto r = br.rows().begin(); r < br.rows().end(); r++) {
                for (auto c = br.cols().begin(); c < br.cols().end(); c++) {
                    temp2[c + r * rows + s * rows * cols] += temp2[c + (r - 1) * cols + s * rows * cols];
                }
            }
        }
    };
    tbb::parallel_for(tbb::blocked_range3d<std::size_t>(0, slices, 1, rows, 0, cols), execute3);

    auto execute4 = [&](tbb::blocked_range2d<std::size_t> &br) {
        for (auto s = br.rows().begin(); s < br.rows().end(); s++) {
            for (auto c = br.cols().begin(); c < br.cols().end(); c++) {
                for (std::size_t r = 0; r < (step + 1); r++) {
                    temp1[c + r * cols + s * rows * cols] = temp2[c + (r + step) * cols + s * rows * cols];
                }
                for (std::size_t r = (step + 1); r < (rows - step); r++) {
                    temp1[c + r * cols + s * rows * cols] = temp2[c + (r + step) * cols + s * rows * cols] -
                                                            temp2[c + (r - step - 1) * cols + s * rows * cols];
                }
                for (std::size_t r = (rows - step); r < rows; r++) {
                    temp1[c + r * cols + s * rows * cols] = temp2[c + (r - 1) * cols + s * rows * cols] -
                                                            temp2[c + (r - step - 1) * cols + s * rows * cols];
                }
            }
        }
    };
    tbb::parallel_for(tbb::blocked_range2d<std::size_t>(0, slices, 0, cols), execute4);

    auto execute5 = [&](tbb::blocked_range3d<std::size_t> &br) {
        for (auto s = br.pages().begin(); s < br.pages().end(); s++) {
            for (auto r = br.rows().begin(); r < br.rows().end(); r++) {
                for (auto c = br.cols().begin(); c < br.cols().end(); c++) {
                    temp1[c + r * rows + s * rows * cols] += temp1[c + r * cols + (s - 1) * rows * cols];
                }
            }
        }
    };
    tbb::parallel_for(tbb::blocked_range3d<std::size_t>(1, slices, 0, rows, 0, cols), execute5);

    auto execute6 = [&](tbb::blocked_range2d<std::size_t> &br) {
        for (auto r = br.rows().begin(); r < br.rows().end(); r++) {
            for (auto c = br.cols().begin(); c < br.cols().end(); c++) {
                for (std::size_t s = 0; s < (step + 1); s++) {
                    input[c + r * cols + s * rows * cols] = temp1[c + r * cols + (s + step) * rows * cols];
                }
                for (std::size_t s = (step + 1); s < (slices - step); s++) {
                    input[c + r * cols + s * rows * cols] = temp1[c + r * cols + (s + step) * rows * cols] -
                                                            temp1[c + r * cols + (s - step - 1) * rows * cols];
                }
                for (std::size_t s = (slices - step); s < slices; s++) {
                    input[c + r * cols + s * rows * cols] = temp1[c + r * cols + (s - 1) * rows * cols] -
                                                            temp1[c + r * cols + (s - step - 1) * rows * cols];
                }
            }
        }
    };
    tbb::parallel_for(tbb::blocked_range2d<std::size_t>(0, rows, 0, cols), execute6);
}
