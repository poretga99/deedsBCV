#include "deeds/deedsBCV.h"

#include "tbb/tbb.h"

#include <algorithm>
#include <memory>
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
    std::vector<std::uint64_t> moving_mind(size);

    auto fixed_data = static_cast<float *>(m_fixedImage->data);
    auto moving_data = static_cast<float *>(m_movingImage->data);

    for (int level = 0; level < m_parameters->levels; level++) {
        float quant1 = m_parameters->quantisation.at(level);

        float prev_mind_step = mind_steps.at(std::max(level - 1, 0));
        float curr_mind_step = mind_steps.at(level);

        // If needed, recalculate MIND descriptors
        if (level == 0 || prev_mind_step != curr_mind_step) {
            descriptor(fixed_mind.data(), fixed_data, cols, rows, slices, curr_mind_step);
            descriptor(moving_mind.data(), moving_data, cols, rows, slices, curr_mind_step);
        }
    }
}

void DeedsBCV::descriptor(uint64_t *mindq, float *image, int cols, int rows, int slices, int step) {
    // Image neighbour patch offset. 3D image -> 6 neighbours
    const int dx[6] = {+step, +step, -step, +0, +step, +0};
    const int dy[6] = {+step, -step, +0, -step, +0, +step};
    const int dz[6] = {0, +0, +step, +step, +step, +step};

    // SSC connections. See the MIND paper for more info.
    const int num_connections = 12;
    // Swap sx and sy with original code due to the sx being column offset.
    const int sy[num_connections] = {-step, +0, -step, +0, +0, +step, +0, +0, +0, -step, +0, +0};
    const int sx[num_connections] = {+0, -step, +0, +step, +0, +0, +0, +step, +0, +0, +0, -step};
    const int sz[num_connections] = {+0, +0, +0, +0, -step, +0, -step, +0, -step, +0, -step, +0};
    const int index[num_connections] = {0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5};
    const std::uint32_t tablei[6] = {0, 1, 3, 7, 15, 31};

    std::size_t size = rows * cols * slices;

    // Allocate 6 new images, which will be shifted for more efficient calculation.
    std::vector<float> distances(rows * cols * slices * 6);
    tbb::parallel_for(tbb::blocked_range<std::size_t>(0, 6), [&](tbb::blocked_range<std::size_t> &r) {
        for (auto idx = r.begin(); idx < r.end(); idx++) {
            calculateDistances(image, distances.data(), cols, rows, slices, step, idx);
        }
    });

    // Calculate MINDSSC with quantization
    float compare[5]{};
    for (int i = 0; i < 5; i++) {
        compare[i] = -std::log((static_cast<float>(i) + 1.5f) / 6.0f);
    }

    auto executeMIND = [&](tbb::blocked_range<std::size_t> &sl) {
        for (auto s = sl.begin(); s < sl.end(); s++) {
            float mind[num_connections]; // 12 connections for MIND-SSC
            for (std::size_t r = 0; r < rows; r++) {
                for (std::size_t c = 0; c < cols; c++) {
                    for (std::size_t l = 0; l < num_connections; l++) {
                        if (c + sx[l] > 0 && c + sx[l] < cols && r + sy[l] > 0 && r + sy[l] < rows && s + sz[l] > 0 &&
                            s + sz[l] < slices) {
                            mind[l] = distances[(c + sx[l]) + (r + sy[l]) * cols + (s + sz[l]) * rows * cols +
                                                size * index[l]];
                        } else {
                            mind[l] = distances[c + r * cols + s * rows * cols + size * index[l]];
                        }
                    }

                    // MIND equation. Estimate noise as patch sum
                    float minval = *std::min_element(mind, mind + num_connections);
                    float sumnoise = 0.0f;
                    for (std::size_t l = 0; l < num_connections; l++) {
                        mind[l] -= minval;
                        sumnoise += mind[l];
                    }
                    float noise = std::max(sumnoise / static_cast<float>(num_connections), 1e-6f);
                    for (std::size_t l = 0; l < num_connections; l++) {
                        mind[l] /= noise;
                    }

                    // Quantize MINDSSC
                    std::uint64_t accum = 0ull;
                    std::uint64_t tabled = 1ull;
                    const int quantization = 6;
                    for (std::size_t l = 0; l < num_connections; l++) {
                        int mindval = 0;
                        for (std::size_t c = 0; c < quantization - 1; c++) {
                            mindval += compare[c] > mind[l] ? 1 : 0;
                        }
                        accum += tablei[mindval] * tabled;
                        tabled *= 32ull;
                    }
                    mindq[c + r * cols + s * rows * cols] = accum;
                }
            }
        }
    };

    tbb::parallel_for(tbb::blocked_range<std::size_t>(0, slices), executeMIND);
}

void DeedsBCV::calculateDistances(float *image, float *distances, int cols, int rows, int slices, int step, int index) {
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

void DeedsBCV::imshift(const float *iImage, float *oImage, const int dy, int dx, int dz, int cols, int rows,
                       int slices) {
    auto execute = [&](tbb::blocked_range3d<std::size_t> &br) {
        for (auto s = br.pages().begin(); s < br.pages().end(); s++) {
            for (auto r = br.rows().begin(); r < br.rows().end(); r++) {
                for (auto c = br.cols().begin(); c < br.cols().end(); c++) {
                    if (c + dx >= 0 && c + dx < cols && r + dy >= 0 && r + dy < rows && s + dz >= 0 &&
                        s + dz < slices) {
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
    auto execute1 = [&](tbb::blocked_range<std::size_t> &br) {
        for (auto s = br.begin(); s < br.end(); s++) {
            for (std::size_t r = 0; r < rows; r++) {
                for (std::size_t c = 1; c < cols; c++) {
                    temp1[c + r * cols + s * rows * cols] += temp1[(c - 1) + r * cols + s * rows * cols];
                }
            }
        }
    };
    tbb::parallel_for(tbb::blocked_range<std::size_t>(0, slices), execute1);

    auto execute2 = [&](tbb::blocked_range<std::size_t> &br) {
        for (auto s = br.begin(); s < br.end(); s++) {
            for (std::size_t r = 0; r < rows; r++) {
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
    tbb::parallel_for(tbb::blocked_range<std::size_t>(0, slices), execute2);

    auto execute3 = [&](tbb::blocked_range<std::size_t> &br) {
        for (auto s = br.begin(); s < br.end(); s++) {
            for (std::size_t r = 1; r < rows; r++) {
                for (std::size_t c = 0; c < cols; c++) {
                    temp2[c + r * cols + s * rows * cols] += temp2[c + (r - 1) * cols + s * rows * cols];
                }
            }
        }
    };
    tbb::parallel_for(tbb::blocked_range<std::size_t>(0, slices), execute3);

    auto execute4 = [&](tbb::blocked_range<std::size_t> &br) {
        for (auto s = br.begin(); s < br.end(); s++) {
            for (std::size_t c = 0; c < cols; c++) {
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
    tbb::parallel_for(tbb::blocked_range<std::size_t>(0, slices), execute4);

    // This one must be executed on rows, so that there is no race condition!
    auto execute5 = [&](tbb::blocked_range<std::size_t> &br) {
        for (auto r = br.begin(); r < br.end(); r++) {
            for (std::size_t s = 1; s < slices; s++) {
                for (std::size_t c = 0; c < cols; c++) {
                    temp1[c + r * cols + s * rows * cols] += temp1[c + r * cols + (s - 1) * rows * cols];
                }
            }
        }
    };
    tbb::parallel_for(tbb::blocked_range<std::size_t>(0, rows), execute5);

    auto execute6 = [&](tbb::blocked_range<std::size_t> &br) {
        for (auto r = br.begin(); r < br.end(); r++) {
            for (std::size_t c = 0; c < cols; c++) {
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
    tbb::parallel_for(tbb::blocked_range<std::size_t>(0, rows), execute6);
}

void DeedsBCV::interp3(float *interp, const float *input, const float *x1, const float *y1, const float *z1, int m,
                       int n, int o, int m2, int n2, int o2, bool flag) {
    auto execute = [&](tbb::blocked_range3d<std::size_t> &br) {
        for (auto k = br.pages().begin(); k < br.pages().end(); k++) {
            for (auto j = br.rows().begin(); j < br.rows().end(); j++) {
                for (auto i = br.cols().begin(); i < br.cols().end(); i++) {
                    int x = std::floor(x1[i + j * m + k * m * n]);
                    int y = std::floor(y1[i + j * m + k * m * n]);
                    int z = std::floor(z1[i + j * m + k * m * n]);
                    float dx = x1[i + j * m + k * m * n] - x;
                    float dy = y1[i + j * m + k * m * n] - y;
                    float dz = z1[i + j * m + k * m * n] - z;

                    if (flag) {
                        x += j;
                        y += i;
                        z += k;
                    }
                    interp[i + j * m + k * m * n] =
                        (1.0 - dx) * (1.0 - dy) * (1.0 - dz) *
                            input[std::min(std::max(y, 0), m2 - 1) + std::min(std::max(x, 0), n2 - 1) * m2 +
                                  std::min(std::max(z, 0), o2 - 1) * m2 * n2] +
                        (1.0 - dx) * dy * (1.0 - dz) *
                            input[std::min(std::max(y + 1, 0), m2 - 1) + std::min(std::max(x, 0), n2 - 1) * m2 +
                                  std::min(std::max(z, 0), o2 - 1) * m2 * n2] +
                        dx * (1.0 - dy) * (1.0 - dz) *
                            input[std::min(std::max(y, 0), m2 - 1) + std::min(std::max(x + 1, 0), n2 - 1) * m2 +
                                  std::min(std::max(z, 0), o2 - 1) * m2 * n2] +
                        (1.0 - dx) * (1.0 - dy) * dz *
                            input[std::min(std::max(y, 0), m2 - 1) + std::min(std::max(x, 0), n2 - 1) * m2 +
                                  std::min(std::max(z + 1, 0), o2 - 1) * m2 * n2] +
                        dx * dy * (1.0 - dz) *
                            input[std::min(std::max(y + 1, 0), m2 - 1) + std::min(std::max(x + 1, 0), n2 - 1) * m2 +
                                  std::min(std::max(z, 0), o2 - 1) * m2 * n2] +
                        (1.0 - dx) * dy * dz *
                            input[std::min(std::max(y + 1, 0), m2 - 1) + std::min(std::max(x, 0), n2 - 1) * m2 +
                                  std::min(std::max(z + 1, 0), o2 - 1) * m2 * n2] +
                        dx * (1.0 - dy) * dz *
                            input[std::min(std::max(y, 0), m2 - 1) + std::min(std::max(x + 1, 0), n2 - 1) * m2 +
                                  std::min(std::max(z + 1, 0), o2 - 1) * m2 * n2] +
                        dx * dy * dz *
                            input[std::min(std::max(y + 1, 0), m2 - 1) + std::min(std::max(x + 1, 0), n2 - 1) * m2 +
                                  std::min(std::max(z + 1, 0), o2 - 1) * m2 * n2];
                }
            }
        }
    };
    tbb::parallel_for(tbb::blocked_range3d<std::size_t>(0, o, 0, n, 0, m), execute);
}

void DeedsBCV::upsampleDeformationsCL(float *u1, float *v1, float *w1, float *u0, float *v0, float *w0, int m, int n,
                                      int o, int m2, int n2, int o2) {
    float scale_m = static_cast<float>(m) / static_cast<float>(m2);
    float scale_n = static_cast<float>(n) / static_cast<float>(n2);
    float scale_o = static_cast<float>(o) / static_cast<float>(o2);

    std::vector<float> x1(m * n * o);
    std::vector<float> y1(m * n * o);
    std::vector<float> z1(m * n * o);
    auto execute = [&](tbb::blocked_range3d<std::size_t> &br) {
        for (auto k = br.pages().begin(); k < br.pages().end(); k++) {
            for (auto j = br.rows().begin(); j < br.rows().end(); j++) {
                for (auto i = br.cols().begin(); i < br.cols().end(); i++) {
                    x1[i + j * m + k * m * n] = j / scale_n;
                    y1[i + j * m + k * m * n] = i / scale_m;
                    z1[i + j * m + k * m * n] = k / scale_o;
                }
            }
        }
    };
    tbb::parallel_for(tbb::blocked_range3d<std::size_t>(0, o, 0, n, 0, m), execute);
    

    interp3(u1, u0, x1.data(), y1.data(), z1.data(), m, n, o, m2, n2, o2, false);
    interp3(v1, v0, x1.data(), y1.data(), z1.data(), m, n, o, m2, n2, o2, false);
    interp3(w1, w0, x1.data(), y1.data(), z1.data(), m, n, o, m2, n2, o2, false);
}
