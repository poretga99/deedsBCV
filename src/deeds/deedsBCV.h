#pragma once
#include <memory>
#include <string>
#include <vector>

#include "nifti/nifti1_io.h"

class DeedsBCV {
  public:
    struct Parameters {
        float alpha{};
        int levels{};
        std::vector<int> grid_spacing{};
        std::vector<int> search_radius{};
        std::vector<int> quantisation{};
        std::string fixed_image{};
        std::string moving_image{};
    };

  public:
    DeedsBCV() = default;
    explicit DeedsBCV(const Parameters &params);

    void setParameters(const Parameters &params);

    void execute();

    void descriptor(uint64_t *mindq, float *image, int cols, int rows, int slices, int step);

    /**
     * @brief Calculate patch distances for neighbourhood
     * @param image 
     * @param distances 
     * @param cols 
     * @param rows 
     * @param slices 
     * @param step 
     * @param index 
    */
    void calculateDistances(float *image, float *distances, int cols, int rows, int slices, int step, int index);

    void imshift(const float *iImage, float *oImage, const int dx, int dy, int dz, int cols, int rows, int slices);

    void boxfilter(float *input, float *temp1, float *temp2, int step, int cols, int rows, int slices);

    void interp3(float *interp, const float *input, const float *x1, const float *y1, const float *z1, int m, int n,
                 int o, int m2, int n2, int o2, bool flag);

    void upsampleDeformationsCL(float *u1, float *v1, float *w1, float *u0, float *v0, float *w0, int m, int n, int o,
                                int m2, int n2, int o2);


  private:
      struct NiftiImageDeleter {
        void operator()(nifti_image *image) { nifti_image_free(image); }
    };
  private:
    std::unique_ptr<Parameters> m_parameters{};
    std::unique_ptr<nifti_image, NiftiImageDeleter> m_fixedImage{};
    std::unique_ptr<nifti_image, NiftiImageDeleter> m_movingImage{};
    std::unique_ptr<nifti_image, NiftiImageDeleter> m_deformedImage{};
};