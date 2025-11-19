/**
 * CUDA visualization kernel interfaces
 */

#ifndef NNDEPLOY_DEMO_YOLO_RTSP_CUDA_DRAW_H_
#define NNDEPLOY_DEMO_YOLO_RTSP_CUDA_DRAW_H_

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Convert NV12 to RGBA on GPU
void cudaNV12ToRGBA(const unsigned char* d_nv12, uchar4* d_rgba,
                    int width, int height, cudaStream_t stream = 0);

// Convert BGR (cv::Mat) to RGBA on GPU
void cudaBGRToRGBA(const unsigned char* d_bgr, uchar4* d_rgba,
                   int width, int height, cudaStream_t stream = 0);

// Convert RGBA back to BGR
void cudaRGBAToBGR(const uchar4* d_rgba, unsigned char* d_bgr,
                   int width, int height, cudaStream_t stream = 0);

// Draw rectangle on GPU
void cudaDrawRectangle(uchar4* d_image, int width, int height,
                       int x1, int y1, int x2, int y2,
                       unsigned char r, unsigned char g, unsigned char b,
                       int thickness = 2, cudaStream_t stream = 0);

#ifdef __cplusplus
}
#endif

#endif  // NNDEPLOY_DEMO_YOLO_RTSP_CUDA_DRAW_H_
