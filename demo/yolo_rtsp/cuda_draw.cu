/**
 * CUDA kernels for GPU-accelerated visualization
 * - Draw bounding boxes on GPU frames
 * - NV12 to RGBA conversion
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA kernel: Draw filled rectangle (bounding box)
__global__ void drawRectangleKernel(uchar4* image, int width, int height,
                                     int x1, int y1, int x2, int y2,
                                     uchar4 color, int thickness) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Draw top/bottom borders
    if ((y >= y1 && y < y1 + thickness) || (y >= y2 - thickness && y < y2)) {
        if (x >= x1 && x < x2) {
            image[y * width + x] = color;
        }
    }

    // Draw left/right borders
    if ((x >= x1 && x < x1 + thickness) || (x >= x2 - thickness && x < x2)) {
        if (y >= y1 && y < y2) {
            image[y * width + x] = color;
        }
    }
}

// CUDA kernel: NV12 to RGBA conversion
__global__ void nv12ToRgbaKernel(const unsigned char* nv12, uchar4* rgba,
                                  int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int uv_offset = width * height;
    int uv_x = x / 2;
    int uv_y = y / 2;

    // Get Y, U, V components
    unsigned char Y = nv12[y * width + x];
    unsigned char U = nv12[uv_offset + uv_y * width + uv_x * 2];
    unsigned char V = nv12[uv_offset + uv_y * width + uv_x * 2 + 1];

    // YUV to RGB conversion (ITU-R BT.709)
    int C = Y - 16;
    int D = U - 128;
    int E = V - 128;

    int R = (298 * C + 409 * E + 128) >> 8;
    int G = (298 * C - 100 * D - 208 * E + 128) >> 8;
    int B = (298 * C + 516 * D + 128) >> 8;

    // Clamp to [0, 255]
    R = min(max(R, 0), 255);
    G = min(max(G, 0), 255);
    B = min(max(B, 0), 255);

    rgba[y * width + x] = make_uchar4(R, G, B, 255);
}

// CUDA kernel: BGR to RGBA conversion (for cv::Mat)
__global__ void bgrToRgbaKernel(const unsigned char* bgr, uchar4* rgba,
                                 int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx_bgr = (y * width + x) * 3;
    int idx_rgba = y * width + x;

    rgba[idx_rgba] = make_uchar4(bgr[idx_bgr + 2],  // R
                                  bgr[idx_bgr + 1],  // G
                                  bgr[idx_bgr],      // B
                                  255);              // A
}

// CUDA kernel: RGBA to BGR conversion (back to cv::Mat)
__global__ void rgbaToBgrKernel(const uchar4* rgba, unsigned char* bgr,
                                 int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx_rgba = y * width + x;
    int idx_bgr = (y * width + x) * 3;

    uchar4 pixel = rgba[idx_rgba];
    bgr[idx_bgr] = pixel.z;      // B
    bgr[idx_bgr + 1] = pixel.y;  // G
    bgr[idx_bgr + 2] = pixel.x;  // R
}

// Host wrapper functions
extern "C" {

// Convert NV12 to RGBA on GPU
void cudaNV12ToRGBA(const unsigned char* d_nv12, uchar4* d_rgba,
                    int width, int height, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    nv12ToRgbaKernel<<<grid, block, 0, stream>>>(d_nv12, d_rgba, width, height);
}

// Convert BGR (cv::Mat) to RGBA on GPU
void cudaBGRToRGBA(const unsigned char* d_bgr, uchar4* d_rgba,
                   int width, int height, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    bgrToRgbaKernel<<<grid, block, 0, stream>>>(d_bgr, d_rgba, width, height);
}

// Convert RGBA back to BGR
void cudaRGBAToBGR(const uchar4* d_rgba, unsigned char* d_bgr,
                   int width, int height, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    rgbaToBgrKernel<<<grid, block, 0, stream>>>(d_rgba, d_bgr, width, height);
}

// Draw rectangle on GPU
void cudaDrawRectangle(uchar4* d_image, int width, int height,
                       int x1, int y1, int x2, int y2,
                       unsigned char r, unsigned char g, unsigned char b,
                       int thickness, cudaStream_t stream) {
    // Clamp coordinates
    x1 = max(0, min(x1, width - 1));
    y1 = max(0, min(y1, height - 1));
    x2 = max(0, min(x2, width - 1));
    y2 = max(0, min(y2, height - 1));

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    uchar4 color = make_uchar4(r, g, b, 255);
    drawRectangleKernel<<<grid, block, 0, stream>>>(
        d_image, width, height, x1, y1, x2, y2, color, thickness);
}

} // extern "C"
