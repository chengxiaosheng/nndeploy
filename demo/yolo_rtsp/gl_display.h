/**
 * OpenGL/GLFW window for GPU-direct display
 * - Zero-copy CUDA-OpenGL interop
 * - Direct GPU texture rendering
 */

#ifndef NNDEPLOY_DEMO_YOLO_RTSP_GL_DISPLAY_H_
#define NNDEPLOY_DEMO_YOLO_RTSP_GL_DISPLAY_H_

#include <GL/glew.h>  // GLEW必须在GLFW之前
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <string>

// 前向声明
class GPUTextRenderer;

class GLDisplay {
 public:
  GLDisplay(int width, int height, const char* title = "YOLO Detection");
  ~GLDisplay();

  // Initialize OpenGL window and CUDA-GL interop
  bool init();

  // Render RGBA GPU buffer to screen (zero-copy)
  void render(uchar4* d_rgba_image);

  // 在GPU上绘制文字（叠加到图像上）
  void renderText(const std::string& text, float x, float y,
                  float scale = 1.0f, float r = 1.0f, float g = 1.0f, float b = 1.0f);

  // 交换缓冲区（在所有绘制完成后调用）
  void present();

  // Check if window should close
  bool shouldClose();

  // Get window dimensions
  int getWidth() const { return width_; }
  int getHeight() const { return height_; }

 private:
  int width_;
  int height_;
  GLFWwindow* window_ = nullptr;

  // OpenGL resources
  GLuint texture_id_ = 0;
  GLuint pbo_id_ = 0;  // Pixel Buffer Object

  // CUDA-OpenGL interop
  cudaGraphicsResource_t cuda_pbo_resource_ = nullptr;

  // Shader program for texture rendering
  GLuint shader_program_ = 0;
  GLuint vao_ = 0;  // Vertex Array Object
  GLuint vbo_ = 0;  // Vertex Buffer Object

  // GPU文字渲染器
  GPUTextRenderer* text_renderer_ = nullptr;

  void createShaders();
  void cleanup();
};

#endif  // NNDEPLOY_DEMO_YOLO_RTSP_GL_DISPLAY_H_
