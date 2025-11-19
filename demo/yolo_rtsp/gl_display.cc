/**
 * OpenGL/GLFW window implementation with CUDA interop
 */

#include "gl_display.h"
#include "gpu_text.h"
#include <GL/glew.h>  // GLEW必须在GLFW之前包含
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <stdio.h>
#include <stdlib.h>

// Vertex shader source (fullscreen quad)
const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 TexCoord;

void main() {
    gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);
    TexCoord = aTexCoord;
}
)";

// Fragment shader source (texture rendering)
const char* fragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;
in vec2 TexCoord;

uniform sampler2D textureSampler;

void main() {
    FragColor = texture(textureSampler, TexCoord);
}
)";

// Fullscreen quad vertices (position + texcoord)
// 注意：翻转Y轴纹理坐标以修正图像倒置
float quadVertices[] = {
    // positions   // texCoords (Y轴翻转)
    -1.0f,  1.0f,  0.0f, 0.0f,  // top left (tex: 0,0)
    -1.0f, -1.0f,  0.0f, 1.0f,  // bottom left (tex: 0,1)
     1.0f, -1.0f,  1.0f, 1.0f,  // bottom right (tex: 1,1)

    -1.0f,  1.0f,  0.0f, 0.0f,  // top left (tex: 0,0)
     1.0f, -1.0f,  1.0f, 1.0f,  // bottom right (tex: 1,1)
     1.0f,  1.0f,  1.0f, 0.0f   // top right (tex: 1,0)
};

static void glfwErrorCallback(int error, const char* description) {
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

GLDisplay::GLDisplay(int width, int height, const char* title)
    : width_(width), height_(height) {
    glfwSetErrorCallback(glfwErrorCallback);
}

GLDisplay::~GLDisplay() {
    cleanup();
}

bool GLDisplay::init() {
    // Initialize GLFW
    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return false;
    }

    // OpenGL 3.3 Core Profile
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Create window
    window_ = glfwCreateWindow(width_, height_, "YOLO Detection (GPU)", nullptr, nullptr);
    if (!window_) {
        fprintf(stderr, "Failed to create GLFW window\n");
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(window_);
    glfwSwapInterval(1);  // Enable vsync

    // 初始化GLEW（必须在创建OpenGL context之后）
    glewExperimental = GL_TRUE;  // 启用实验性功能（需要对于核心配置文件）
    GLenum glew_status = glewInit();
    if (glew_status != GLEW_OK) {
        fprintf(stderr, "Failed to initialize GLEW: %s\n", glewGetErrorString(glew_status));
        glfwDestroyWindow(window_);
        glfwTerminate();
        return false;
    }

    printf("OpenGL version: %s\n", glGetString(GL_VERSION));
    printf("GLSL version: %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

    // Create OpenGL texture
    glGenTextures(1, &texture_id_);
    glBindTexture(GL_TEXTURE_2D, texture_id_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width_, height_, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    // Create Pixel Buffer Object (PBO) for CUDA-GL interop
    glGenBuffers(1, &pbo_id_);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_id_);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width_ * height_ * 4, nullptr, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // Register PBO with CUDA
    cudaError_t err = cudaGraphicsGLRegisterBuffer(
        &cuda_pbo_resource_, pbo_id_, cudaGraphicsMapFlagsWriteDiscard);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to register PBO with CUDA: %s\n", cudaGetErrorString(err));
        return false;
    }

    // Create shaders
    createShaders();

    // Create VAO and VBO for fullscreen quad
    glGenVertexArrays(1, &vao_);
    glGenBuffers(1, &vbo_);

    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);

    // Position attribute
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Texture coordinate attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                          (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);

    // 初始化GPU文字渲染器
    text_renderer_ = new GPUTextRenderer(width_, height_);
    // 使用系统字体（Ubuntu使用DejaVu Sans）
    const char* font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf";
    if (!text_renderer_->init(font_path, 24)) {
        fprintf(stderr, "Warning: Failed to initialize text renderer\n");
        delete text_renderer_;
        text_renderer_ = nullptr;
    }

    printf("OpenGL display initialized: %dx%d\n", width_, height_);
    return true;
}

void GLDisplay::createShaders() {
    // Compile vertex shader
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
    glCompileShader(vertexShader);

    // Check vertex shader compilation
    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vertexShader, 512, nullptr, infoLog);
        fprintf(stderr, "Vertex shader compilation failed: %s\n", infoLog);
    }

    // Compile fragment shader
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
    glCompileShader(fragmentShader);

    // Check fragment shader compilation
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fragmentShader, 512, nullptr, infoLog);
        fprintf(stderr, "Fragment shader compilation failed: %s\n", infoLog);
    }

    // Link shader program
    shader_program_ = glCreateProgram();
    glAttachShader(shader_program_, vertexShader);
    glAttachShader(shader_program_, fragmentShader);
    glLinkProgram(shader_program_);

    // Check linking
    glGetProgramiv(shader_program_, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shader_program_, 512, nullptr, infoLog);
        fprintf(stderr, "Shader program linking failed: %s\n", infoLog);
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

void GLDisplay::render(uchar4* d_rgba_image) {
    // 1. Map PBO to CUDA (zero-copy access)
    cudaGraphicsMapResources(1, &cuda_pbo_resource_, 0);

    void* d_pbo_ptr = nullptr;
    size_t num_bytes = 0;
    cudaGraphicsResourceGetMappedPointer(&d_pbo_ptr, &num_bytes, cuda_pbo_resource_);

    // 2. Copy GPU image to PBO (GPU→GPU, no CPU involvement)
    cudaMemcpy(d_pbo_ptr, d_rgba_image, width_ * height_ * sizeof(uchar4),
               cudaMemcpyDeviceToDevice);

    cudaGraphicsUnmapResources(1, &cuda_pbo_resource_, 0);

    // 3. Update OpenGL texture from PBO
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_id_);
    glBindTexture(GL_TEXTURE_2D, texture_id_);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width_, height_,
                    GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // 4. Render fullscreen quad with texture
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(shader_program_);
    glBindVertexArray(vao_);
    glBindTexture(GL_TEXTURE_2D, texture_id_);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    // 注意：不在这里交换缓冲区，等待文字绘制完成后再交换
}

void GLDisplay::present() {
    // 交换缓冲区并处理事件（在所有绘制完成后调用）
    glfwSwapBuffers(window_);
    glfwPollEvents();
}

void GLDisplay::renderText(const std::string& text, float x, float y,
                           float scale, float r, float g, float b) {
    if (text_renderer_) {
        // Y坐标需要翻转（OpenGL坐标系从左下角开始）
        float gl_y = height_ - y;
        text_renderer_->renderText(text, x, gl_y, scale, glm::vec3(r, g, b));
    }
}

bool GLDisplay::shouldClose() {
    return glfwWindowShouldClose(window_);
}

void GLDisplay::cleanup() {
    if (text_renderer_) {
        delete text_renderer_;
        text_renderer_ = nullptr;
    }

    if (cuda_pbo_resource_) {
        cudaGraphicsUnregisterResource(cuda_pbo_resource_);
    }

    if (vao_) glDeleteVertexArrays(1, &vao_);
    if (vbo_) glDeleteBuffers(1, &vbo_);
    if (pbo_id_) glDeleteBuffers(1, &pbo_id_);
    if (texture_id_) glDeleteTextures(1, &texture_id_);
    if (shader_program_) glDeleteProgram(shader_program_);

    if (window_) {
        glfwDestroyWindow(window_);
    }

    glfwTerminate();
}
