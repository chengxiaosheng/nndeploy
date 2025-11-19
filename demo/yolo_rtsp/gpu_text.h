/**
 * GPU字体渲染器 - 使用FreeType + OpenGL
 */

#ifndef NNDEPLOY_DEMO_YOLO_RTSP_GPU_TEXT_H_
#define NNDEPLOY_DEMO_YOLO_RTSP_GPU_TEXT_H_

#include <GL/glew.h>
#include <map>
#include <string>
#include <glm/glm.hpp>

// 字符纹理信息
struct Character {
    GLuint texture_id;  // OpenGL纹理ID
    glm::ivec2 size;    // 字形尺寸
    glm::ivec2 bearing; // 从基线到字形左上角的偏移
    unsigned int advance; // 水平偏移到下一个字形
};

class GPUTextRenderer {
 public:
    GPUTextRenderer(int screen_width, int screen_height);
    ~GPUTextRenderer();

    // 初始化字体（加载所有ASCII字符到GPU）
    bool init(const char* font_path, unsigned int font_size = 24);

    // 在GPU上绘制文字（直接绘制到当前OpenGL帧缓冲）
    void renderText(const std::string& text, float x, float y,
                    float scale, glm::vec3 color);

    // 在GPU纹理上绘制文字（用于叠加到RGBA图像）
    void renderTextToTexture(GLuint target_texture, const std::string& text,
                             float x, float y, float scale, glm::vec3 color);

 private:
    int screen_width_;
    int screen_height_;

    // 字符纹理映射表
    std::map<char, Character> characters_;

    // OpenGL资源
    GLuint vao_, vbo_;
    GLuint shader_program_;

    void createShaders();
};

#endif  // NNDEPLOY_DEMO_YOLO_RTSP_GPU_TEXT_H_
