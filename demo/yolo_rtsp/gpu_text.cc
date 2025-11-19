/**
 * GPU字体渲染器实现
 */

#include "gpu_text.h"
#include <GL/glew.h>
#include <ft2build.h>
#include FT_FREETYPE_H
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>

// 文字渲染着色器（GPU）
const char* textVertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec4 vertex; // <vec2 pos, vec2 tex>
out vec2 TexCoords;

uniform mat4 projection;

void main() {
    gl_Position = projection * vec4(vertex.xy, 0.0, 1.0);
    TexCoords = vertex.zw;
}
)";

const char* textFragmentShaderSource = R"(
#version 330 core
in vec2 TexCoords;
out vec4 color;

uniform sampler2D text;
uniform vec3 textColor;

void main() {
    vec4 sampled = vec4(1.0, 1.0, 1.0, texture(text, TexCoords).r);
    color = vec4(textColor, 1.0) * sampled;
}
)";

GPUTextRenderer::GPUTextRenderer(int screen_width, int screen_height)
    : screen_width_(screen_width), screen_height_(screen_height),
      vao_(0), vbo_(0), shader_program_(0) {}

GPUTextRenderer::~GPUTextRenderer() {
    // 清理纹理
    for (auto& pair : characters_) {
        glDeleteTextures(1, &pair.second.texture_id);
    }
    if (vao_) glDeleteVertexArrays(1, &vao_);
    if (vbo_) glDeleteBuffers(1, &vbo_);
    if (shader_program_) glDeleteProgram(shader_program_);
}

bool GPUTextRenderer::init(const char* font_path, unsigned int font_size) {
    // 初始化FreeType库
    FT_Library ft;
    if (FT_Init_FreeType(&ft)) {
        std::cerr << "Failed to init FreeType" << std::endl;
        return false;
    }

    // 加载字体
    FT_Face face;
    if (FT_New_Face(ft, font_path, 0, &face)) {
        std::cerr << "Failed to load font: " << font_path << std::endl;
        FT_Done_FreeType(ft);
        return false;
    }

    // 设置字体大小
    FT_Set_Pixel_Sizes(face, 0, font_size);

    // 禁用字节对齐限制（OpenGL默认4字节对齐）
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    // 加载所有ASCII字符（32-127）
    for (unsigned char c = 32; c < 128; c++) {
        // 加载字符字形
        if (FT_Load_Char(face, c, FT_LOAD_RENDER)) {
            std::cerr << "Failed to load Glyph: " << c << std::endl;
            continue;
        }

        // 创建OpenGL纹理
        GLuint texture;
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RED,
            face->glyph->bitmap.width,
            face->glyph->bitmap.rows,
            0,
            GL_RED,
            GL_UNSIGNED_BYTE,
            face->glyph->bitmap.buffer
        );

        // 设置纹理选项
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        // 存储字符信息
        Character character = {
            texture,
            glm::ivec2(face->glyph->bitmap.width, face->glyph->bitmap.rows),
            glm::ivec2(face->glyph->bitmap_left, face->glyph->bitmap_top),
            static_cast<unsigned int>(face->glyph->advance.x)
        };
        characters_.insert(std::pair<char, Character>(c, character));
    }

    glBindTexture(GL_TEXTURE_2D, 0);

    // 清理FreeType资源
    FT_Done_Face(face);
    FT_Done_FreeType(ft);

    // 创建着色器
    createShaders();

    // 创建VBO和VAO
    glGenVertexArrays(1, &vao_);
    glGenBuffers(1, &vbo_);
    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6 * 4, NULL, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    std::cout << "GPU Text Renderer initialized with " << characters_.size()
              << " characters" << std::endl;

    return true;
}

void GPUTextRenderer::createShaders() {
    // 编译顶点着色器
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &textVertexShaderSource, NULL);
    glCompileShader(vertexShader);

    // 编译片段着色器
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &textFragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    // 链接着色器程序
    shader_program_ = glCreateProgram();
    glAttachShader(shader_program_, vertexShader);
    glAttachShader(shader_program_, fragmentShader);
    glLinkProgram(shader_program_);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

void GPUTextRenderer::renderText(const std::string& text, float x, float y,
                                  float scale, glm::vec3 color) {
    // 启用混合（用于透明文字）
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // 激活着色器
    glUseProgram(shader_program_);

    // 设置投影矩阵（2D正交投影）
    glm::mat4 projection = glm::ortho(0.0f, static_cast<float>(screen_width_),
                                      0.0f, static_cast<float>(screen_height_));
    glUniformMatrix4fv(glGetUniformLocation(shader_program_, "projection"),
                       1, GL_FALSE, glm::value_ptr(projection));

    glUniform3f(glGetUniformLocation(shader_program_, "textColor"),
                color.x, color.y, color.z);

    glActiveTexture(GL_TEXTURE0);
    glBindVertexArray(vao_);

    // 遍历文字中的所有字符
    std::string::const_iterator c;
    for (c = text.begin(); c != text.end(); c++) {
        Character ch = characters_[*c];

        float xpos = x + ch.bearing.x * scale;
        float ypos = y - (ch.size.y - ch.bearing.y) * scale;

        float w = ch.size.x * scale;
        float h = ch.size.y * scale;

        // 更新VBO（每个字符一个四边形）
        float vertices[6][4] = {
            { xpos,     ypos + h,   0.0f, 0.0f },
            { xpos,     ypos,       0.0f, 1.0f },
            { xpos + w, ypos,       1.0f, 1.0f },

            { xpos,     ypos + h,   0.0f, 0.0f },
            { xpos + w, ypos,       1.0f, 1.0f },
            { xpos + w, ypos + h,   1.0f, 0.0f }
        };

        // 渲染字形纹理到四边形
        glBindTexture(GL_TEXTURE_2D, ch.texture_id);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        // 绘制四边形
        glDrawArrays(GL_TRIANGLES, 0, 6);

        // 移动到下一个字符位置（1/64像素 → 像素）
        x += (ch.advance >> 6) * scale;
    }

    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_BLEND);
}

void GPUTextRenderer::renderTextToTexture(GLuint target_texture, const std::string& text,
                                          float x, float y, float scale, glm::vec3 color) {
    // TODO: 实现FBO绘制到纹理
    // 当前简化版直接绘制到屏幕
    renderText(text, x, y, scale, color);
}
