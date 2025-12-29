// clang-format off
//
// Created by goksu on 4/6/19.
//

#include <algorithm>
#include <vector>
#include "rasterizer.hpp"
#include <opencv2/opencv.hpp>
// #include <math.h>
#include <cmath>


rst::pos_buf_id rst::rasterizer::load_positions(const std::vector<Eigen::Vector3f> &positions)
{
    auto id = get_next_id();
    pos_buf.emplace(id, positions);

    return {id};
}

rst::ind_buf_id rst::rasterizer::load_indices(const std::vector<Eigen::Vector3i> &indices)
{
    auto id = get_next_id();
    ind_buf.emplace(id, indices);

    return {id};
}

rst::col_buf_id rst::rasterizer::load_colors(const std::vector<Eigen::Vector3f> &cols)
{
    auto id = get_next_id();
    col_buf.emplace(id, cols);

    return {id};
}

auto to_vec4(const Eigen::Vector3f& v3, float w = 1.0f)
{
    return Vector4f(v3.x(), v3.y(), v3.z(), w);
}


static bool insideTriangle(float x, float y, const Vector3f* _v)
{
    const Vector3f p(x, y, 1.f);

    const Vector3f a(_v[0].x(), _v[0].y(), 1.f);
    const Vector3f b(_v[1].x(), _v[1].y(), 1.f);
    const Vector3f c(_v[2].x(), _v[2].y(), 1.f);

    const auto z0 = (b-a).cross(p - a).z();
    const auto z1 = (c-b).cross(p - b).z();
    const auto z2 = (a-c).cross(p - c).z();

    return z0 > 0 && z1 > 0 && z2 > 0 ||
           z0 < 0 && z1 < 0 && z2 < 0;
}

static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Vector3f* v)
{
    float c1 = (x*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*y + v[1].x()*v[2].y() - v[2].x()*v[1].y()) / (v[0].x()*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*v[0].y() + v[1].x()*v[2].y() - v[2].x()*v[1].y());
    float c2 = (x*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*y + v[2].x()*v[0].y() - v[0].x()*v[2].y()) / (v[1].x()*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*v[1].y() + v[2].x()*v[0].y() - v[0].x()*v[2].y());
    float c3 = (x*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*y + v[0].x()*v[1].y() - v[1].x()*v[0].y()) / (v[2].x()*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*v[2].y() + v[0].x()*v[1].y() - v[1].x()*v[0].y());
    return {c1,c2,c3};
}

void rst::rasterizer::draw(pos_buf_id pos_buffer, ind_buf_id ind_buffer, col_buf_id col_buffer, Primitive type)
{
    auto& buf = pos_buf[pos_buffer.pos_id];
    auto& ind = ind_buf[ind_buffer.ind_id];
    auto& col = col_buf[col_buffer.col_id];

    float f1 = (50 - 0.1) / 2.0;
    float f2 = (50 + 0.1) / 2.0;

    Eigen::Matrix4f mvp = projection * view * model;
    for (auto& x : ind)
    {
        Triangle t;
        Eigen::Vector4f v[] = {
                mvp * to_vec4(buf[x[0]], 1.0f),
                mvp * to_vec4(buf[x[1]], 1.0f),
                mvp * to_vec4(buf[x[2]], 1.0f)
        };
        //Homogeneous division
        for (auto& vec : v) {
            vec /= vec.w();
        }
        //Viewport transformation
        for (auto & vert : v)
        {
            vert.x() = 0.5*width*(vert.x()+1.0);
            vert.y() = 0.5*height*(vert.y()+1.0);
            vert.z() = vert.z() * f1 + f2;
        }

        for (int x = 0; x < 3; ++x)
        {
            t.setVertex(x, v[x].head<3>());
            t.setVertex(x, v[x].head<3>());
            t.setVertex(x, v[x].head<3>());
        }

        auto col_x = col[x[0]];
        auto col_y = col[x[1]];
        auto col_z = col[x[2]];

        t.setColor(0, col_x[0], col_x[1], col_x[2]);
        t.setColor(1, col_y[0], col_y[1], col_y[2]);
        t.setColor(2, col_z[0], col_z[1], col_z[2]);

        rasterize_triangle(t);
    }
}

//Screen space rasterization
void rst::rasterizer::rasterize_triangle(const Triangle& t) {
    auto v = t.toVector4();
    
    const auto x_min = std::min({v[0].x(), v[1].x(), v[2].x()});
    const auto y_min = std::min({v[0].y(), v[1].y(), v[2].y()});
    const auto x_max = std::max({v[0].x(), v[1].x(), v[2].x()});
    const auto y_max = std::max({v[0].y(), v[1].y(), v[2].y()});

    const auto x_start = std::floor(x_min);
    const auto x_end = std::ceil(x_max);
    const auto y_start = std::floor(y_min);
    const auto y_end = std::ceil(y_max);

    for (int x = x_start; x < x_end; x++)
    {
        for (int y = y_start; y < y_end; y++)
        {
            if (!insideTriangle(x + 0.5f, y + 0.5f, t.v)) continue;
            
            // 插值公式: P = αA + βB + γC (A, B, C 是三角形的 3 个顶点的某属性).
            const auto [alpha, beta, gamma] = computeBarycentric2D(x + 0.5f, y + 0.5f, t.v);
            
            // 需求: 我们需要近大远小, 物体越远 z 越大, 因此 x/z 就越小, 也就是 x 不变时, 物体在屏幕上越远, 看起来就越小.
            // 问题: 矩阵没有除法运算
            // 方案: 虽然没有除法运算, 但是有归一化, 我们想要 x 除以 z, 只需要让第四个分量等于 z 然后归一化的时候除去就好了,
            //       因此投影矩阵必定携带 `将第三个分量 z 复制到第四个分量 w` 的信息.
            //
            // 问题: 如果只是单纯除以 z (也就是除以第四个分量 w), 那么 x 变成了 x/z (成功), y 变成了 y/z (成功),
            //       但是 z 变成了 z/z = 1 (失败).
            //       如果所有点的深度都变成了 1, 我们就丢失了深度信息, 没法做 Z-Buffer 排序了.
            // 方案: 所以投影矩阵对 z 分量做了特殊变换 (变成了 A*z + B 的形式).
            //       这样在归一化之后, 最终的深度值就变成了 (A*z + B) / z = A + B/z.
            //
            // 问题: 投影变换之后的顶点分布是非线性的 (因为投影后屏幕世界是近大远小的).
            //       此时若直接使用屏幕坐标对属性进行线性插值, 得出来的并不是真实世界下的正确结果.
            // 方案: 为了能够使用线性插值, 我们需要将数据转换为线性空间.
            //       经过数学验证, 虽然属性本身非线性, 但 属性/深度 是线性的.
            //       因此, 我们需要得出 1/深度.
            //       经过投影变换后, w 存储的就是这个深度.
            //       而 z 变成了跟 1/深度 成线性关系的值.
            //       1. 插值出当前像素的 1/w.
            //       2. 插值出当前像素的 属性/w.
            //       3. 为了还原到屏幕空间中, 需要将结果乘以 w (即除以 1/w) 乘回来.
            // 
            // 1. 插值出当前像素的 1/w.
            const auto interp_inv_w = alpha * (1 / v[0].w()) +
                                      beta  * (1 / v[1].w()) +
                                      gamma * (1 / v[2].w());

            // 2. 插值出当前像素的 属性/w.
            auto interp_z_over_w = alpha * v[0].z() / v[0].w() +
                                   beta  * v[1].z() / v[1].w() +
                                   gamma * v[2].z() / v[2].w();

            // 3. 为了还原到屏幕空间中, 需要将结果乘以 w (即除以 1/w) 乘回来.
            const auto pixel_w = 1.f / interp_inv_w;
            const auto pixel_z = interp_z_over_w * pixel_w;

            // 是否覆写 buffer.
            const auto index = get_index(x, y);
            if (!(pixel_z < depth_buf[index])) continue;
            depth_buf[index] = pixel_z;
            set_pixel(Eigen::Vector3f(x, y, 1.f), t.getColor());
        }
    }
}

void rst::rasterizer::set_model(const Eigen::Matrix4f& m)
{
    model = m;
}

void rst::rasterizer::set_view(const Eigen::Matrix4f& v)
{
    view = v;
}

void rst::rasterizer::set_projection(const Eigen::Matrix4f& p)
{
    projection = p;
}

void rst::rasterizer::clear(rst::Buffers buff)
{
    if ((buff & rst::Buffers::Color) == rst::Buffers::Color)
    {
        std::fill(frame_buf.begin(), frame_buf.end(), Eigen::Vector3f{0, 0, 0});
    }
    if ((buff & rst::Buffers::Depth) == rst::Buffers::Depth)
    {
        std::fill(depth_buf.begin(), depth_buf.end(), std::numeric_limits<float>::infinity());
    }
}

rst::rasterizer::rasterizer(int w, int h) : width(w), height(h)
{
    frame_buf.resize(w * h);
    depth_buf.resize(w * h);
}

int rst::rasterizer::get_index(int x, int y)
{
    return (height-1-y)*width + x;
}

void rst::rasterizer::set_pixel(const Eigen::Vector3f& point, const Eigen::Vector3f& color)
{
    //old index: auto ind = point.y() + point.x() * width;
    auto ind = (height-1-point.y())*width + point.x();
    frame_buf[ind] = color;

}

// clang-format on
