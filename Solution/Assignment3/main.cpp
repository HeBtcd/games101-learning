#include <iostream>
#include <opencv2/opencv.hpp>

#include "global.hpp"
#include "rasterizer.hpp"
#include "Triangle.hpp"
#include "Shader.hpp"
#include "Texture.hpp"
#include "OBJ_Loader.h"

Eigen::Matrix4f get_view_matrix(Eigen::Vector3f eye_pos)
{
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

    Eigen::Matrix4f translate;
    translate << 1,0,0,-eye_pos[0],
                 0,1,0,-eye_pos[1],
                 0,0,1,-eye_pos[2],
                 0,0,0,1;

    view = translate*view;

    return view;
}

Eigen::Matrix4f get_model_matrix(float angle)
{
    Eigen::Matrix4f rotation;
    angle = angle * MY_PI / 180.f;
    rotation << cos(angle), 0, sin(angle), 0,
                0, 1, 0, 0,
                -sin(angle), 0, cos(angle), 0,
                0, 0, 0, 1;

    Eigen::Matrix4f scale;
    scale << 2.5, 0, 0, 0,
              0, 2.5, 0, 0,
              0, 0, 2.5, 0,
              0, 0, 0, 1;

    Eigen::Matrix4f translate;
    translate << 1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1;

    return translate * rotation * scale;
}

// eye_fov = 45, aspect_ratio = 1, zNear = 0.1, zFar = 50.
Eigen::Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio,
                                      float zNear, float zFar)
{
    // 左手坐标系.
    zNear = -zNear;
    zFar = -zFar;

    // projection = orthogonal * p2o.
    Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();

    // squish frustum to cuboid.
    float k = zNear + zFar;
    float l = -zNear * zFar;
    Eigen::Matrix4f p2o;
    p2o <<
        zNear,     0, 0, 0,
            0, zNear, 0, 0,
            0,     0, k, l,
            0,     0, 1, 0;
    
    // orthogonal = scaler * translation.
    float rad = eye_fov / 180 * MY_PI;
    float tNear = std::abs(zNear) * std::tan(rad / 2);
    float bNear = -tNear;
    float rNear = tNear * aspect_ratio;
    float lNear = -rNear;

    float dx = -(rNear + lNear) / 2;
    float dy = -(tNear + bNear) / 2;
    float dz = -(zNear + zFar) / 2;
    Eigen::Matrix4f translation;
    translation << 
        1, 0, 0, dx,
        0, 1, 0, dy,
        0, 0, 1, dz,
        0, 0, 0, 1;

    float sx = 2 / (rNear - lNear);
    float sy = 2 / (tNear - bNear);
    float sz = 2 / (zNear - zFar); // near > far.
    Eigen::Matrix4f scaler;
    scaler <<
        sx,  0,  0, 0,
         0, sy,  0, 0,
         0,  0, sz, 0,
         0,  0,  0, 1;

    Eigen::Matrix4f orthogonal;
    orthogonal = scaler * translation;

    projection = orthogonal * p2o * projection;

    return projection;
}

Eigen::Vector3f vertex_shader(const vertex_shader_payload& payload)
{
    return payload.position;
}

Eigen::Vector3f normal_fragment_shader(const fragment_shader_payload& payload)
{
    Eigen::Vector3f return_color = (payload.normal.head<3>().normalized() + Eigen::Vector3f(1.0f, 1.0f, 1.0f)) / 2.f;
    Eigen::Vector3f result;
    result << return_color.x() * 255, return_color.y() * 255, return_color.z() * 255;
    return result;
}

static Eigen::Vector3f reflect(const Eigen::Vector3f& vec, const Eigen::Vector3f& axis)
{
    auto costheta = vec.dot(axis);
    return (2 * costheta * axis - vec).normalized();
}

struct light
{
    Eigen::Vector3f position;
    Eigen::Vector3f intensity;
};

Eigen::Vector3f texture_fragment_shader(const fragment_shader_payload& payload)
{
    Eigen::Vector3f return_color = {0, 0, 0};
    if (payload.texture)
    {
        return payload.texture->getColor(payload.tex_coords.x(), payload.tex_coords.y());
    }
    Eigen::Vector3f texture_color;
    texture_color << return_color.x(), return_color.y(), return_color.z();

    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = texture_color / 255.f;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    auto l1 = light{{20, 20, 20}, {500, 500, 500}};
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};
    Eigen::Vector3f eye_pos{0, 0, 10};

    float p = 150;

    Eigen::Vector3f color = texture_color;
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;

    Eigen::Vector3f result_color = {0, 0, 0};
    for (auto& light : lights)
    {
        const auto l = (light.position - point).normalized();
        const auto v = (eye_pos - point).normalized();
        const auto h = (l + v).normalized();

        const auto diff = std::max(0.f, normal.dot(l)); // 漫反射.
        const auto spec = std::max(0.f, normal.dot(h)); // 高光.

        const auto r2 = (light.position - point).squaredNorm(); // 光能散布在以 r 为半径的球面上.

        const auto diff_color = (light.intensity / r2).cwiseProduct(kd) * diff; // 漫反射.
        const auto spec_color = (light.intensity / r2).cwiseProduct(ks) * std::pow(spec, p); // 高光.

        result_color += diff_color + spec_color;
    }
    const auto amb = amb_light_intensity.cwiseProduct(ka); // 环境光.
    result_color += amb;

    return result_color * 255.f;
}

Eigen::Vector3f phong_fragment_shader(const fragment_shader_payload& payload)
{
    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = payload.color;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    auto l1 = light{{20, 20, 20}, {500, 500, 500}};
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};
    Eigen::Vector3f eye_pos{0, 0, 10};

    float p = 150;

    Eigen::Vector3f color = payload.color;
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;

    Eigen::Vector3f result_color = {0, 0, 0};
    for (auto& light : lights)
    {
        const auto l = (light.position - point).normalized();
        const auto v = (eye_pos - point).normalized();
        const auto h = (l + v).normalized();

        const auto diff = std::max(0.f, normal.dot(l)); // 漫反射.
        const auto spec = std::max(0.f, normal.dot(h)); // 高光.

        const auto r2 = (light.position - point).squaredNorm(); // 光能散布在以 r 为半径的球面上.

        const auto diff_color = (light.intensity / r2).cwiseProduct(kd) * diff; // 漫反射.
        const auto spec_color = (light.intensity / r2).cwiseProduct(ks) * std::pow(spec, p); // 高光.

        result_color += diff_color + spec_color;
    }
    const auto amb = amb_light_intensity.cwiseProduct(ka); // 环境光.
    result_color += amb;

    return result_color * 255.f;
}



Eigen::Vector3f displacement_fragment_shader(const fragment_shader_payload& payload)
{
    
    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = payload.color;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    auto l1 = light{{20, 20, 20}, {500, 500, 500}};
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};
    Eigen::Vector3f eye_pos{0, 0, 10};

    float p = 150;

    Eigen::Vector3f color = payload.color; 
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;

    float kh = 0.2, kn = 0.1;
    
    // TBN.
    Eigen::Vector3f t{
        normal.x() * normal.y() / std::sqrt(normal.x() * normal.x() + normal.z() * normal.z()),
        std::sqrt(normal.x() * normal.x() + normal.z() * normal.z()),
        normal.z() * normal.y() / std::sqrt(normal.x() * normal.x() + normal.z() * normal.z())
    };
    Eigen::Vector3f b = normal.cross(t);
    Eigen::Matrix3f TBN;
    TBN.col(0) = t;
    TBN.col(1) = b;
    TBN.col(2) = normal;

    // UV.
    const auto uv = payload.tex_coords;
    const auto h_val = payload.texture->getColor(uv.x(), uv.y()).norm();
    const auto u_val = payload.texture->getColor(uv.x() + 1.f / payload.texture->width, uv.y()).norm();
    const auto v_val = payload.texture->getColor(uv.x(), uv.y() + 1.f / payload.texture->height).norm();
    
    const auto du = kh * kn * (u_val - h_val);
    const auto dv = kh * kn * (v_val - h_val);
    const auto ln = Eigen::Vector3f(-du, -dv, 1.f);

    // Position.
    point += kn * normal * h_val;

    // Normal.
    Eigen::Vector3f new_normal = TBN * ln;
    new_normal.normalize();

    Eigen::Vector3f result_color = {0, 0, 0};
    for (auto& light : lights)
    {
        const auto l = (light.position - point).normalized();
        const auto v = (eye_pos - point).normalized();
        const auto h = (l + v).normalized();

        const auto diff = std::max(0.f, new_normal.dot(l)); // 漫反射.
        const auto spec = std::max(0.f, new_normal.dot(h)); // 高光.

        const auto r2 = (light.position - point).squaredNorm(); // 光能散布在以 r 为半径的球面上.

        const auto diff_color = (light.intensity / r2).cwiseProduct(kd) * diff; // 漫反射.
        const auto spec_color = (light.intensity / r2).cwiseProduct(ks) * std::pow(spec, p); // 高光.

        result_color += diff_color + spec_color;
    }
    const auto amb = amb_light_intensity.cwiseProduct(ka); // 环境光.
    result_color += amb;

    return result_color * 255.f;
}


Eigen::Vector3f bump_fragment_shader(const fragment_shader_payload& payload)
{
    
    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = payload.color;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    auto l1 = light{{20, 20, 20}, {500, 500, 500}};
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};
    Eigen::Vector3f eye_pos{0, 0, 10};

    float p = 150;

    Eigen::Vector3f color = payload.color; 
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;


    float kh = 0.2, kn = 0.1;

    // TBN.
    Eigen::Vector3f t{
        normal.x() * normal.y() / std::sqrt(normal.x() * normal.x() + normal.z() * normal.z()),
        std::sqrt(normal.x() * normal.x() + normal.z() * normal.z()),
        normal.z() * normal.y() / std::sqrt(normal.x() * normal.x() + normal.z() * normal.z())
    };
    Eigen::Vector3f b = normal.cross(t);
    Eigen::Matrix3f TBN;
    TBN.col(0) = t;
    TBN.col(1) = b;
    TBN.col(2) = normal;

    // UV.
    const auto uv = payload.tex_coords;
    const auto h_val = payload.texture->getColor(uv.x(), uv.y()).norm();
    const auto u_val = payload.texture->getColor(uv.x() + 1.f / payload.texture->width, uv.y()).norm();
    const auto v_val = payload.texture->getColor(uv.x(), uv.y() + 1.f / payload.texture->height).norm();
    
    const auto du = kh * kn * (u_val - h_val);
    const auto dv = kh * kn * (v_val - h_val);
    const auto ln = Eigen::Vector3f(-du, -dv, 1.f);

    // Normal.
    Eigen::Vector3f new_normal = TBN * ln;
    new_normal.normalize();

    Eigen::Vector3f result_color = {0, 0, 0};
    result_color = new_normal;

    return result_color * 255.f;
}

int main(int argc, const char** argv)
{
    std::vector<Triangle*> TriangleList;

    float angle = 140.0;
    bool command_line = false;

    std::string filename = "output.png";
    objl::Loader Loader;
    std::string obj_path = "../models/spot/";

    // Load .obj File
    bool loadout = Loader.LoadFile("../models/spot/spot_triangulated_good.obj");
    for(auto mesh:Loader.LoadedMeshes)
    {
        for(int i=0;i<mesh.Vertices.size();i+=3)
        {
            Triangle* t = new Triangle();
            for(int j=0;j<3;j++)
            {
                t->setVertex(j,Vector4f(mesh.Vertices[i+j].Position.X,mesh.Vertices[i+j].Position.Y,mesh.Vertices[i+j].Position.Z,1.0));
                t->setNormal(j,Vector3f(mesh.Vertices[i+j].Normal.X,mesh.Vertices[i+j].Normal.Y,mesh.Vertices[i+j].Normal.Z));
                t->setTexCoord(j,Vector2f(mesh.Vertices[i+j].TextureCoordinate.X, mesh.Vertices[i+j].TextureCoordinate.Y));
            }
            TriangleList.push_back(t);
        }
    }

    rst::rasterizer r(700, 700);

    auto texture_path = "hmap.jpg";
    r.set_texture(Texture(obj_path + texture_path));

    std::function<Eigen::Vector3f(fragment_shader_payload)> active_shader = phong_fragment_shader;

    if (argc >= 2)
    {
        command_line = true;
        filename = std::string(argv[1]);

        if (argc == 3 && std::string(argv[2]) == "texture")
        {
            std::cout << "Rasterizing using the texture shader\n";
            active_shader = texture_fragment_shader;
            texture_path = "spot_texture.png";
            r.set_texture(Texture(obj_path + texture_path));
        }
        else if (argc == 3 && std::string(argv[2]) == "normal")
        {
            std::cout << "Rasterizing using the normal shader\n";
            active_shader = normal_fragment_shader;
        }
        else if (argc == 3 && std::string(argv[2]) == "phong")
        {
            std::cout << "Rasterizing using the phong shader\n";
            active_shader = phong_fragment_shader;
        }
        else if (argc == 3 && std::string(argv[2]) == "bump")
        {
            std::cout << "Rasterizing using the bump shader\n";
            active_shader = bump_fragment_shader;
        }
        else if (argc == 3 && std::string(argv[2]) == "displacement")
        {
            std::cout << "Rasterizing using the bump shader\n";
            active_shader = displacement_fragment_shader;
        }
    }

    Eigen::Vector3f eye_pos = {0,0,10};

    r.set_vertex_shader(vertex_shader);
    r.set_fragment_shader(active_shader);

    int key = 0;
    int frame_count = 0;

    if (command_line)
    {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);
        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45.0, 1, 0.1, 50));

        r.draw(TriangleList);
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);

        cv::imwrite(filename, image);

        return 0;
    }

    while(key != 27)
    {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45.0, 1, 0.1, 50));

        //r.draw(pos_id, ind_id, col_id, rst::Primitive::Triangle);
        r.draw(TriangleList);
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);

        cv::imshow("image", image);
        cv::imwrite(filename, image);
        key = cv::waitKey(10);

        if (key == 'a' )
        {
            angle -= 0.1;
        }
        else if (key == 'd')
        {
            angle += 0.1;
        }

    }
    return 0;
}
