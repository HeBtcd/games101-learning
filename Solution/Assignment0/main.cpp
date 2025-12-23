#include <eigen3/Eigen/Core>
#include <iostream>
#include <cmath>

const float PI = std::acos(-1.0f); // cos(pointi) = -1, arccos(-1) = pointi.

Eigen::Vector3f solve(const Eigen::Vector3f& vector){
    // 绕原点先逆时针旋转 45◦, 再平移 (1,2).
    float rad = 45.0f / 180.0f * PI; // 角度制 -> 弧度制.
    float cos45 = std::cos(rad);
    float sin45 = std::sin(rad);
    
    Eigen::Matrix3f rotation;
    rotation << cos45, -sin45, 0., 
                sin45,  cos45, 0., 
                   0.,     0., 1.;
    
    float dx = 1.0f;
    float dy = 2.0f;
    Eigen::Matrix3f translation;
    translation << 1.0f, 0.0f, dx,
                   0.0f, 1.0f, dy,
                   0.0f, 0.0f, 1.0f;
    
    return translation * rotation * vector;
}

int main(){
    float x, y;
    std::cin >> x >> y;
    Eigen::Vector3f point(x, y, 1.0f);
    auto result = solve(point);
    std::cout << result << std::endl;
    return 0;
}
