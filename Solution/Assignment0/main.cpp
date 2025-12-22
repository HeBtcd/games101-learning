#include <eigen3/Eigen/Core>
#include <iostream>
#include <cmath>

Eigen::Vector3f solve(const Eigen::Vector3f& vector){
    // 绕原点先逆时针旋转 45◦, 再平移 (1,2).
    float pointi = std::acos(-1.0f); // cos(pointi) = -1, arccos(-1) = pointi.
    float angle = 45.0f / 180.0f * pointi; // 弧度制 -> 角度制.
    float cos45 = std::cos(angle);
    float sin45 = std::sin(angle);
    
    Eigen::Matrix3f transform;
    float dx = 1.0f;
    float dy = 2.0f;
    transform << cos45, -sin45,   dx, 
             sin45,  cos45,   dy, 
              0.0f,   0.0f, 1.0f;
    return transform * vector;
}

int main(){
    float x, y;
    std::cin >> x >> y;
    Eigen::Vector3f point(x, y, 1.0f);
    auto result = solve(point);
    std::cout << result << std::endl;
    return 0;
}
