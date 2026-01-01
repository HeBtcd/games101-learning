#include <chrono>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <opencv2/opencv.hpp>

std::vector<cv::Point2f> control_points;

const auto STEP = 0.001f;

static inline void put_pixel_if_better(cv::Mat& image, int x, int y, int channel, int value)
{
    if (x < 0 || y < 0 || x >= image.cols || y >= image.rows) return;
    value = std::clamp(value, 0, 255);
    auto& pixel = image.at<cv::Vec3b>(y, x)[channel];
    pixel = static_cast<uchar>(std::max<int>(pixel, value));
}

static inline void draw_antialiased(cv::Mat& image, const cv::Point2f& p, int channel)
{
    const int x0 = static_cast<int>(std::floor(p.x));
    const int y0 = static_cast<int>(std::floor(p.y));

    const float fx = p.x - static_cast<float>(x0);
    const float fy = p.y - static_cast<float>(y0);

    const float w00 = (1.0f - fx) * (1.0f - fy);
    const float w10 = fx * (1.0f - fy);
    const float w01 = (1.0f - fx) * fy;
    const float w11 = fx * fy;

    put_pixel_if_better(image, x0, y0, channel, static_cast<int>(std::round(w00 * 255.0f)));
    put_pixel_if_better(image, x0 + 1, y0, channel, static_cast<int>(std::round(w10 * 255.0f)));
    put_pixel_if_better(image, x0, y0 + 1, channel, static_cast<int>(std::round(w01 * 255.0f)));
    put_pixel_if_better(image, x0 + 1, y0 + 1, channel, static_cast<int>(std::round(w11 * 255.0f)));
}

void mouse_handler(int event, int x, int y, int flags, void *userdata) 
{
    if (event == cv::EVENT_LBUTTONDOWN && control_points.size() < 4) 
    {
        std::cout << "Left button of the mouse is clicked - position (" << x << ", "
        << y << ")" << '\n';
        control_points.emplace_back(x, y);
    }     
}

void naive_bezier(const std::vector<cv::Point2f> &points, cv::Mat &window) 
{
    auto &p_0 = points[0];
    auto &p_1 = points[1];
    auto &p_2 = points[2];
    auto &p_3 = points[3];

    for (double t = 0.0; t <= 1.0; t += 0.001) 
    {
        auto point = std::pow(1 - t, 3) * p_0 + 3 * t * std::pow(1 - t, 2) * p_1 +
                 3 * std::pow(t, 2) * (1 - t) * p_2 + std::pow(t, 3) * p_3;

        draw_antialiased(window, point, 2);
    }
}

cv::Point2f recursive_bezier(const std::vector<cv::Point2f> &control_points, float t) 
{
    if (control_points.size() == 1) return control_points[0];

    std::vector<cv::Point2f> next;
    for (size_t i = 0; i + 1 < control_points.size(); i++)
    {
        auto new_point = (1 - t) * control_points[i] + t * control_points[i + 1];
        next.push_back(new_point);
    }

    return recursive_bezier(next, t);
}

void bezier(const std::vector<cv::Point2f> &control_points, cv::Mat &window) 
{
    for (float t = 0; t < 1; t += STEP)
    {
        auto result = recursive_bezier(control_points, t);
        draw_antialiased(window, result, 1);
    }
}

int main(int argc, const char** argv) 
{
    cv::Mat window = cv::Mat(700, 700, CV_8UC3, cv::Scalar(0));

    if (argc == 9)
    {
        control_points.clear();
        for (int i = 1; i < 9; i += 2)
        {
            control_points.emplace_back(static_cast<float>(std::stoi(argv[i])),
                                        static_cast<float>(std::stoi(argv[i + 1])));
        }
        naive_bezier(control_points, window);
        bezier(control_points, window);
        cv::imwrite("my_bezier_curve.png", window);
        return 0;
    }

    cv::namedWindow("Bezier Curve", cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback("Bezier Curve", mouse_handler, nullptr);

    int key = -1;
    while (key != 27) 
    {
        for (auto &point : control_points) 
        {
            cv::circle(window, point, 3, {255, 255, 255}, 3);
        }

        if (control_points.size() == 4) 
        {
            naive_bezier(control_points, window);
            bezier(control_points, window);

            cv::imshow("Bezier Curve", window);
            cv::imwrite("my_bezier_curve.png", window);
            key = cv::waitKey(0);

            return 0;
        }

        cv::imshow("Bezier Curve", window);
        key = cv::waitKey(20);
    }

    return 0;
}
