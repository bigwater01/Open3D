// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include "Open3D/Camera/PinholeCameraTrajectory.h"
#include "Open3D/Geometry/TriangleMesh.h"
#include "Open3D/IO/ClassIO/IJsonConvertibleIO.h"
#include "Open3D/IO/ClassIO/ImageIO.h"
#include "Open3D/IO/ClassIO/PointCloudIO.h"
#include "Open3D/Visualization/Utility/GLHelper.h"
#include "Open3D/Visualization/Visualizer/ViewParameters.h"
#include "Open3D/Visualization/Visualizer/ViewTrajectory.h"
#include "Open3D/Visualization/Visualizer/Visualizer.h"

#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/opencv.h>
#include "opencv2/core/fast_math.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"


namespace open3d {
namespace visualization {
    
namespace {
std::vector<std::array<int, 2>> DetectFaceLandmarks(
        const cv::Mat image,
        const std::string &dlib_sp_filename,
        const std::string &output_filename = "") {
    dlib::cv_image<dlib::bgr_pixel> dlib_img(image);
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    std::vector<dlib::rectangle> dets = detector(dlib_img);
    dlib::shape_predictor sp;
    std::string sp_filename = dlib_sp_filename;
    dlib::deserialize(sp_filename) >> sp;

    std::vector<std::array<int, 2>> face_landmarks;
    if (dets.size() == 0) {
        std::cout << "no face detected!" << std::endl;
    } else if (dets.size() > 1) {
        std::cout << "more than one face detected!" << std::endl;
    } else {
        dlib::full_object_detection shape = sp(dlib_img, dets[0]);
        for (size_t i = 0; i < 68; ++i)
            face_landmarks.push_back(
                    {image.rows - shape.part(i).y(), shape.part(i).x()});
    }

    if (!output_filename.empty()) {
        std::ofstream file(output_filename);
        if (file.is_open()) {
            for (auto lm : face_landmarks) {
                file << std::to_string(lm[0]) << " " << std::to_string(lm[1])
                     << "\n";
            }
        }
    }
    return face_landmarks;
}
}  // namespace

bool Visualizer::InitOpenGL() {
    glewExperimental = true;
    if (glewInit() != GLEW_OK) {
        utility::LogWarning("Failed to initialize GLEW.");
        return false;
    }

    glGenVertexArrays(1, &vao_id_);
    glBindVertexArray(vao_id_);

    // depth test
    glEnable(GL_DEPTH_TEST);
    glClearDepth(1.0f);

    // pixel alignment
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    // polygon rendering
    glEnable(GL_CULL_FACE);

    // glReadPixels always read front buffer
    glReadBuffer(GL_FRONT);

    return true;
}

void Visualizer::Render() {
    glfwMakeContextCurrent(window_);

    view_control_ptr_->SetViewMatrices();

    glEnable(GL_MULTISAMPLE);
    glDisable(GL_BLEND);
    auto &background_color = render_option_ptr_->background_color_;
    glClearColor((GLclampf)background_color(0), (GLclampf)background_color(1),
                 (GLclampf)background_color(2), 1.0f);
    glClearDepth(1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    for (const auto &renderer_ptr : geometry_renderer_ptrs_) {
        renderer_ptr->Render(*render_option_ptr_, *view_control_ptr_);
    }
    for (const auto &renderer_ptr : utility_renderer_ptrs_) {
        RenderOption *opt = render_option_ptr_.get();
        auto optIt = utility_renderer_opts_.find(renderer_ptr);
        if (optIt != utility_renderer_opts_.end()) {
            opt = &optIt->second;
        }
        renderer_ptr->Render(*opt, *view_control_ptr_);
    }

    glfwSwapBuffers(window_);
}

void Visualizer::ResetViewPoint(bool reset_bounding_box /* = false*/) {
    if (reset_bounding_box) {
        view_control_ptr_->ResetBoundingBox();
        for (const auto &geometry_ptr : geometry_ptrs_) {
            view_control_ptr_->FitInGeometry(*(geometry_ptr));
        }
        if (coordinate_frame_mesh_ptr_ && coordinate_frame_mesh_renderer_ptr_) {
            const auto &boundingbox = view_control_ptr_->GetBoundingBox();
            *coordinate_frame_mesh_ptr_ =
                    *geometry::TriangleMesh::CreateCoordinateFrame(
                            boundingbox.GetMaxExtent() * 0.2,
                            boundingbox.min_bound_);
            coordinate_frame_mesh_renderer_ptr_->UpdateGeometry();
        }
    }
    view_control_ptr_->Reset();
    is_redraw_required_ = true;
}

void Visualizer::CopyViewStatusToClipboard() {
    ViewParameters current_status;
    if (view_control_ptr_->ConvertToViewParameters(current_status) == false) {
        utility::LogError("Something is wrong copying view status.");
    }
    ViewTrajectory trajectory;
    trajectory.view_status_.push_back(current_status);
    std::string clipboard_string;
    if (io::WriteIJsonConvertibleToJSONString(clipboard_string, trajectory) ==
        false) {
        utility::LogError("Something is wrong copying view status.");
    }
    glfwSetClipboardString(window_, clipboard_string.c_str());
}

void Visualizer::CopyViewStatusFromClipboard() {
    const char *clipboard_string_buffer = glfwGetClipboardString(window_);
    if (clipboard_string_buffer != NULL) {
        std::string clipboard_string(clipboard_string_buffer);
        ViewTrajectory trajectory;
        if (io::ReadIJsonConvertibleFromJSONString(clipboard_string,
                                                   trajectory) == false) {
            utility::LogError("Something is wrong copying view status.");
        }
        if (trajectory.view_status_.size() != 1) {
            utility::LogError("Something is wrong copying view status.");
        }
        view_control_ptr_->ConvertFromViewParameters(
                trajectory.view_status_[0]);
    }
}

void Visualizer::CopyViewStatusFromJSONFile(std::string filename) {
    ViewTrajectory trajectory;
    if (io::ReadIJsonConvertibleFromJSON(filename, trajectory) == false) {
        utility::LogError("Something is wrong reading view status Json file.");
    }
    if (trajectory.view_status_.size() != 1) {
        utility::LogError("Something is wrong reading view status Json file.");
    }
    view_control_ptr_->ConvertFromViewParameters(trajectory.view_status_[0]);
}

void Visualizer::CopyViewStatusFromJSONString(std::string json_string) {
    ViewTrajectory trajectory;
    if (io::ReadIJsonConvertibleFromJSONString(json_string, trajectory) ==
        false) {
        utility::LogError(
                "Something is wrong reading view status Json string.");
    }
    if (trajectory.view_status_.size() != 1) {
        utility::LogError(
                "Something is wrong reading view status Json string.");
    }
    view_control_ptr_->ConvertFromViewParameters(trajectory.view_status_[0]);
}

std::shared_ptr<geometry::Image> Visualizer::CaptureScreenFloatBuffer(
        bool do_render /* = true*/) {
    geometry::Image screen_image;
    screen_image.Prepare(view_control_ptr_->GetWindowWidth(),
                         view_control_ptr_->GetWindowHeight(), 3, 4);
    if (do_render) {
        Render();
        is_redraw_required_ = false;
    }
    glFinish();
    glReadPixels(0, 0, view_control_ptr_->GetWindowWidth(),
                 view_control_ptr_->GetWindowHeight(), GL_RGB, GL_FLOAT,
                 screen_image.data_.data());

    // glReadPixels get the screen in a vertically flipped manner
    // Thus we should flip it back.
    auto image_ptr = std::make_shared<geometry::Image>();
    image_ptr->Prepare(view_control_ptr_->GetWindowWidth(),
                       view_control_ptr_->GetWindowHeight(), 3, 4);
    int bytes_per_line = screen_image.BytesPerLine();
    for (int i = 0; i < screen_image.height_; i++) {
        memcpy(image_ptr->data_.data() + bytes_per_line * i,
               screen_image.data_.data() +
                       bytes_per_line * (screen_image.height_ - i - 1),
               bytes_per_line);
    }
    return image_ptr;
}

void Visualizer::CaptureScreenImage(const std::string &filename /* = ""*/,
                                    bool do_render /* = true*/) {
    std::string png_filename = filename;
    std::string camera_filename;
    if (png_filename.empty()) {
        std::string timestamp = utility::GetCurrentTimeStamp();
        png_filename = "ScreenCapture_" + timestamp + ".png";
        camera_filename = "ScreenCamera_" + timestamp + ".json";
    }
    geometry::Image screen_image;
    screen_image.Prepare(view_control_ptr_->GetWindowWidth(),
                         view_control_ptr_->GetWindowHeight(), 3, 1);
    if (do_render) {
        Render();
        is_redraw_required_ = false;
    }
    glFinish();
    glReadPixels(0, 0, view_control_ptr_->GetWindowWidth(),
                 view_control_ptr_->GetWindowHeight(), GL_RGB, GL_UNSIGNED_BYTE,
                 screen_image.data_.data());

    // glReadPixels get the screen in a vertically flipped manner
    // Thus we should flip it back.
    geometry::Image png_image;
    png_image.Prepare(view_control_ptr_->GetWindowWidth(),
                      view_control_ptr_->GetWindowHeight(), 3, 1);
    int bytes_per_line = screen_image.BytesPerLine();
    for (int i = 0; i < screen_image.height_; i++) {
        memcpy(png_image.data_.data() + bytes_per_line * i,
               screen_image.data_.data() +
                       bytes_per_line * (screen_image.height_ - i - 1),
               bytes_per_line);
    }

    utility::LogDebug("[Visualizer] Screen capture to {}",
                      png_filename.c_str());
    io::WriteImage(png_filename, png_image);
    if (!camera_filename.empty()) {
        utility::LogDebug("[Visualizer] Screen camera capture to {}",
                          camera_filename.c_str());
        camera::PinholeCameraParameters parameter;
        view_control_ptr_->ConvertToPinholeCameraParameters(parameter);
        io::WriteIJsonConvertible(camera_filename, parameter);
    }
}

std::shared_ptr<geometry::Image> Visualizer::CaptureDepthFloatBuffer(
        bool do_render /* = true*/) {
    geometry::Image depth_image;
    depth_image.Prepare(view_control_ptr_->GetWindowWidth(),
                        view_control_ptr_->GetWindowHeight(), 1, 4);
    if (do_render) {
        Render();
        is_redraw_required_ = false;
    }
    glFinish();

#if __APPLE__
    // On OSX with Retina display and glfw3, there is a bug with glReadPixels().
    // When using glReadPixels() to read a block of depth data. The data is
    // horizontally stretched (vertically it is fine). This issue is related
    // to GLFW_SAMPLES hint. When it is set to 0 (anti-aliasing disabled),
    // glReadPixels() works fine. See this post for details:
    // http://stackoverflow.com/questions/30608121/glreadpixel-one-pass-vs-looping-through-points
    // The reason of this bug is unknown. The current workaround is to read
    // depth buffer column by column. This is 15~30 times slower than one block
    // reading glReadPixels().
    std::vector<float> float_buffer(depth_image.height_);
    float *p = (float *)depth_image.data_.data();
    for (int j = 0; j < depth_image.width_; j++) {
        glReadPixels(j, 0, 1, depth_image.height_, GL_DEPTH_COMPONENT, GL_FLOAT,
                     float_buffer.data());
        for (int i = 0; i < depth_image.height_; i++) {
            p[i * depth_image.width_ + j] = float_buffer[i];
        }
    }
#else   //__APPLE__
    // By default, glReadPixels read a block of depth buffer.
    glReadPixels(0, 0, depth_image.width_, depth_image.height_,
                 GL_DEPTH_COMPONENT, GL_FLOAT, depth_image.data_.data());
#endif  //__APPLE__

    // glReadPixels get the screen in a vertically flipped manner
    // We should flip it back, and convert it to the correct depth value
    auto image_ptr = std::make_shared<geometry::Image>();
    double z_near = view_control_ptr_->GetZNear();
    double z_far = view_control_ptr_->GetZFar();

    image_ptr->Prepare(view_control_ptr_->GetWindowWidth(),
                       view_control_ptr_->GetWindowHeight(), 1, 4);
    for (int i = 0; i < depth_image.height_; i++) {
        float *p_depth = (float *)(depth_image.data_.data() +
                                   depth_image.BytesPerLine() *
                                           (depth_image.height_ - i - 1));
        float *p_image = (float *)(image_ptr->data_.data() +
                                   image_ptr->BytesPerLine() * i);
        for (int j = 0; j < depth_image.width_; j++) {
            if (p_depth[j] == 1.0) {
                continue;
            }
            double z_depth =
                    2.0 * z_near * z_far /
                    (z_far + z_near -
                     (2.0 * (double)p_depth[j] - 1.0) * (z_far - z_near));
            p_image[j] = (float)z_depth;
        }
    }
    return image_ptr;
}

void Visualizer::CaptureDepthImage(const std::string &filename /* = ""*/,
                                   bool do_render /* = true*/,
                                   double depth_scale /* = 1000.0*/) {
    std::string png_filename = filename;
    std::string camera_filename;
    if (png_filename.empty()) {
        std::string timestamp = utility::GetCurrentTimeStamp();
        png_filename = "DepthCapture_" + timestamp + ".png";
        camera_filename = "DepthCamera_" + timestamp + ".json";
    }
    geometry::Image depth_image;
    depth_image.Prepare(view_control_ptr_->GetWindowWidth(),
                        view_control_ptr_->GetWindowHeight(), 1, 4);

    if (do_render) {
        Render();
        is_redraw_required_ = false;
    }
    glFinish();

#if __APPLE__
    // On OSX with Retina display and glfw3, there is a bug with glReadPixels().
    // When using glReadPixels() to read a block of depth data. The data is
    // horizontally streched (vertically it is fine). This issue is related
    // to GLFW_SAMPLES hint. When it is set to 0 (anti-aliasing disabled),
    // glReadPixels() works fine. See this post for details:
    // http://stackoverflow.com/questions/30608121/glreadpixel-one-pass-vs-looping-through-points
    // The reason of this bug is unknown. The current workaround is to read
    // depth buffer column by column. This is 15~30 times slower than one block
    // reading glReadPixels().
    std::vector<float> float_buffer(depth_image.height_);
    float *p = (float *)depth_image.data_.data();
    for (int j = 0; j < depth_image.width_; j++) {
        glReadPixels(j, 0, 1, depth_image.width_, GL_DEPTH_COMPONENT, GL_FLOAT,
                     float_buffer.data());
        for (int i = 0; i < depth_image.height_; i++) {
            p[i * depth_image.width_ + j] = float_buffer[i];
        }
    }
#else   //__APPLE__
    // By default, glReadPixels read a block of depth buffer.
    glReadPixels(0, 0, depth_image.width_, depth_image.height_,
                 GL_DEPTH_COMPONENT, GL_FLOAT, depth_image.data_.data());
#endif  //__APPLE__

    // glReadPixels get the screen in a vertically flipped manner
    // We should flip it back, and convert it to the correct depth value
    geometry::Image png_image;
    double z_near = view_control_ptr_->GetZNear();
    double z_far = view_control_ptr_->GetZFar();

    png_image.Prepare(view_control_ptr_->GetWindowWidth(),
                      view_control_ptr_->GetWindowHeight(), 1, 2);
    for (int i = 0; i < depth_image.height_; i++) {
        float *p_depth = (float *)(depth_image.data_.data() +
                                   depth_image.BytesPerLine() *
                                           (depth_image.height_ - i - 1));
        uint16_t *p_png = (uint16_t *)(png_image.data_.data() +
                                       png_image.BytesPerLine() * i);
        for (int j = 0; j < depth_image.width_; j++) {
            if (p_depth[j] == 1.0) {
                continue;
            }
            double z_depth =
                    2.0 * z_near * z_far /
                    (z_far + z_near -
                     (2.0 * (double)p_depth[j] - 1.0) * (z_far - z_near));
            p_png[j] = (uint16_t)std::min(std::round(depth_scale * z_depth),
                                          (double)INT16_MAX);
        }
    }

    utility::LogDebug("[Visualizer] Depth capture to {}", png_filename.c_str());
    io::WriteImage(png_filename, png_image);
    if (!camera_filename.empty()) {
        utility::LogDebug("[Visualizer] Depth camera capture to {}",
                          camera_filename.c_str());
        camera::PinholeCameraParameters parameter;
        view_control_ptr_->ConvertToPinholeCameraParameters(parameter);
        io::WriteIJsonConvertible(camera_filename, parameter);
    }
}

void Visualizer::CaptureDepthPointCloud(
        const std::string &filename /* = ""*/,
        bool do_render /* = true*/,
        bool convert_to_world_coordinate /* = false*/) {
    std::string ply_filename = filename;
    std::string camera_filename;
    if (ply_filename.empty()) {
        std::string timestamp = utility::GetCurrentTimeStamp();
        ply_filename = "DepthCapture_" + timestamp + ".ply";
        camera_filename = "DepthCamera_" + timestamp + ".json";
    }
    geometry::Image depth_image;
    depth_image.Prepare(view_control_ptr_->GetWindowWidth(),
                        view_control_ptr_->GetWindowHeight(), 1, 4);

    if (do_render) {
        Render();
        is_redraw_required_ = false;
    }
    glFinish();

#if __APPLE__
    // On OSX with Retina display and glfw3, there is a bug with glReadPixels().
    // When using glReadPixels() to read a block of depth data. The data is
    // horizontally stretched (vertically it is fine). This issue is related
    // to GLFW_SAMPLES hint. When it is set to 0 (anti-aliasing disabled),
    // glReadPixels() works fine. See this post for details:
    // http://stackoverflow.com/questions/30608121/glreadpixel-one-pass-vs-looping-through-points
    // The reason of this bug is unknown. The current workaround is to read
    // depth buffer column by column. This is 15~30 times slower than one block
    // reading glReadPixels().
    std::vector<float> float_buffer(depth_image.height_);
    float *p = (float *)depth_image.data_.data();
    for (int j = 0; j < depth_image.width_; j++) {
        glReadPixels(j, 0, 1, depth_image.width_, GL_DEPTH_COMPONENT, GL_FLOAT,
                     float_buffer.data());
        for (int i = 0; i < depth_image.height_; i++) {
            p[i * depth_image.width_ + j] = float_buffer[i];
        }
    }
#else   //__APPLE__
    // By default, glReadPixels read a block of depth buffer.
    glReadPixels(0, 0, depth_image.width_, depth_image.height_,
                 GL_DEPTH_COMPONENT, GL_FLOAT, depth_image.data_.data());
#endif  //__APPLE__

    GLHelper::GLMatrix4f mvp_matrix;
    if (convert_to_world_coordinate) {
        mvp_matrix = view_control_ptr_->GetMVPMatrix();
    } else {
        mvp_matrix = view_control_ptr_->GetProjectionMatrix();
    }

    // glReadPixels get the screen in a vertically flipped manner
    // We should flip it back, and convert it to the correct depth value
    geometry::PointCloud depth_pointcloud;
    for (int i = 0; i < depth_image.height_; i++) {
        float *p_depth = (float *)(depth_image.data_.data() +
                                   depth_image.BytesPerLine() * i);
        for (int j = 0; j < depth_image.width_; j++) {
            if (p_depth[j] == 1.0) {
                continue;
            }
            depth_pointcloud.points_.push_back(GLHelper::Unproject(
                    Eigen::Vector3d(j + 0.5, i + 0.5, p_depth[j]), mvp_matrix,
                    view_control_ptr_->GetWindowWidth(),
                    view_control_ptr_->GetWindowHeight()));
        }
    }

    utility::LogDebug("[Visualizer] Depth point cloud capture to {}",
                      ply_filename.c_str());
    io::WritePointCloud(ply_filename, depth_pointcloud);
    if (!camera_filename.empty()) {
        utility::LogDebug("[Visualizer] Depth camera capture to {}",
                          camera_filename.c_str());
        camera::PinholeCameraParameters parameter;
        view_control_ptr_->ConvertToPinholeCameraParameters(parameter);
        io::WriteIJsonConvertible(camera_filename, parameter);
    }
}

void Visualizer::CaptureRenderOption(const std::string &filename /* = ""*/) {
    std::string json_filename = filename;
    if (json_filename.empty()) {
        std::string timestamp = utility::GetCurrentTimeStamp();
        json_filename = "RenderOption_" + timestamp + ".json";
    }
    utility::LogDebug("[Visualizer] Render option capture to {}",
                      json_filename.c_str());
    io::WriteIJsonConvertible(json_filename, *render_option_ptr_);
}

std::vector<std::array<double, 3>> Visualizer::CaptureFaceLandmarks(
        const std::string &dlib_sp_fn,
        const std::vector<int> &selected_landmark_ids,
        const std::string &output_fn) {
    geometry::Image screen_image;
    screen_image.Prepare(view_control_ptr_->GetWindowWidth(),
                         view_control_ptr_->GetWindowHeight(), 3, 1);

    geometry::Image depth_image;
    depth_image.Prepare(view_control_ptr_->GetWindowWidth(),
                        view_control_ptr_->GetWindowHeight(), 1, 4);

    Render();
    is_redraw_required_ = false;

    glFinish();
    glReadPixels(0, 0, view_control_ptr_->GetWindowWidth(),
                 view_control_ptr_->GetWindowHeight(), GL_RGB, GL_UNSIGNED_BYTE,
                 screen_image.data_.data());

    // glReadPixels get the screen in a vertically flipped manner
    // Thus we should flip it back.
    geometry::Image png_image;
    png_image.Prepare(view_control_ptr_->GetWindowWidth(),
                      view_control_ptr_->GetWindowHeight(), 3, 1);
    int bytes_per_line = screen_image.BytesPerLine();
    for (int i = 0; i < screen_image.height_; i++) {
        memcpy(png_image.data_.data() + bytes_per_line * i,
               screen_image.data_.data() +
                       bytes_per_line * (screen_image.height_ - i - 1),
               bytes_per_line);
    }

    cv::Mat cv_image = cv::Mat(png_image.height_, png_image.width_, CV_8UC1);
    memcpy(cv_image.data, png_image.data_.data(),
           png_image.data_.size() * sizeof(uchar));

#if __APPLE__
    // On OSX with Retina display and glfw3, there is a bug with glReadPixels().
    // When using glReadPixels() to read a block of depth data. The data is
    // horizontally stretched (vertically it is fine). This issue is related
    // to GLFW_SAMPLES hint. When it is set to 0 (anti-aliasing disabled),
    // glReadPixels() works fine. See this post for details:
    // http://stackoverflow.com/questions/30608121/glreadpixel-one-pass-vs-looping-through-points
    // The reason of this bug is unknown. The current workaround is to read
    // depth buffer column by column. This is 15~30 times slower than one block
    // reading glReadPixels().
    std::vector<float> float_buffer(depth_image.height_);
    float *p = (float *)depth_image.data_.data();
    for (int j = 0; j < depth_image.width_; j++) {
        glReadPixels(j, 0, 1, depth_image.width_, GL_DEPTH_COMPONENT, GL_FLOAT,
                     float_buffer.data());
        for (int i = 0; i < depth_image.height_; i++) {
            p[i * depth_image.width_ + j] = float_buffer[i];
        }
    }
#else   //__APPLE__
    // By default, glReadPixels read a block of depth buffer.
    glReadPixels(0, 0, depth_image.width_, depth_image.height_,
                 GL_DEPTH_COMPONENT, GL_FLOAT, depth_image.data_.data());
#endif  //__APPLE__

    GLHelper::GLMatrix4f mvp_matrix;
    bool convert_to_world_coordinate = true;
    if (convert_to_world_coordinate) {
        mvp_matrix = view_control_ptr_->GetMVPMatrix();
    } else {
        mvp_matrix = view_control_ptr_->GetProjectionMatrix();
    }

    std::vector<std::array<int, 2>> landmarks, landmarks_uv;
    landmarks = DetectFaceLandmarks(cv_image, dlib_sp_fn);
    landmarks_uv.resize(0);
    if (landmarks_uv.size() == 0) return std::vector<std::array<double, 3>>();
    if (selected_landmark_ids.size() > 0) {
        for (int i = 0; i < selected_landmark_ids.size(); ++i) {
            landmarks_uv.push_back(landmarks[selected_landmark_ids[i]]);
        }
    }

    std::vector<std::array<double, 3>> landmarks_xyz;
    for (int i = 0; i < landmarks_uv.size(); ++i) {
        landmarks_xyz.push_back({0.0, 0.0, 0.0});
    }
    // glReadPixels get the screen in a vertically flipped manner
    // We should flip it back, and convert it to the correct depth value
    geometry::PointCloud depth_pointcloud;
    int count = 0;
    for (int i = 0; i < depth_image.height_; i++) {
        float *p_depth = (float *)(depth_image.data_.data() +
                                   depth_image.BytesPerLine() * i);
        for (int j = 0; j < depth_image.width_; j++) {
            auto find_uv = [=](std::array<int, 2> &uv) {
                return (uv[0] == i) && (uv[1] == j);
            };
            if (p_depth[j] == 1.0) {
                continue;
            }
            auto point = GLHelper::Unproject(
                    Eigen::Vector3d(j + 0.5, i + 0.5, p_depth[j]), mvp_matrix,
                    view_control_ptr_->GetWindowWidth(),
                    view_control_ptr_->GetWindowHeight());
            depth_pointcloud.points_.push_back(point);

            auto it = std::find_if(landmarks_uv.begin(), landmarks_uv.end(),
                                   find_uv);
            if (it != landmarks_uv.end()) {
                landmarks_xyz[it - landmarks_uv.begin()] = {point[0], point[1],
                                                            point[2]};
            }
        }
    }

    if (!output_fn.empty()) {
        std::ofstream file(output_fn);
        if (file.is_open()) {
            for (auto lm : landmarks_xyz) {
                file << std::to_string(lm[0]) << " " << std::to_string(lm[1])
                     << " " << std::to_string(lm[2]) << "\n";
            }
        }
    }

    return landmarks_xyz;
}


}  // namespace visualization
}  // namespace open3d
