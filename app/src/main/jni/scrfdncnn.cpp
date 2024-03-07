// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <android/asset_manager_jni.h>
#include <android/native_window_jni.h>
#include <android/native_window.h>

#include <android/log.h>

#include <jni.h>

#include <string>
#include <vector>

#include <platform.h>
#include <benchmark.h>

#include "scrfd.h"

#include "ndkcamera.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

    class ContourWithData {
    public:
        int erosion_n = 1;
        int MIN_CONTOUR_AREA = 6;
        int MAX_CONTOUR_AREA = 300;
        std::vector<cv::Point> contour;
        cv::Rect boundingRect;
        cv::RotatedRect rotRect;
        cv::Point center;
        double area;
        int groupID;

        bool isContourValid()
        {
            float height(rotRect.size.height);
            float width(rotRect.size.width);
            float short_side = height < width ? height : width;
            float long_side = height > width ? height : width;
            short_side *= 2;

            return ((short_side / long_side > 0.4) & (area > MIN_CONTOUR_AREA));
        }

        static bool sortByBoundingRectCenterY(const ContourWithData& cwdLeft, const ContourWithData& cwdRight)
        {
            return (cwdLeft.center.x > cwdRight.center.x);
        }
        static bool sortByBoundingRectX(const ContourWithData& cwdLeft, const ContourWithData& cwdRight)
        {
            return ((cwdLeft.boundingRect.x + 100000 * (cwdLeft.boundingRect.y)) > (cwdRight.boundingRect.x + 100000 * (cwdRight.boundingRect.y)));
            //return ((cwdLeft.boundingRect.x + 10000 * (cwdLeft.boundingRect.y / 25 * 25)) > (cwdRight.boundingRect.x + 10000 * (cwdRight.boundingRect.y / 25 * 25)));
        }
        static bool sortByBoundingRectWidth(const ContourWithData& cwdLeft, const ContourWithData& cwdRight)
        {
            return ((cwdLeft.boundingRect.width - cwdLeft.boundingRect.height) > (cwdRight.boundingRect.width - cwdRight.boundingRect.height));
        }

        static bool sortByBoundingRectCenter(const ContourWithData& cwdLeft, const ContourWithData& cwdRight)
        {
            return ((cwdLeft.center.y + 10000 * (cwdLeft.center.x)) > (cwdRight.center.y + 10000 * (cwdRight.center.x)));
        }
    };

    class HopeOMr
    {
    public:

        HopeOMr(cv::InputOutputArray _src, bool dbg = false);
        void drawRotatedRect(cv::InputOutputArray src, cv::RotatedRect rrect, cv::Scalar color = cv::Scalar(0, 255, 0), int thickness = 1);
        bool getRects();
        bool getRects2();
        bool drawRects();
        cv::RotatedRect RRect;
        std::vector<std::vector<cv::Point>> vec_pts;
        cv::Mat thresh0;
        cv::Mat thresh1;
        bool debug;

    protected:
        cv::Mat image;
        cv::Mat gray;
        std::vector<ContourWithData> validContoursWithData;
        std::vector<cv::Rect> rects;
        int erosion_n = 2;
        bool getRRectParts();
        cv::RotatedRect getRRectPart(std::vector<cv::Point> contour);
    };

    const static cv::Scalar colors[] =
    {
        cv::Scalar(0,0,0),       // 0 Black  
        cv::Scalar(255,255,255), // 1 White
        cv::Scalar(255,0,0),     // 2 Blue
        cv::Scalar(0,255,0),     // 3 Lime
        cv::Scalar(0,0,255),     // 4 Red
        cv::Scalar(0,255,255),   // 5 Yellow
        cv::Scalar(255,255,0),   // 6 Cyan
        cv::Scalar(255,0,255),   // 7 Magenta
        cv::Scalar(128,0,0),
        cv::Scalar(0,128,0),
        cv::Scalar(0,0,128),
        cv::Scalar(0,128,128),
        cv::Scalar(128,128,0),
        cv::Scalar(128,0,128),
        cv::Scalar(64,0,0),
        cv::Scalar(0,64,0),
        cv::Scalar(0,0,64),
        cv::Scalar(0,64,64),
        cv::Scalar(64,64,0),
        cv::Scalar(64,0,64),
    };

    struct point_sorter_x_asc // sorts points by their x ascending
    {
        bool operator ()(const cv::Point& a, const cv::Point& b)
        {
            return a.x < b.x;
        }
    };

    struct contour_sorter // 'less' for contours
    {
        bool operator ()(const std::vector<cv::Point>& a, const std::vector<cv::Point>& b)
        {
            cv::Rect ra(cv::boundingRect(a));
            cv::Rect rb(cv::boundingRect(b));
            // scale factor for y should be larger than img.width
            return ((ra.x + 1000 * ra.y) < (rb.x + 1000 * rb.y));
        }
    };

    static void adjustRotatedRect(cv::RotatedRect& rrect)
    {
        if (rrect.angle < -45.)
        {
            rrect.angle += 90.0;
            std::swap(rrect.size.width, rrect.size.height);
        }
    }

    static cv::Point2f getCentroid(cv::InputArray Points)
    {
        cv::Moments mm = cv::moments(Points, false);

        cv::Point2f Coord = cv::Point2f(static_cast<float>(mm.m10 / (mm.m00 + 1e-5)),
            static_cast<float>(mm.m01 / (mm.m00 + 1e-5)));
        return Coord;
    }


    HopeOMr::HopeOMr(cv::InputOutputArray src, bool dbg)
    {
        debug = dbg;
        image = src.getMat();
        gray.create(image.size(), CV_8U);
        thresh0.create(image.size(), CV_8U);
        thresh1.create(image.size(), CV_8U);
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    }

    void HopeOMr::drawRotatedRect(cv::InputOutputArray src, cv::RotatedRect rrect, cv::Scalar color, int thickness)
    {
        cv::Point2f vtx[4];
        rrect.points(vtx);
        for (int i = 0; i < 4; i++)
            cv::line(src, vtx[i], vtx[(i + 1) % 4], color, thickness);
    }

    bool getRects2helper(cv::Mat& cwindow, cv::Mat& gwindow, cv::Mat& twindow)
    {
        cv::cvtColor(cwindow, gwindow, cv::COLOR_BGR2GRAY);
        cv::adaptiveThreshold(gwindow, twindow, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 11, 10);

        std::vector<std::vector<cv::Point> > contours;
        cv::findContours(twindow, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for (size_t i = 0; i < contours.size(); i++)
        {
            cv::polylines(cwindow, contours[i], false, cv::Scalar(0, 0, 255), 1);
        }
        return true;
    }

    bool HopeOMr::getRects2()
    {
        cv::Rect center_rect(image.cols / 2, image.rows / 2, 60, 60);
        cv::Mat cwindow = image(center_rect);
        cv::Mat gwindow = gray(center_rect);
        cv::Mat twindow = thresh0(center_rect);

        getRects2helper(cwindow, gwindow, twindow);
        //cwindow.setTo(Scalar(255, 0, 0), twindow);
        return true;
     }

    bool HopeOMr::getRects()
    {
        cv::Point pt;
        cv::adaptiveThreshold(gray, thresh0, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 31, 10);
        cv::erode(thresh0, thresh1, cv::Mat(), cv::Point(-1, -1), erosion_n);

        std::vector<std::vector<cv::Point> > contours;
        cv::findContours(thresh1, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

        cv::Point2f vtx[4];

        for (size_t i = 0; i < contours.size(); i++)
        {
            ContourWithData contourWithData;
            contourWithData.contour = contours[i];
            contourWithData.boundingRect = cv::boundingRect(contourWithData.contour);
            contourWithData.rotRect = cv::minAreaRect(contours[i]);
            contourWithData.center = contourWithData.rotRect.center;

            contourWithData.area = contourWithData.boundingRect.width * contourWithData.boundingRect.height;

            if (contourWithData.isContourValid())
            {
                contourWithData.groupID = 1;
                validContoursWithData.push_back(contourWithData);
            }
        }

        cv::Point chain_pt0;
        if (validContoursWithData.size() > 0)
            chain_pt0 = validContoursWithData[0].center;

        int avgdim = thresh1.cols / 90;
        int mindim = cvCeil(avgdim * 0.6);
        int maxdim = avgdim * 3;

        std::vector<cv::Point> pts;

        for (size_t i = 0; i < validContoursWithData.size(); i++)
        {
            if (debug)
                cv::polylines(image, validContoursWithData[i].contour, true, colors[6], 1);

            cv::Rect r = validContoursWithData[i].rotRect.boundingRect();

            int rdim = abs(r.width - r.height) < r.width / 2 ? (r.width + r.height) / 2 : 0;

            if ((rdim > mindim) && (rdim < maxdim))
            {
                cv::Point center_of_rect = (r.br() + r.tl()) * 0.5;
                pts.push_back(center_of_rect);
                r.x = center_of_rect.x - (rdim / 2);
                r.y = center_of_rect.y - (rdim / 2);
                r.width = r.height = rdim;
                pts.push_back(r.br());
                pts.push_back(r.tl());
                rects.push_back(r);

                if (debug)
                    cv::polylines(image, validContoursWithData[i].contour, true, colors[5], 1);
            }
        }


        if (pts.size() > 60)
        {
            RRect = cv::minAreaRect(pts);
            adjustRotatedRect(RRect);
            return true;
        }

        return false;
    }

    bool HopeOMr::drawRects()
    {
        return getRRectParts();
    }

    cv::RotatedRect HopeOMr::getRRectPart(std::vector<cv::Point> contour)
    {
        //Point contour_center = (contour[0] + contour[2]) * 0.5;
        std::vector<cv::Point> pts;
        std::vector<cv::Point> hull;
        std::vector <std::vector<cv::Point>> hullpts;

        for (size_t i = 0; i < rects.size(); i++)
        {
            double d = cv::pointPolygonTest(contour, rects[i].br(), false);


            if (d > 0)
            {
                cv::Point pt_center = (rects[i].br() + rects[i].tl()) * 0.5;
                pts.push_back(pt_center);
            }
        }

        vec_pts.push_back(pts);

        if (pts.size() > 0)
        {
            cv::convexHull(pts, hull, true);
            cv::Point center_pt = getCentroid(hull);

            double min_dist = INT16_MAX;
            cv::Point nearest_pt;

            for (size_t i = 0; i < pts.size(); i++)
            {
                double dist = norm(center_pt - pts[i]);
                if (dist < min_dist)
                {
                    min_dist = dist;
                    nearest_pt = pts[i];
                }
            }

            center_pt = nearest_pt;
            min_dist = 0;
            for (size_t i = 0; i < hull.size(); i++)
            {
                double dist = cv::norm(center_pt - pts[i]);
                if (dist > min_dist)
                {
                    min_dist = dist;
                    nearest_pt = pts[i];
                }
            }

            cv::RotatedRect r = cv::minAreaRect(hull);
            adjustRotatedRect(r);

            r.size.width = (float)(r.size.width * 1.03);
            r.size.height = (float)(r.size.height * 1.15);

            if (debug)
                drawRotatedRect(image, r);

            return r;
        }
        return cv::RotatedRect();
    }

    bool HopeOMr::getRRectParts()
    {
        bool result = false;
        std::vector<cv::Point> contour4;
        std::vector<cv::Point> contour3;
        std::vector<cv::Point> contour2;
        std::vector<cv::Point> contour1;

        cv::RotatedRect RRect2 = RRect;

        cv::Point2f vtx[4];
        RRect.size += cv::Size2f(10, 10);
        RRect.size.height += 10;
        RRect.size.width += 10;

        RRect.points(vtx);

        contour4.push_back(vtx[1]);
        contour4.push_back(vtx[2]);

        contour1.push_back(vtx[0]);
        contour1.push_back(vtx[3]);

        RRect.size.height /= 2;
        RRect.size.height += 5;
        RRect.points(vtx);

        contour4.push_back(vtx[2]);
        contour4.push_back(vtx[1]);

        contour1.push_back(vtx[3]);
        contour1.push_back(vtx[0]);

        contour2.push_back(vtx[3]);
        contour2.push_back(vtx[0]);

        contour3.push_back(vtx[2]);
        contour3.push_back(vtx[1]);

        RRect.size.height = 4;
        RRect.points(vtx);

        contour2.push_back(vtx[0]);
        contour2.push_back(vtx[3]);

        contour3.push_back(vtx[1]);
        contour3.push_back(vtx[2]);

        std::vector<cv::RotatedRect> rr;
        rr.push_back(getRRectPart(contour4));
        rr.push_back(getRRectPart(contour3));
        rr.push_back(getRRectPart(contour2));
        rr.push_back(getRRectPart(contour1));

        float h1 = rr[0].size.height;
        float h2 = rr[1].size.height;
        float h3 = rr[2].size.height;
        float h4 = rr[3].size.height;

        float a1 = rr[0].angle;
        float a2 = rr[1].angle;
        float a3 = rr[2].angle;
        float a4 = rr[3].angle;

        int found_circle_count = 0;

        if ((abs(h1 + h2 - h3 - h4) < 10) & (abs(a1 + a2 - a3 - a4) < 3))
        {

            for (int i = 0; i < 4; i++)
            {
                cv::Rect crect = rr[i].boundingRect() & cv::Rect(0, 0, image.cols, image.rows);

                cv::Mat br1 = thresh0(crect);
                cv::Mat display = image(crect);

                int mindim = br1.cols / 60;
                int maxdim = mindim * 2;

                std::vector<std::vector<cv::Point> > contours;

                cv::findContours(br1, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

                cv::Mat m(display.size(), CV_8U, cv::Scalar(0));

                for (size_t j = 0; j < contours.size(); j++)
                {
                    cv::Rect r = cv::boundingRect(contours[j]);

                    int rdim = abs(r.width - r.height) < r.width / 2 ? (r.width + r.height) / 2 : 0;

                    if ((rdim > mindim) && (rdim < maxdim))
                    {
                        cv::rectangle(m, r, cv::Scalar(255), -1);
                    }
                }

                cv::findContours(m, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
                std::sort(contours.begin(), contours.end(), contour_sorter());

                if (contours.size() > 150)
                {
                    drawRotatedRect(image, rr[i], colors[3], 2);
                    found_circle_count++;
                }
            }
            result = found_circle_count > 3;
        }
        return result;
    }

    bool doHopeOMr(cv::InputOutputArray src, int method)
    {
        cv::Mat img = src.getMat();
        int x = img.cols / 10;
        int y = img.rows / 8;
        cv::Rect c_area(x, y, img.cols - (x * 2), img.rows - (y * 2));

        cv::Mat image = img(c_area);
        HopeOMr omr(image);
        omr.debug = method;

        image = image + cv::Scalar(15, 20, 50);

        if (omr.getRects())
        {
            return omr.drawRects();
        }

        return false;
    }

static int draw_unsupported(cv::Mat& rgb)
{
    const char text[] = "unsupported";

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 1.0, 1, &baseLine);

    int y = (rgb.rows - label_size.height) / 2;
    int x = (rgb.cols - label_size.width) / 2;

    cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                    cv::Scalar(255, 255, 255), -1);

    cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0));

    return 0;
}

static int draw_fps(cv::Mat& rgb)
{
    // resolve moving average
    float avg_fps = 0.f;
    {
        static double t0 = 0.f;
        static float fps_history[10] = {0.f};

        double t1 = ncnn::get_current_time();
        if (t0 == 0.f)
        {
            t0 = t1;
            return 0;
        }

        float fps = 1000.f / (t1 - t0);
        t0 = t1;

        for (int i = 9; i >= 1; i--)
        {
            fps_history[i] = fps_history[i - 1];
        }
        fps_history[0] = fps;

        if (fps_history[9] == 0.f)
        {
            return 0;
        }

        for (int i = 0; i < 10; i++)
        {
            avg_fps += fps_history[i];
        }
        avg_fps /= 10.f;
    }

    char text[32];
    sprintf(text, "FPS=%.2f", avg_fps);

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    int y = 0;
    int x = rgb.cols - label_size.width;

    cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                    cv::Scalar(255, 255, 255), -1);

    cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

    return 0;
}

static SCRFD* g_scrfd = 0;
static ncnn::Mutex lock;

class MyNdkCamera : public NdkCameraWindow
{
public:
    virtual void on_image_render(cv::Mat& rgb) const;
};

void MyNdkCamera::on_image_render(cv::Mat& rgb) const
{
    // scrfd
    {
        ncnn::MutexLockGuard g(lock);

        if (!g_scrfd)
        {
            std::vector<FaceObject> faceobjects;
            g_scrfd->detect(rgb, faceobjects);

            g_scrfd->draw(rgb, faceobjects);
        }
        else
        {
            doHopeOMr(rgb,1);
        }
    }

    draw_fps(rgb);
}

static MyNdkCamera* g_camera = 0;

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnLoad");

    g_camera = new MyNdkCamera;

    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnUnload");

    {
        ncnn::MutexLockGuard g(lock);

        delete g_scrfd;
        g_scrfd = 0;
    }

    delete g_camera;
    g_camera = 0;
}

// public native boolean loadModel(AssetManager mgr, int modelid, int cpugpu);
JNIEXPORT jboolean JNICALL Java_com_tencent_scrfdncnn_SCRFDNcnn_loadModel(JNIEnv* env, jobject thiz, jobject assetManager, jint modelid, jint cpugpu)
{
    if (modelid < 0 || modelid > 7 || cpugpu < 0 || cpugpu > 1)
    {
        return JNI_FALSE;
    }

    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "loadModel %p", mgr);

    const char* modeltypes[] =
    {
        "500m",
        "500m_kps",
        "1g",
        "2.5g",
        "2.5g_kps",
        "10g",
        "10g_kps",
        "34g"
    };

    const char* modeltype = modeltypes[(int)modelid];
    bool use_gpu = (int)cpugpu == 1;

    // reload
    {
        ncnn::MutexLockGuard g(lock);

        if (use_gpu && ncnn::get_gpu_count() == 0)
        {
            // no gpu
            delete g_scrfd;
            g_scrfd = 0;
        }
        else
        {
            if (!g_scrfd)
                g_scrfd = new SCRFD;
            g_scrfd->load(mgr, modeltype, use_gpu);
        }
    }

    return JNI_TRUE;
}

// public native boolean openCamera(int facing);
JNIEXPORT jboolean JNICALL Java_com_tencent_scrfdncnn_SCRFDNcnn_openCamera(JNIEnv* env, jobject thiz, jint facing)
{
    if (facing < 0 || facing > 1)
        return JNI_FALSE;

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "openCamera %d", facing);

    g_camera->open((int)facing);

    return JNI_TRUE;
}

// public native boolean closeCamera();
JNIEXPORT jboolean JNICALL Java_com_tencent_scrfdncnn_SCRFDNcnn_closeCamera(JNIEnv* env, jobject thiz)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "closeCamera");

    g_camera->close();

    return JNI_TRUE;
}

// public native boolean setOutputWindow(Surface surface);
JNIEXPORT jboolean JNICALL Java_com_tencent_scrfdncnn_SCRFDNcnn_setOutputWindow(JNIEnv* env, jobject thiz, jobject surface)
{
    ANativeWindow* win = ANativeWindow_fromSurface(env, surface);

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "setOutputWindow %p", win);

    g_camera->set_window(win);

    return JNI_TRUE;
}

}
