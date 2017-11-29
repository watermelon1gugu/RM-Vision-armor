//
// Created by arch1tect_leaves on 17-11-25.
//
#include <vector>
#include <opencv/cv.hpp>
#include <iostream>
#include <Eigen/Dense>
#include "RobotCatcher.h"

#define MaxBuffSize 5;

using namespace cv;

class RoboCatcher {
private:


    Point2f buffPoint[5] = {};
    Mat *currentMat = new Mat;
    Eigen::MatrixXd *Xtrajectory = new Eigen::MatrixXd;//X轨迹
    Eigen::MatrixXd *Ytrajectory = new Eigen::MatrixXd;//Y轨迹
    int index = -1;
    int count = 0;
    int missNum = 0;

    float getMax(float a, float b) {
        if (a > b)
            return a;
        else
            return b;
    }

    float getMin(float a, float b) {
        if (a > b) {
            return b;
        } else return a;
    }

    float division(float a, float b) {
        if (a > b) {
            return b / a;
        }
        return a / b;
    }

    double mu(double number, int n) {
        if (n == 0) {
            return 1;
        } else {
            double temp = number;
            for (int i = 1; i < n; i++) {
                number *= temp;
            }
            return number;
        }
    }

    Point2f trajectoryPredict(int n) {
        Point2f result;
        if (n == 5) {

            Eigen::VectorXd vectorXd(5);
            vectorXd << 1, 2, 3, 4, 5;//默认值
            Eigen::VectorXd vectorY(5);//Y坐标记录
            Eigen::VectorXd vectorX(5);//X坐标记录
            Eigen::MatrixXd matrixX(3, 3);
            Eigen::MatrixXd matrixY(3, 3);
            Eigen::VectorXd vectorBX(3);
            Eigen::VectorXd vectorBY(3);
            for (int i = 0; i < 5; i++) {
                vectorX[i] = buffPoint[index].x;
                vectorY[i] = buffPoint[index].y;
                index++;
                index %= 5;
            }
            for (int i = 0; i < 3; i++) {
                double sumX = 0;
                double sumY = 0;
                for (int j = 0; j < 5; j++) {
                    sumX += vectorX[j] * mu(vectorXd[j], i);
                    sumY += vectorY[j] * mu(vectorXd[j], i);
                }
                vectorBX[i] = sumX;
                vectorBY[i] = sumY;
            }
            for (int r = 0; r < 3; r++) {
                for (int c = 0; c < 3; c++) {
                    double sumX = 0;
                    double sumY = 0;
                    for (int i = 0; i < 5; i++) {
                        sumX += mu(vectorXd[i], r + c);
                        sumY += mu(vectorXd[i], r + c);
                    }
                    matrixX(r, c) = sumX;
                    matrixY(r, c) = sumY;
                }
            }

            *Xtrajectory = leastSquare(matrixX,vectorBX);
            *Ytrajectory = leastSquare(matrixY,vectorBY);
        }
        result.x = (*Xtrajectory)(0, 0) + (*Xtrajectory)(1, 0) * n + (*Xtrajectory)(2, 0) * (n) * (n);
        result.y = (*Ytrajectory)(0, 0) + (*Ytrajectory)(1, 0) * n + (*Ytrajectory)(2, 0) * (n) * (n);
        std::cout << "-----------------------------------------------" << std::endl;
        return result;
    }

    Eigen::VectorXd leastSquare(Eigen::MatrixXd x, Eigen::VectorXd b) {


        return ((x * x.transpose()).inverse()) * x.transpose() * b;


    }

    void findContours(Mat mat, std::vector<std::vector<Point>> contours) {
        cv::findContours(mat, contours, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
    }

    void binaryzation(Mat *mat, int lowerLimit, int topLimit) {
        cv::cvtColor(*mat, *mat, CV_BGR2GRAY);
        cv::threshold(*mat, *mat, lowerLimit, topLimit, CV_THRESH_BINARY);
    }

    void shape(Mat *mat, int x, int y, int width, int height) {
        cv::Rect rect(x, y, width, height);
        (*mat)(rect);
    }

    //中值滤波
    void medianBlur(Mat *mat, int ksize) {
        cv::medianBlur(*mat, *mat, ksize);
    }

    double getLength(Point2f a, Point2f b) {
        return sqrt((a.x - b.x) * (a.x - b.x) +
                    (a.y - b.y) * (a.y - b.y));
    }

    void getRotatedRects(Mat mat, std::vector<RotatedRect> rotatedRects, std::vector<RotatedRect> vertical,
                         std::vector<RotatedRect> horizon) {
        std::vector<std::vector<Point>> contours;//轮廓容器
        binaryzation(&mat, 254, 255);//二值化
        medianBlur(&mat, 5);//中值滤波
        findContours(mat, contours);
        /* 遍历所有可疑轮廓
         * 通过角度 长宽比 面积 进行分类*/
        for (int i = 0; i < contours.size(); i++) {
            RotatedRect temp = minAreaRect(Mat(contours[i]));
            if ((temp.angle < -45 && temp.size.width > temp.size.height &&
                 temp.size.height * temp.size.width >= 75 && temp.size.width / temp.size.height > 1.7)
                || (temp.angle > -45 && temp.size.width < temp.size.height &&
                    temp.size.height * temp.size.width >= 75 && temp.size.height / temp.size.width > 1.7)) {
                vertical.push_back(temp);
                temp.center.x += mat.cols;
                rotatedRects.push_back(temp);
            } else if (temp.size.height * temp.size.width >= 100 &&
                       (temp.size.width / temp.size.height > 2 || temp.size.height / temp.size.width > 2)) {
                horizon.push_back(temp);
                //temp.center.x += mat.cols;//测试中存在切割
                rotatedRects.push_back(temp);
            }
        }
    }

    /*获取竖线*/
    std::vector<RotatedRect> getVertical(Mat mat, std::vector<RotatedRect> vertical) {
        std::vector<std::vector<Point>> contours;
        cv::findContours(mat, contours, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
        for (int i = 0; i < contours.size(); i++) {
            RotatedRect temp = minAreaRect(Mat(contours[i]));
            if ((temp.angle < -45 && temp.size.width > temp.size.height &&
                 temp.size.height * temp.size.width >= 75 && temp.size.width / temp.size.height > 1)
                || (temp.angle > -45 && temp.size.width < temp.size.height &&
                    temp.size.height * temp.size.width >= 75 && temp.size.height / temp.size.width > 1)) {
                temp.center.x += mat.cols;//测试中存在切割
                vertical.push_back(temp);
            }
        }
        return vertical;
    }

public:
    RoboCatcher() {

    }

    RoboCatcher(Mat mat) {
        setCurrentMat(mat);
    }

    Point2f getTargetPoint() {
        int min = 0;
        Point2f result;
        std::vector<RotatedRect> vertical;
        vertical = getVertical(*currentMat, vertical);
        for (int i = 0; i < vertical.size(); i++) {
            for (int j = i + 1; j < vertical.size(); j++) {

                float distance = getLength(vertical[i].center, vertical[j].center);
                if (abs(vertical[i].center.y - vertical[j].center.y) <
                    (vertical[i].size.height + vertical[j].size.height) / 2
                    && ((getMax(vertical[i].size.width, vertical[i].size.height) +
                         getMax(vertical[j].size.width, vertical[j].size.height)) / 2 > 0.4 * distance
                        && (getMax(vertical[i].size.width, vertical[i].size.height) +
                            getMax(vertical[j].size.width, vertical[j].size.height)) / 2 < 0.6 * distance
                    )) {
                    float similar = division(getMax(vertical[i].size.height, vertical[i].size.width),
                                             getMax(vertical[j].size.height, vertical[j].size.width))
                                    * division(getMin(vertical[i].size.height, vertical[i].size.width),
                                               getMin(vertical[j].size.height, vertical[j].size.width));

                    if (similar > min) {
                        min = similar;
                        Point2f point2f1((vertical[i].center.x + vertical[j].center.x) / 2 /*+ buffMat.cols*/,
                                         (vertical[i].center.y + vertical[j].center.y) / 2);
                        result = point2f1;

                    }
                }
            }
        }
        //判断结果输出
        Point2f point2f;
        if ((result.x == 0 && result.y == 0)) {//目标丢失
            missNum++;
            if (count < 5) {//不可预测
                if (count != 0 && count < 5) {//前一个点有值
                    count = 0;
                    return buffPoint[index];
                    //return point2f;
                } else {
                    return result;
                    //return point2f;
                }
            } else {
                if (missNum <= 5) {
                    result = trajectoryPredict(count);
                    buffPoint[index] = result;
                    count = 5;
                    index++;
                    index %= 5;

                    return result;
                } else
                    count = 0;
                return result;
            }
        } else {
            missNum = 0;
            if (count < 5) {
                count++;
            }


            buffPoint[index] = result;
            index++;
            index %= 5;
            return result;
            //return point2f;
        }
    }


    void setCurrentMat(Mat mat) {

        *currentMat = mat;
        binaryzation(currentMat, 254, 255);//二值化
        medianBlur(&mat, 5);//中值滤波
    }


};

int main(void) {
    RoboCatcher *roboCatcher = new RoboCatcher();
    // 读取视频流
    cv::VideoCapture capture("/home/arch1tect_leaves/桌面/视觉/vision/1.mov");
    // 检测视频是否读取成功
    if (!capture.isOpened()) {
        std::cout << "No Input Image";
        return 1;
    }

    // 获取图像帧率
    double rate = capture.get(CV_CAP_PROP_FPS);
    bool stop(false);
    cv::Mat frame; // 当前视频帧
    cv::namedWindow("Extracted Frame");

    // 每一帧之间的延迟
    int delay = 1000 / rate;

    // 遍历每一帧
    int i = 0;
    while (!stop) {
        // 尝试读取下一帧
        if (!capture.read(frame))
            break;

        // 引入延迟
        cv::Rect rect(frame.cols / 2, 0, frame.cols / 2, frame.rows);
        frame = frame(rect);


        roboCatcher->setCurrentMat(frame);
        Point2f point2f = roboCatcher->getTargetPoint();
        point2f.x -= frame.cols;
        i++;
        std::cout << i << point2f << std::endl;
        circle(frame, point2f, 10, CV_RGB(0, 0, 255), 2, 8, 0);
        imshow("Extracted Frame", frame);


        if (cv::waitKey(delay) >= 0)
            stop = true;
    }

    return 0;
}

int eeem(void) {
    Eigen::MatrixXd matrixXd(2, 2);
    matrixXd << 1, 2, 3, 4;
    std::cout << matrixXd << std::endl;
    std::cout << matrixXd.transpose() << std::endl;


}