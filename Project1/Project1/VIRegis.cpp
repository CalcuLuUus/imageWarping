#include "VIRegis.h"
#include "neon_math.h"
#include <algorithm>
#include <cmath>
#include <map>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

#include "Utils.h"


#define for_indexed(...) for_indexed_v(i, __VA_ARGS__)
using namespace cv;
using namespace std;

/*
global data
*/
const int SeqId = 5;
//window size
const int window_size = 80;

//thres of edge proposals
const int thres = 200;

//overlapped len of bbox
const int overlap_len = 25;

const float matching_thres = 0.3;
const float nms_thres = 0.2;
const float nms_resampling_thres = 0.1;
const float tps_lambda = 0.2;

//the scaling factor for a proposal, when searching in the background
const float bbox_scaling_factor = 1.2;



namespace TPS {
    /*
    Thin plate spline
    */
    const float eps = 1e-6;
    const int wa_split_pivot = 3;


    void unsq_l2_dist_dim2(const float *a, const float *b, float * dst, int n, int m) {

        // a.shape, b.shape -> [n, 3], [m, 3]
        // dst.shape -> [n, m]
        // dim m is assumed to be the smaller dim

        for (uint32_t i = 0; i < n; ++ i){

            int dim4m = m >> 2, left4m = m & 3;
            float32x4_t an_v_0 = vdupq_n_f32(*a);
            float a0 = *(a++);
            float32x4_t an_v_1 = vdupq_n_f32(*a);
            float a1 = *(a++);
            for (; dim4m > 0; --dim4m, dst += 4, b += 8){
                float32x4x2_t bn_v = vld2q_f32(b);
                float32x4_t tmp_0 = vsubq_f32(an_v_0, bn_v.val[0]);
                float32x4_t tmp_1 = vsubq_f32(an_v_1, bn_v.val[1]);
                tmp_0 = vmulq_f32(tmp_0, tmp_0);
                tmp_1 = vmulq_f32(tmp_1, tmp_1);
                float32x4_t ret = vaddq_f32(tmp_0, tmp_1);
                vst1q_f32(dst, ret);
            }

            for (;left4m > 0; -- left4m){ // b.shape = [m, 3] thus b += 3
                float b0 = *(b++);
                float b1 = *(b++);
                *(dst++) = (a0 - b0) * (a0 - b0) + (a1 - b1) * (a1 - b1);
            }
            b -= 2 * m;
        }

    }

    void unsq_l2_dist_dim3(const float *a, const float *b, float * dst, int n, int m) {

        // a.shape, b.shape -> [n, 3], [m, 3]
        // dst.shape -> [n, m]
        // dim m is assumed to be the smaller dim


        for (uint32_t i = 0; i < n; ++ i, ++ a){
            int dim4m = m >> 2, left4m = m & 3;
            float32x4_t an_v_0 = vdupq_n_f32(*a);
            float a0 = *(a++);
            float32x4_t an_v_1 = vdupq_n_f32(*a);
            float a1 = *(a++);
            for (; dim4m > 0; --dim4m, dst += 4, b += 12){
                float32x4x3_t bn_v = vld3q_f32(b);
                float32x4_t tmp_0 = vsubq_f32(an_v_0, bn_v.val[0]);
                float32x4_t tmp_1 = vsubq_f32(an_v_1, bn_v.val[1]);
                tmp_0 = vmulq_f32(tmp_0, tmp_0);
                tmp_1 = vmulq_f32(tmp_1, tmp_1);
                float32x4_t ret = vaddq_f32(tmp_0, tmp_1);
                vst1q_f32(dst, ret);
            }

            for (;left4m > 0; -- left4m, ++ b){ // b.shape = [m, 3] thus b += 3
                float b0 = *(b++);
                float b1 = *(b++);
                *(dst++) = (a0 - b0) * (a0 - b0) + (a1 - b1) * (a1 - b1);
            }
            b -= 3 * m;
        }

    }

     void wa_split(const float *src, float *dist_w, float &a0, float &a1, float &a2, int n) {
        // src and dist_w are assumed to be continuous;
        memcpy(dist_w, src, (n - 3) * sizeof(float));
        a0 = *(src + n - 3);
        a1 = *(src + n - 2);
        a2 = *(src + n - 1);
    }

    float sum_1d(const float *src, int n) {
        // src is assumed to be continous, thus we use arm_neon to accelerate
        int dim4 = n >> 2, left4 = n & 3;
        float32x4_t sum_vec = vdupq_n_f32(0.0);
        for (; dim4 > 0; dim4--, src += 4) {
            float32x4_t data_vec = vld1q_f32(src);
            sum_vec = vaddq_f32(sum_vec, data_vec);
        }

        float sum = vgetq_lane_f32(sum_vec, 0) + vgetq_lane_f32(sum_vec, 1) +
                    vgetq_lane_f32(sum_vec, 2) + vgetq_lane_f32(sum_vec, 3);
        for (; left4 > 0; left4--, src++) sum += (*src);
        return sum;
    }

     void reproduce_w(const float *src, float *dst, int n) {
        // assume src & dst are continuous
        *dst = -sum_1d(src, n - 1);
        memcpy(dst + 1, src, (n - 1) * sizeof(float));
    }

    void inplace_sum_log(const float *src, float * dst, int n, int m){
        int len = n * m;
        int dim4 = len >> 2, rmd = len & 3;

        float32x4_t half = vdupq_n_f32(0.5);
        float32x4_t eps_v = vdupq_n_f32(eps);
        for (; dim4 > 0; dim4--, src += 4, dst += 4) {
            float32x4_t src_v = vld1q_f32(src);
            float32x4_t log_v = log_ps(vaddq_f32(src_v, eps_v));
            src_v = vmulq_f32(vmulq_f32(src_v, log_v), half);
            vst1q_f32(dst, src_v);
        }
        for (; rmd > 0; rmd--){
            float log_t = log(*(src) + eps);
            *(dst++) = 0.5f * (*src ++) * log_t;
        }
    }

    cv::Mat du(const cv::Mat &src_x, const cv::Mat &src_c, int n, int m) {
        cv::Mat dst;
        dst.create(cv::Size(m, n), CV_32FC1);
        if (src_x.size[1] == 2){
            unsq_l2_dist_dim2(src_x.ptr<float>(0), src_c.ptr<float>(0), dst.ptr<float>(0), n, m);

        }else {
            unsq_l2_dist_dim3(src_x.ptr<float>(0), src_c.ptr<float>(0), dst.ptr<float>(0), n, m);
        }
        cv::Mat ret;
        ret.create(cv::Size(m, n), CV_32FC1);
        inplace_sum_log(dst.ptr<float>(0), ret.ptr<float>(0), n, m);
        return ret;
    }

    void z_split_src_x(const float *src, float *dst_1, float *dst_2, int n) {
        // src and dsts are assumed to be continuous.
        int dim4 = n >> 2, rmd = n & 3;


        for (; dim4 > 0; dim4--, src += 8, dst_1 += 4, dst_2 += 4) {
            float32x4x2_t src_v = vld2q_f32(src);
            vst1q_f32(dst_1, src_v.val[0]);
            vst1q_f32(dst_2, src_v.val[1]);
        }
        for (; rmd > 0; rmd--){
            (*dst_1++) = (*src ++);
            (*dst_2++) = (*src ++);
        }
    }

    cv::Mat z(const cv::Mat &src_x, const cv::Mat &src_c, const cv::Mat &theta) {
        int n = src_x.rows, m = src_c.rows;
        cv::Mat U = du(src_x, src_c, n, m);

        int theta_qty = theta.size[0];
        cv::Mat w;
        w.create(cv::Size(1, theta_qty - 3), CV_32FC1);
        float a0, a1, a2;
        wa_split(theta.ptr<float>(0), w.ptr<float>(0), a0, a1, a2, theta_qty);

        bool reduced = theta_qty == src_c.size[0] + 2;

        if (reduced) {
            cv::Mat w_reduced;
            w_reduced.create(cv::Size(1, theta_qty - 2), CV_32FC1);
            reproduce_w(w.ptr<float>(0), w_reduced.ptr<float>(0), theta_qty - 2);

            cv::Mat b = U * w_reduced;

            cv::Mat xs_0, xs_1;
            xs_0.create(n, 1, CV_32FC1);
            xs_1.create(n, 1, CV_32FC1);
            z_split_src_x(src_x.ptr<float>(0), xs_0.ptr<float>(0), xs_1.ptr<float>(0), n);
            return a0 + a1 * xs_0 + a2 * xs_1 + b;
        } else {
            cv::Mat b = U * w;
            cv::Mat xs_0, xs_1;
            xs_0.create(n, 1, CV_32FC1);
            xs_1.create(n, 1, CV_32FC1);
            z_split_src_x(src_x.ptr<float>(0), xs_0.ptr<float>(0), xs_1.ptr<float>(0), n);
            return a0 + a1 * xs_0 + a2 * xs_1 + b;
        }
    }

    void split_pv(const cv::Mat &src, cv::Mat &dst_p, cv::Mat &dst_v, int n) {
        for (uint32_t i = 0; i < n; ++i) {
            dst_p.at<float>(i, 0) = 1;
            //vst1_f16_x3()
            dst_p.at<float>(i, 1) = src.at<float>(i, 0);
            dst_p.at<float>(i, 2) = src.at<float>(i, 1);

            dst_v.at<float>(i, 0) = src.at<float>(i, 2);
        }
    }


    void construct_A(const cv::Mat &src_k, const cv::Mat &src_p, cv::Mat &dst, int n) {
        // dst is assumed to be a [n + 3, n + 3] all-zero matrix

        const int k_len = n + 3;


        for (uint32_t i = 0; i < n; ++i) {
            for (uint32_t j = 0; j < n; ++j) {
                dst.at<float>(i, j) = src_k.at<float>(i, j);
            }
        }
        for (uint32_t i = 0; i < n; ++i) {
            dst.at<float>(i, n) = src_p.at<float>(i, 0);
            dst.at<float>(i, n + 1) = src_p.at<float>(i, 1);
            dst.at<float>(i, n + 2) = src_p.at<float>(i, 2);

            dst.at<float>(n, i) = src_p.at<float>(i, 0);
            dst.at<float>(n + 1, i) = src_p.at<float>(i, 1);
            dst.at<float>(n + 2, i) = src_p.at<float>(i, 2);
        }

    }

    void reproduce_theta(const float *src, float *dst, int n) {
        //src and dst are assumed to be continuous
        memcpy(dst, src + 1, sizeof(float) * n);

    }

    cv::Mat fit(const cv::Mat &c, float lambd, bool reduced) {

        int n = c.size[0];

        cv::Mat U = du(c, c, n, n);
        cv::Mat K = U + lambd * cv::Mat::eye(cv::Size(n, n), CV_32FC1);
        cv::Mat A = cv::Mat::zeros(cv::Size(n + 3, n + 3), CV_32FC1);
        cv::Mat P = cv::Mat::ones(cv::Size(3, n), CV_32FC1);
        cv::Mat v = cv::Mat::zeros(cv::Size(1, n + 3), CV_32FC1);
        split_pv(c, P, v, n);
        construct_A(K, P, A, n);
        cv::Mat theta = cv::Mat::zeros(cv::Size(1, n + 3), CV_32FC1);
        bool ret = cv::solve(A, v, theta, 0);
        if (reduced) {
            cv::Mat theta_reduced = cv::Mat::zeros(cv::Size(1, n + 2), CV_32FC1);
            reproduce_theta(theta.ptr<float>(0), theta_reduced.ptr<float>(0), n + 2);
            return theta_reduced;
        } else return theta;
    }
};

void uniform_grid(cv::Mat &dst, int h, int w) {
    for (uint32_t j = 0; j < w; ++j) dst.at<float>(j, 0) = j / float(w - 1);
    for (uint32_t i = 1; i < h; ++i) {
        for (uint32_t j = 0; j < w; ++j) {
            dst.at<float>(i * w + j, 0) = dst.at<float>(j, 0);
        }
    }
    for (uint32_t i = 0; i < h; ++i) dst.at<float>(i * w, 1) = i / float(h - 1);
    for (uint32_t i = 0; i < h; ++i) {
        for (uint32_t j = 1; j < w; ++j) {
            dst.at<float>(i * w + j, 1) = dst.at<float>(i * w, 1);
        }
    }
}

void split_theta(const cv::Mat &src, cv::Mat &dst_1, cv::Mat &dst_2, int n) {
    for (uint32_t i = 0; i < n; ++i) {
        dst_1.at<float>(i, 0) = src.at<float>(i, 0);
        dst_2.at<float>(i, 0) = src.at<float>(i, 1);
    }
}

void tps_grid_merge_dgrid(const cv::Mat &src_1, const cv::Mat &src_2, cv::Mat &dst, int n) {
    for (int i = 0; i < n; ++i) {
        dst.at<float>(i, 0) = src_1.at<float>(i, 0);
        dst.at<float>(i, 1) = src_2.at<float>(i, 0);
    }
}

cv::Mat tps_grid(const cv::Mat &theta, const cv::Mat &c_dst, int shape_x, int shape_y) {
    //bool reduced = c_dst.size[0] + 2 == theta.size[0];
    cv::Mat ugrid;
    ugrid.create(cv::Size(2, shape_x * shape_y), CV_32FC1);
    uniform_grid(ugrid, shape_x, shape_y);

    cv::Mat theta_1, theta_2;
    theta_1.create(cv::Size(1, theta.size[0]), CV_32FC1);
    theta_2.create(cv::Size(1, theta.size[0]), CV_32FC1);
    split_theta(theta, theta_1, theta_2, theta.size[0]);
    cv::Mat dgrid;
    dgrid.create(cv::Size(2, shape_y * shape_x), CV_32FC1);

    cv::Mat dgrid_1 = TPS::z(ugrid, c_dst, theta_1), dgrid_2 = TPS::z(ugrid, c_dst, theta_2);
    tps_grid_merge_dgrid(dgrid_1, dgrid_2, dgrid, dgrid_1.size[0]);
    cv::Mat grid = dgrid + ugrid;
    return grid; //* 2 - 1;
}

void column_stack_split_last(const cv::Mat &src_c, const cv::Mat &src_d,
                             cv::Mat &dst_0, cv::Mat &dst_1, int n) {

    for (uint32_t i = 0; i < n; ++i) {
        dst_0.at<float>(i, 0) = src_c.at<float>(i, 0);
        dst_0.at<float>(i, 1) = src_c.at<float>(i, 1);
        dst_0.at<float>(i, 2) = src_d.at<float>(i, 0);

        dst_1.at<float>(i, 0) = src_c.at<float>(i, 0);
        dst_1.at<float>(i, 1) = src_c.at<float>(i, 1);
        dst_1.at<float>(i, 2) = src_d.at<float>(i, 1);
    }

}

void column_stack_split_last(const float *src_c, const float *src_d,
                             float *dst_0, float *dst_1, int n) {
    int dim4 = n >> 2, left4 = n & 3;
    for (; dim4 > 0; dim4--, src_c += 8, src_d += 8, dst_0 += 12, dst_1 += 12) {
        float32x4x2_t src_c_v = vld2q_f32(src_c);
        float32x4x2_t src_d_v = vld2q_f32(src_d);
        float32x4x3_t tgt_0 = {src_c_v.val[0], src_c_v.val[1], src_d_v.val[0]};
        float32x4x3_t tgt_1 = {src_c_v.val[0], src_c_v.val[1], src_d_v.val[1]};
        vst3q_f32(dst_0, tgt_0);
        vst3q_f32(dst_1, tgt_1);
    }
    for (;left4 > 0; --left4){
        *(dst_0 ++) = *(src_c);
        *(dst_1 ++) = *(src_c ++);
        *(dst_0 ++) = *(src_c);
        *(dst_1 ++) = *(src_c ++);
        *(dst_0 ++) = *(src_d ++);
        *(dst_1 ++) = *(src_d ++);
    }
}
//
//void merge_1cf(const cv::Mat &src_1, const cv::Mat &src_2, cv::Mat &dst, int n) {
//    int dim4 = n >> 2, left4 = n & 3;
//
//    for (uint32_t i = 0; i < n; ++i) {
//        dst.at<float>(i, 0) = src_1.at<float>(i, 0);
//        dst.at<float>(i, 1) = src_2.at<float>(i, 0);
//    }
//}

void merge_1cf(const float *src_1, const float *src_2, float *dst, int n) {
    int dim4 = n >> 2, left4 = n & 3;

    for (;dim4 > 0; -- dim4, dst += 8, src_1 += 4, src_2 += 4){
        float32x4_t tmp_1 = vld1q_f32(src_1);
        float32x4_t tmp_2 = vld1q_f32(src_2);
        float32x4x2_t ret = {tmp_1, tmp_2};
        vst2q_f32(dst, ret);
    }
    for (; left4 > 0; left4--) {
        *(dst ++) = *(src_1 ++);
        *(dst ++) = *(src_2 ++);
    }
}


cv::Mat tps_theta_from_points(const cv::Mat &c_src, const cv::Mat &c_dst, float lambd, bool reduced) {
    cv::Mat delta = c_src - c_dst;
    int n = c_src.size[0];
    cv::Mat dx, dy;
    dx.create(cv::Size(3, n), CV_32FC1);
    dy.create(cv::Size(3, n), CV_32FC1);
    column_stack_split_last(c_dst.ptr<float>(0), delta.ptr<float>(0), dx.ptr<float>(0), dy.ptr<float>(0), n);

    cv::Mat ret;
    ret.create(cv::Size(2, n + 2), CV_32FC1);
    auto fit_l = TPS::fit(dx, lambd, reduced); //.ptr<float>(0);
    auto fit_r = TPS::fit(dy, lambd, reduced);
    merge_1cf(fit_l.ptr<float>(0), fit_r.ptr<float>(0), ret.ptr<float>(0), n + 2);
    return ret;
}

void split_to_remap(const cv::Mat &src, cv::Mat &dst_1, cv::Mat &dst_2, int n, int size_x, int size_y) {

    for (int i = 0; i < size_x; ++i) {
        for (int j = 0; j < size_y; ++j) {
            dst_1.at<float>(i, j) = src.at<float>(i * size_y + j, 0) * size_y;
            dst_2.at<float>(i, j) = src.at<float>(i * size_y + j, 1) * size_x;
        }
    }
}

void tps_grid_to_remap(const cv::Mat &grid, std::vector<cv::Mat> &dsts, int size_x, int size_y) {
    cv::Mat rx, ry;
    rx.create(cv::Size(size_y, size_x), CV_32FC1);
    ry.create(cv::Size(size_y, size_x), CV_32FC1);
    dsts.emplace_back(rx);
    dsts.emplace_back(ry);
    split_to_remap(grid, dsts[0], dsts[1], grid.size[0], size_x, size_y);
}


bool cmp(float a, float b) {
    return a > b;
}


vector<float> topk(const Mat &mat, const int &k) {
    vector<float> array;
    if (mat.isContinuous()) {
        array.assign(mat.datastart, mat.dataend);
    } else {
        for (int i = 0; i < mat.rows; ++i) {
            array.insert(array.end(), mat.ptr<float>(i), mat.ptr<float>(i) + mat.cols);
        }
    }

    sort(array.begin(), array.end(), cmp);
    if (array.size() > k) {
        vector<float> new_array;
        new_array.assign(array.begin(), array.begin() + k);
        return new_array;
    } else {
        return array;
    }


}

void multiScaleMatchTemplate(const cv::Mat &foreground, const cv::Mat &background, int num_scales,
                             float precision,           //default: num_scale=5, precision=0.1
                             float &scaleMaxVal, Point &scaleMaxLoc, Mat &scaleMaxTpl) {
    /*
    """
    Multi-scale template matching.
    Parameters
    ----------
    num_scales : the number of scales employing template matching.
    precision : the step between different scales.
    ----------
    """
    */

    float scale_start = 1.0 - precision * (num_scales / 2);
    float scale_end = 1.0 + precision * (num_scales / 2);
    float distance = (scale_end - scale_start) / (num_scales - 1);
    float val = scale_start;
    vector<float> scales;
    for (int cnt = 0; cnt < num_scales; val += distance, cnt++) {
        scales.push_back(val);
    }
    scaleMaxVal = -1;

    for (float scale: scales) {
        // resize the template using a certain scale value
        Mat tpl;
        resize(foreground, tpl, Size(min(int(foreground.size().width * scale), background.size().width),
                                     min(int(foreground.size().height * scale), background.size().height)));

        Mat result;
        matchTemplate(background, tpl, result, cv::TM_CCORR_NORMED);

        // Good matches are expected to be unique.
        vector<float> result_topk = topk(result, 5);
        float topk_diff = 0;
        for (int i = 1; i < result_topk.size(); ++i) {
            topk_diff += (abs(result_topk[i] - result_topk[i - 1]));
        }
        double minVal, maxVal;
        Point minLoc, maxLoc;
        minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
        if (maxVal > scaleMaxVal) {
            scaleMaxVal = maxVal;
            scaleMaxLoc = maxLoc;
            tpl.copyTo(scaleMaxTpl);
        }
    }
}

vector<int> argsort(const std::vector<float> &v, int st = -1) // default st = -1, st == -1 dec, st == 0 asc
{
    vector<pair<float, int>> tmp;
    for (int i = 0; i < v.size(); ++i) {
        tmp.emplace_back(make_pair(v[i], i));
    }
    if (st == 0)
        sort(tmp.begin(), tmp.end(), [](const std::pair<float, int> &a, const std::pair<float, int> &b) -> bool {
            return a.first != b.first ? a.first < b.first : a.second < b.second;
        });
    else {
        sort(tmp.begin(), tmp.end(), [](const std::pair<float, int> &a, const std::pair<float, int> &b) -> bool {
            return a.first != b.first ? a.first > b.first : a.second < b.second;
        });
    }
    vector<int> ret;
    for (int i = 0; i < v.size(); ++i) {
        ret.emplace_back(tmp[i].second);
    }

    return ret;
}

vector<int> nms(Mat &boxes, const std::vector<float> &scores, float nms_thr) {
    if (scores.size() == 0) return {};

    Mat x1 = boxes.col(0).clone().reshape(1, 1);
    Mat y1 = boxes.col(1).clone().reshape(1, 1);
    Mat x2 = boxes.col(2).clone().reshape(1, 1);
    Mat y2 = boxes.col(3).clone().reshape(1, 1);

    Mat areas = (x2 - x1 + 1).mul(y2 - y1 + 1);

    vector<int> order = argsort(scores, -1);

    map<int, int> mp_id; //map[idx] -> id_of_bbox
    for (int i = 0; i < order.size(); ++i) {
        mp_id[i] = order[i];
    }

    vector<int> keep;
    while (mp_id.size() > 0) {
        int i = (mp_id.begin()->second);
        keep.emplace_back(i);
        mp_id.erase(mp_id.begin());

        int val_x1 = x1.at<int>(i), val_y1 = y1.at<int>(i), val_x2 = x2.at<int>(i), val_y2 = y2.at<int>(i);
        for (auto it = mp_id.begin(); it != mp_id.end();) {
            int xx1 = 0, yy1 = 0, xx2 = 0, yy2 = 0;
            int j = it->second; //id_of_bbox
            xx1 = max(val_x1, x1.at<int>(j));
            yy1 = max(val_y1, y1.at<int>(j));
            xx2 = min(val_x2, x2.at<int>(j));
            yy2 = min(val_y2, y2.at<int>(j));

            int w = max(0, xx2 - xx1 + 1);
            int h = max(0, yy2 - yy1 + 1);
            int inter = w * h;
            int area1 = areas.at<int>(i), area2 = areas.at<int>(j);
            double ovr = (double) inter / (double) (area1 + area2 - inter);

            if (ovr > nms_thr) {
                it = mp_id.erase(it);
            } else {
                it++;
            }
        }
    }

    return keep;
}


Mat gen_bboxes_from_edge_nms(const cv::Mat &edge_map) {
    /*
        Generating a bbox at the center of an edge pixel(`thres` as the threshold).
        The window size is determined by `ws`.
    */
    int H = edge_map.size().height, W = edge_map.size().width;

    vector<int> indx, indy;
    vector<float> weights;

    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            if (edge_map.at<uchar>(i, j) > thres) {
                indx.emplace_back(i);
                indy.emplace_back(j);
                weights.emplace_back(edge_map.at<uchar>(i, j) / 255.0);
            }
        }
    }
    Mat mat_indx(indx), mat_indy(indy);
    mat_indx.convertTo(mat_indx, CV_16UC1);
    mat_indy.convertTo(mat_indy, CV_16UC1);
    Mat edge_centers(indx.size(), 2, CV_16UC1, Scalar(0));
    mat_indx.reshape(1, indx.size()).copyTo(edge_centers.col(0));
    mat_indy.reshape(1, indy.size()).copyTo(edge_centers.col(1));


    //long stt = clock();
    Mat bboxes = Mat::zeros(edge_centers.rows, 10, CV_16UC1);
    Mat tmp = edge_centers - window_size / 2;
    tmp.copyTo(bboxes.colRange(0, 2));
    cv::threshold(edge_centers.col(0) + window_size / 2, bboxes.col(2), H - 1, H - 1, THRESH_TRUNC);
    cv::threshold(edge_centers.col(1) + window_size / 2, bboxes.col(3), W - 1, W - 1, THRESH_TRUNC);
    edge_centers.copyTo(bboxes.colRange(4, 6));
    tmp = (bboxes.col(0) + bboxes.col(2)) / 2;
    tmp.copyTo(bboxes.col(6));
    tmp = (bboxes.col(1) + bboxes.col(3)) / 2;
    tmp.copyTo(bboxes.col(7));
    bboxes.convertTo(bboxes, CV_32SC1);
    tmp = bboxes.col(2) - bboxes.col(0);
    tmp.copyTo(bboxes.col(8));
    tmp = bboxes.col(3) - bboxes.col(1);
    tmp.copyTo(bboxes.col(9));

    // do NMS for generated bboxes.
    vector<int> nms_indices = nms(bboxes, weights, nms_thres);

    Mat bboxes_nms(nms_indices.size(), 10, CV_32SC1);
    for (int i = 0; i < nms_indices.size(); ++i) {
        bboxes.row(nms_indices[i]).copyTo(bboxes_nms.row(i));
    }

    return bboxes_nms;
}

struct bbox_ret{
    int item[10];
};

bbox_ret bbox_scaling(const cv::Mat &bbox, float factor, int maxH, int maxW) {
    /*
    Scaling the `bbox` according to `factor`, maximum height and width
    are constrained by `maxH` and `maxW`, respectively.
    */
    int wh = bbox.at<int>(8) * factor, ww = bbox.at<int>(9) * factor;
    int cx = bbox.at<int>(6), cy = bbox.at<int>(7);
    //# 0,1: upper-left corner; 2,3: bottom-right corner; 4,5: edge point; 6,7: center point; 8,9: height, width;
    bbox_ret ret;
    ret.item[0] = max(cx - wh / 2, 0), ret.item[1] = max(cy - ww / 2, 0), ret.item[2] = min(cx + wh / 2, maxH - 1);
    ret.item[3] = min(cy + ww / 2, maxW - 1), ret.item[4] = bbox.at<int>(4);
    ret.item[5] = bbox.at<int>(5), ret.item[6] = cx, ret.item[7] = cy, ret.item[8] = wh, ret.item[9] = ww;
    return ret;
}


pair<Mat, Mat>
resampling_keypoints(const Mat &edge_map1, const Mat &edge_map2, const Mat &b1, const Mat &b2, int maxW = 1,
                     int maxH = 1) {
    int local_ws = 10;

    //# handling the boundary cases
    int shift_x = max(b2.at<int>(2) - maxH, 0);
    int shift_y = max(b2.at<int>(3) - maxW, 0);

    //# obtaining pixel indices of the intersected region
    Mat edge_map1_rect = edge_map1(Range(b1.at<int>(0), b1.at<int>(2) - shift_x),
                                   Range(b1.at<int>(1), b1.at<int>(3) - shift_y));
    Mat edge_map2_rect = edge_map2(Range(b2.at<int>(0), b2.at<int>(2) - shift_x),
                                   Range(b2.at<int>(1), b2.at<int>(3) - shift_y));

    Mat edge_map1_mask = Mat(edge_map1_rect.size().height, edge_map1_rect.size().width, CV_8UC1);
    threshold(edge_map1_rect, edge_map1_mask, thres, 1, THRESH_BINARY);
    Mat edge_map2_mask = Mat(edge_map2_rect.size().height, edge_map2_rect.size().width, CV_8UC1);
    threshold(edge_map2_rect, edge_map2_mask, thres, 1, THRESH_BINARY);

    Mat logical_and = Mat(min(edge_map1_mask.size().height, edge_map2_mask.size().height),
                          min(edge_map1_mask.size().width, edge_map2_mask.size().width), CV_8UC1);
    vector<int> indices[2];
    vector<pair<float, float>> edge_centers;
    for (int i = 0; i < logical_and.size().height; ++i) {
        auto mp1_mask_row = edge_map1_mask.ptr<uchar>(i);
        auto mp2_mask_row = edge_map2_mask.ptr<uchar>(i);
        for (int j = 0; j < logical_and.size().width; ++j) {
            if (mp1_mask_row[j] & mp2_mask_row[j]) {
                indices[0].emplace_back(i);
                indices[1].emplace_back(j);
                edge_centers.emplace_back(make_pair(i, j));
            }
        }
    }

    //# shift the intersection indices w.r.t the whole edge map
    Mat edge_centers1 = Mat(edge_centers.size(), 2, CV_32FC1, Scalar(0));
    for (int i = 0; i < edge_centers.size(); ++i) {
        edge_centers1.at<float>(i, 0) = edge_centers[i].first + b1.at<int>(0);
        edge_centers1.at<float>(i, 1) = edge_centers[i].second + b1.at<int>(1);
    }

    Mat edge_centers2 = Mat(edge_centers.size(), 2, CV_32FC1, Scalar(0));
    for (int i = 0; i < edge_centers.size(); ++i) {
        edge_centers2.at<float>(i, 0) = edge_centers[i].first + b2.at<int>(0);
        edge_centers2.at<float>(i, 1) = edge_centers[i].second + b2.at<int>(1);
    }

    //# the weights for the NMS
    vector<float> weights;
    for (int i = 0; i < indices[0].size(); ++i) {
        int x = indices[0][i], y = indices[1][i];
        weights.emplace_back(((int) edge_map1.at<uchar>(x, y) + (int) edge_map2.at<uchar>(x, y)) / (255. * 2));
    }

    Mat edge_bboxes1 = Mat(edge_centers1.size().height, 4, CV_32FC1, Scalar(0));
    for (int i = 0; i < edge_bboxes1.size().height; ++i) {
        edge_bboxes1.at<float>(i, 0) = max((float) 0, edge_centers1.at<float>(i, 0) - local_ws / 2);
        edge_bboxes1.at<float>(i, 1) = max((float) 0, edge_centers1.at<float>(i, 1) - local_ws / 2);
        edge_bboxes1.at<float>(i, 2) = min(max((float) 0, edge_centers1.at<float>(i, 0) + local_ws / 2),
                                           (float) b1.at<int>(2));
        edge_bboxes1.at<float>(i, 3) = min(max((float) 0, edge_centers1.at<float>(i, 1) + local_ws / 2),
                                           (float) b1.at<int>(3));
    }
    edge_bboxes1.convertTo(edge_bboxes1, CV_32SC1);
    //# do NMS for sampled points
    vector<int> nms_indices;
    if (edge_bboxes1.size().height) {
        nms_indices = nms(edge_bboxes1, weights, nms_resampling_thres);
    }

    Mat edge_centers_nms1(nms_indices.size(), 2, CV_32FC1), edge_centers_nms2(nms_indices.size(), 2, CV_32FC1);
    for (int i = 0; i < nms_indices.size(); ++i) {
        edge_centers1.row(nms_indices[i]).copyTo(edge_centers_nms1.row(i));
        edge_centers2.row(nms_indices[i]).copyTo(edge_centers_nms2.row(i));
    }

    //# normalize the sampled points
    for (int i = 0; i < edge_centers_nms1.size().height; ++i) {
        edge_centers_nms1.at<float>(i, 0) /= maxH;
        edge_centers_nms1.at<float>(i, 1) /= maxW;
        edge_centers_nms2.at<float>(i, 0) /= maxH;
        edge_centers_nms2.at<float>(i, 1) /= maxW;
    }
    return make_pair(std::move(edge_centers_nms1), std::move(edge_centers_nms2));
}

cv::Mat warp_image_cv(const cv::Mat &img, const cv::Mat &c_src, const cv::Mat &c_dst, int shape_x, int shape_y) {
    // Warping `img` following the sets of key points  from `c_src` to `c_dst`.
    cv::Mat theta = tps_theta_from_points(c_src, c_dst, tps_lambda, true);
    cv::Mat grid = tps_grid(theta, c_dst, shape_x, shape_y);
    std::vector<cv::Mat> maps; //>cv::Mat mapx, mapy;
    tps_grid_to_remap(grid, maps, img.rows, img.cols);
    cv::Mat ret;
    ret.create(img.size(), img.type());
    cv::remap(img, ret, maps[0], maps[1], cv::INTER_CUBIC, cv::BORDER_REPLICATE);
    return ret;
}

cv::Mat point_vector_to_int(const std::vector<std::vector<float>> &points) {
    cv::Mat ret;
    ret.create(points.size(), points[0].size(), CV_32FC1);

    for (int i = 0; i < points.size(); ++i) {
        for (int j = 0; j < points[0].size(); ++j) {
            ret.at<float>(i, j) = points[i][j];
        }
    }
    return ret;
}

void tm_registration(Mat im_vise, Mat im_infe, Mat im_vis, Mat im_inf, Mat &im_blended) {
    Mat im_infe_gray;
    cvtColor(im_infe, im_infe_gray, COLOR_RGB2GRAY);
    Mat im_vise_gray;
    cvtColor(im_vise, im_vise_gray, COLOR_RGB2GRAY);


    int W = im_infe.size().width;
    int H = im_infe.size().height;
    Mat background = im_vise_gray.clone();

    // roughly matching the infrared and visible images and visualize their blended results
    float scaleMaxVal;
    Point scaleMaxLoc;
    Mat scaleMaxTpl;
    float scaleMaxDif;
    auto t = GetCurrentTime();
    double et;
    multiScaleMatchTemplate(im_infe_gray, background, 5, 0.1, scaleMaxVal, scaleMaxLoc, scaleMaxTpl);
    et = GetElapsedTime(t);
    LOGE("Roughly TM procedure costs %f ms", et);

    vector<int> rough_match_bbox;
    rough_match_bbox.push_back(scaleMaxLoc.x);
    rough_match_bbox.push_back(scaleMaxLoc.y);
    rough_match_bbox.push_back(scaleMaxTpl.size().height);
    rough_match_bbox.push_back(scaleMaxTpl.size().width);


    int tH = scaleMaxTpl.size().height, tW = scaleMaxTpl.size().width;
    int i_l = max(scaleMaxLoc.y, 0), j_l = max(scaleMaxLoc.x, 0);
    int i_r = min(scaleMaxLoc.y + tH, H), j_r = min(scaleMaxLoc.x + tW, W);

    Mat background_rm = im_vise_gray(Range(i_l, i_r), Range(j_l, j_r));
    Mat im_vis_crop = im_vis(Range(i_l, i_r), Range(j_l, j_r));

    // obtaining proposals `bboxes` at the centers of edge of the infrared image
    Mat foreground_e = scaleMaxTpl.clone();

    t = GetCurrentTime();
    Mat bboxes = gen_bboxes_from_edge_nms(foreground_e);
    et = GetElapsedTime(t);
    LOGE("Re-sampling procedure costs %f ms", et);

    map<string, vector<vector<float>>> matched_points;
    matched_points["inf"] = vector<vector<float>>();
    matched_points["vis"] = vector<vector<float>>();
    int scaleMaxH = background_rm.size().height, scaleMaxW = background_rm.size().width;


    t = GetCurrentTime();
    for (int bbox_row = 0; bbox_row < bboxes.size().height; bbox_row++) {
        Mat b;
        bboxes.row(bbox_row).copyTo(b);


        Mat tpl = foreground_e(Range(b.at<int>(0), b.at<int>(2)), Range(b.at<int>(1), b.at<int>(3)));
        bbox_ret bg_bbox = bbox_scaling(b, bbox_scaling_factor, background_rm.size().height,
                                           background_rm.size().width);
        Mat bg = background_rm(Range(bg_bbox.item[0], bg_bbox.item[2]), Range(bg_bbox.item[1], bg_bbox.item[3]));

        //# fine-grained matching between the local cross-domain windows.
        multiScaleMatchTemplate(tpl, bg, 5, 0.05, scaleMaxVal, scaleMaxLoc, scaleMaxTpl);

        if (scaleMaxVal > matching_thres) {
            scaleMaxLoc.x = scaleMaxLoc.x + bg_bbox.item[1];
            scaleMaxLoc.y = scaleMaxLoc.y + bg_bbox.item[0];

            //# resampling more control points from the matched cross-domain edges, in order to
            //# preserve the structure.
            pair<Mat, Mat> tmpret = resampling_keypoints(foreground_e, background_rm, b.colRange(0, 4),
                                                         (Mat_<int>(1, 4) << scaleMaxLoc.y, scaleMaxLoc.x,
                                                                 scaleMaxLoc.y + b.at<int>(8), scaleMaxLoc.x +
                                                                                               b.at<int>(9)),
                                                         scaleMaxW, scaleMaxH);
            Mat sampled_point_inf = tmpret.first, sampled_point_vis = tmpret.second;
            for (int i = 0; i < sampled_point_inf.size().height; i++) {
                vector<float> inf_part, vis_part;
                for (int j = sampled_point_inf.size().width - 1; j >= 0; j--) {
                    inf_part.push_back(sampled_point_inf.at<float>(i, j));
                    vis_part.push_back(sampled_point_vis.at<float>(i, j));
                }
                matched_points["inf"].push_back(inf_part);
                matched_points["vis"].push_back(vis_part);
            }
        }
    }
    et = GetElapsedTime(t);
    LOGE("Fine-grained TM procedure costs %f ms", et);

    if(matched_points["inf"].empty() || matched_points["vis"].empty()) return;

    t = GetCurrentTime();
    cv::Mat matched_points_inf = point_vector_to_int(matched_points["inf"]);
    cv::Mat matched_points_vis = point_vector_to_int(matched_points["vis"]);
    float score = 0.;
    cv::resize(im_inf, im_inf, cv::Size(scaleMaxW, scaleMaxH));

    im_inf = warp_image_cv(im_inf, matched_points_inf, matched_points_vis, scaleMaxH, scaleMaxW);
    cv::addWeighted(im_vis_crop, 0.5, im_inf, 0.5, 3, im_blended);
    cv::resize(im_blended, im_blended, cv::Size(384, 288));
    et = GetElapsedTime(t);
    LOGE("Warping procedure costs %f ms", et);
}

Mat matchVisualization(Mat Template, Mat background, pair<int, int> location, string title = NULL, string text = NULL) {
    /*
    """
    Visualize the matched template with its background.
    """
    */
    int tH = Template.size().height, tW = Template.size().width;
    if (Template.channels() == 1) {
        Mat zero_like_temp = Mat::zeros(Size(tH, tW), CV_8UC1);
        vector<Mat> mats = (2, zero_like_temp.clone());
        mats.push_back(Template);
        merge(mats, Template);
    }
    addWeighted(background(Range(location.second, location.second + tH), Range(location.first, location.first + tW)),
                0.5, Template, 0.5, 3,
                background(Range(location.second, location.second + tH), Range(location.first, location.first + tW)));
    return background;
}