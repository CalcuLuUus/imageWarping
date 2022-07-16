#include <iostream>
//#include <arm_neon.h>
#include <cstring>
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <fstream>

namespace TPS {
    /*
    Thin plate spline
    */
    const float eps = 1e-8;
    const int wa_split_pivot = 3;

    void unsq_l2_dist(const float *a, const float *b, float *dst, int n, int m) {
        // a.shape, b.shape -> [n, 3], [m, 3]
        // dst.shape -> [n, m]
        for (uint32_t i = 0; i < n; ++i) {
            for (uint32_t j = 0; j < m; ++j) {
                uint32_t dist_idx = i * m + j;
                uint32_t a_idx = i * 3, b_idx = j * 3;
                float tmp1 = a[a_idx] - b[b_idx];
                float tmp2 = a[a_idx + 1] - b[b_idx + 1];
                dst[dist_idx] = tmp1 * tmp1 + tmp2 * tmp2;
            }
        }
    }

    void wa_split(const float *src, float *dist_w, float &a0, float &a1, float &a2, int n) {
        memcpy(dist_w, src, (n - 3) * sizeof(float));
        //for (uint32_t i = 0; i < n - 3; ++ i) dist_w[i] = src[i];
        a0 = src[n - 3];
        a1 = src[n - 2];
        a2 = src[n - 1];
    }

    float sum_1d(const float *src, int n) {
        float sum = 0;
        for (uint32_t i = 0; i < n; ++i) sum += src[i];
        return sum;
    }

    void reproduce_w(const float *src, float *dst, int n) {
        dst[0] = -sum_1d(src, n);
        memcpy(dst + 1, src, (n - 1) * sizeof(float));
    }

    cv::Mat du(const cv::Mat &src_x, const cv::Mat &src_c, int n, int m) {
        cv::Mat dst = cv::Mat::zeros(cv::Size(n, m), CV_32FC1);
        unsq_l2_dist(src_x.ptr<float>(0), src_c.ptr<float>(0), dst.ptr<float>(0), n, m);
        cv::Mat log_dst;
        cv::log(dst + eps, log_dst);
        return dst.mul(log_dst * 0.5);
    }

    cv::Mat z(const cv::Mat &src_x, const cv::Mat &src_c, const cv::Mat &theta) {
        int n = src_x.rows, m = src_c.rows;
        cv::Mat U = du(src_x, src_c, n, m);
        int theta_qty = theta.size[0];
        cv::Mat w = cv::Mat::zeros(cv::Size(theta_qty - 3, 1), CV_32FC1);
        float a0, a1, a2;
        wa_split(theta.ptr<float>(0), w.ptr<float>(0), a0, a1, a2, theta_qty);

        bool reduced = theta_qty == src_c.size[0] + 2;
        cv::Mat b;
        if (reduced) {
            cv::Mat w_reduced = cv::Mat::zeros(cv::Size(theta_qty + 1, 1), CV_32FC1);
            reproduce_w(w.ptr<float>(0), w_reduced.ptr<float>(0), theta_qty);
            b = U * w_reduced;
        } else b = U * w;
        std::vector<cv::Mat> xs(2);
        cv::split(src_x, xs);
        return a0 + a1 * xs[0] + a2 * xs[1] + b;
    }

    void split_pv(const float *src, float *dst_p, float *dst_v, int n) {
        // dst_v, dst_p are assumed to be all-zero matrix
        for (uint32_t i = 0; i < n; ++i) {
            dst_p[i * n + 1] = src[i * n];
            dst_p[i * n + 2] = src[i * n + 1];
            dst_v[i] = src[i * n + 2];
        }
    }


    void construct_A(float *src_k, float *src_p, float *dst, int n) {
        // dst is assumed to be a [n + 3, n + 3] all-zero matrix
        // dst_v is assumed to be a [n + 3, 1] all-zero matrix
        const int k_len = n + 3;
        for (uint32_t i = 0; i < n; ++i) {
            for (uint32_t j = 0; j < n; ++j) {
                dst[i * k_len + j] = src_k[i * n + j];
            }
        }
        for (uint32_t i = 0; i < n; ++i) {
            dst[i * k_len + n] = src_p[i];
            dst[i * k_len + n + 1] = src_p[i + n];
            dst[i * k_len + n + 2] = src_p[i + n * 2];
            // transposed P
            dst[n * k_len + i] = src_p[i];
            dst[n * k_len + i + n] = src_p[i + n];
            dst[n * k_len + i + n * 2] = src_p[i + n * 2];
        }
    }

    void reproduce_theta(const float *src, float *dst, int n) {
        memcpy(dst + 1, src, (n - 1) * sizeof(float));
    }

    cv::Mat fit(const cv::Mat &c, float lambd, bool reduced) {
        int n = c.size[0];
        cv::Mat U = du(c, c, n, n);
        cv::Mat K = U + lambd * cv::Mat::eye(cv::Size(n, n), CV_32FC1);
        cv::Mat A = cv::Mat::zeros(cv::Size(n + 3, n + 3), CV_32FC1);
        cv::Mat P = cv::Mat::zeros(cv::Size(n, 3), CV_32FC1);
        cv::Mat v = cv::Mat::zeros(cv::Size(n + 3, 1), CV_32FC1);
        split_pv(c.ptr<float>(0), P.ptr<float>(0), v.ptr<float>(0), n);
        construct_A(K.ptr<float>(0), P.ptr<float>(0), A.ptr<float>(0), n);
        cv::Mat theta = cv::Mat::zeros(cv::Size(n + 3, 1), CV_32FC1);
        cv::solve(A, v, theta);
        if (reduced) {
            cv::Mat theta_reduced = cv::Mat::zeros(cv::Size(n + 2, 1), CV_32FC1);
            reproduce_theta(theta.ptr<float>(0), theta_reduced.ptr<float>(0), n + 2);
            return theta_reduced;
        } else return theta;
    }

};

void uniform_grid(float *dst, int h, int w) {
    for (uint32_t j = 0; j < w; ++j ) dst[j << 1] = j / float(w - 1);
    for (uint32_t i = 1; i < h; ++i) {
        for (uint32_t j = 0; j < w; ++j) {
            dst[i * w * 2 + j * 2] = dst[j << 1];
        }
    }
    for (uint32_t i = 0; i < h; ++i) dst[i * w * 2 + 1] = i / float(h - 1);
    for (uint32_t i = 0; i < h; ++i) {
        for (uint32_t j = 1; j < w; ++j) {
            dst[i * w * 2 + j * 2 + 1] = dst[i * w * 2 + 1];
        }
    }
}

cv::Mat tps_grid(const cv::Mat &theta, const cv::Mat &c_dst, const cv::Size &dshape) {
    //bool reduced = c_dst.size[0] + 2 == theta.size[0];
    cv::Mat ugrid = cv::Mat::zeros(cv::Size(dshape.height * dshape.width, 2), CV_32FC1);
    uniform_grid(ugrid.ptr<float>(0), dshape.width, dshape.height);
    std::vector<cv::Mat> thetas(2);
    cv::split(theta, thetas);
    cv::Mat dgrid;
    cv::merge(std::vector<cv::Mat>({TPS::z(ugrid, c_dst, thetas[0]), TPS::z(ugrid, c_dst, thetas[1])}), dgrid);
    auto grid = dgrid + ugrid;
    return grid;
}

void column_stack_split_last(const float *src_c, const float *src_d,
                             float *dst_0, float *dst_1, int n) {
    for (uint32_t i = 0; i < n; ++i) {
        dst_0[i * 3] = src_c[i << 1];
        dst_0[i * 3 + 1] = src_c[i << 1 | 1];
        dst_0[i * 3 + 2] = src_d[i << 1];
        dst_1[i * 3] = src_c[i << 1];
        dst_1[i * 3 + 1] = src_c[i << 1 | 1];
        dst_1[i * 3 + 2] = src_d[i << 1 | 1];
    }

}

cv::Mat tps_theta_from_points(const cv::Mat &c_src, const cv::Mat &c_dst, float lambd, bool reduced) {
    cv::Mat delta = c_src - c_dst;
    int n = c_src.size[0];
    cv::Mat dx = cv::Mat::zeros(cv::Size(n, 3), CV_32FC1);
    cv::Mat dy = cv::Mat::zeros(cv::Size(n, 3), CV_32FC1);
    column_stack_split_last(c_dst.ptr<float>(0), delta.ptr<float>(0) + n, dx.ptr<float>(0), dy.ptr<float>(0), n);
    cv::Mat ret;
    cv::merge(std::vector<cv::Mat>({TPS::fit(dx, lambd, reduced), TPS::fit(dy, lambd, reduced)}), ret);
    return ret;
}

void tps_grid_to_remap(const cv::Mat &grid, std::vector<cv::Mat> &dsts, int size_x, int size_y) {
    cv::split(grid, dsts);
    dsts[0] *= size_x;
    dsts[1] *= size_y;
}

int main() {

}