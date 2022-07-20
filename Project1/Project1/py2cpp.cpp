#include <iostream>  
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <filesystem>
#include <io.h>
#include <algorithm>
#include <direct.h>
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
		}
		else b = U * w;
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
		}
		else return theta;
	}

};

void uniform_grid(float *dst, int h, int w) {
	for (uint32_t j = 0; j < w; ++j) dst[j << 1] = j / float(w - 1);
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
	std::vector<cv::Mat> mats;
	auto mat1 = TPS::z(ugrid, c_dst, thetas[0]);
	auto mat2 = TPS::z(ugrid, c_dst, thetas[1]);
	//cv::merge(std::vector<cv::Mat>({ TPS::z(ugrid, c_dst, thetas[0]), TPS::z(ugrid, c_dst, thetas[1]) }), dgrid);
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
	cv::merge(std::vector<cv::Mat>({ TPS::fit(dx, lambd, reduced), TPS::fit(dy, lambd, reduced) }), ret);
	return ret;
}

void tps_grid_to_remap(const cv::Mat &grid, std::vector<cv::Mat> &dsts, int size_x, int size_y) {
	cv::split(grid, dsts);
	dsts[0] *= size_x;
	dsts[1] *= size_y;
}

using namespace cv;
using namespace std;

const float eps = 1e-6;

/*
global data
*/
int SeqId = 5;
//window size
int window_size = 80;

//thres of edge proposals
int thres = 200;

//overlapped len of bbox
int overlap_len = 25;

float matching_thres = 0.3;
float nms_thres = 0.2;
float nms_resampling_thres = 0.1;
float tps_lambda = 0.2;

//the scaling factor for a proposal, when searching in the background
float bbox_scaling_factor = 1.2;

// if visualize the edge bboxes
bool vis_edge_bbox_checked = false;

// input cross domain paths
string infe_dir = "./data/EdgeMaps/Seq" + to_string(SeqId) + "/infrared";
string vise_dir = "./data/EdgeMaps/Seq" + to_string(SeqId) + "/visible";

string inf_dir = "./data/Preprocessed/Seq" + to_string(SeqId) + "/infrared";
string vis_dir = "./data/Preprocessed/Seq" + to_string(SeqId) + "/visible_color";

string infe_path;
string vise_path;

/*for debug*/
void showimg(Mat img)
{
	imshow("img", img);
	waitKey(0);
}

void makedir(const string &path)
{
	if (0 != _access(path.c_str(), 0))
	{
		// if this folder not exist, create a new one.
		_mkdir(path.c_str());
	}
}

void listdir(const string &path, vector<string> &files)
{
	intptr_t hFile = 0;

	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
			{
				files.emplace_back(fileinfo.name);
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}

bool cmp(const float &a, const float &b)
{
	return a > b + eps;
}

vector<float> topk(const Mat &mat, const int &k)
{
	vector<float> array;
	if (mat.isContinuous())
	{
		array.assign(mat.datastart, mat.dataend);
	}
	else
	{
		for (int i = 0; i < mat.rows; ++i)
		{
			array.insert(array.end(), mat.ptr<float>(i), mat.ptr<float>(i) + mat.cols);
		}
	}

	sort(array.begin(), array.end(), cmp);
	if (array.size() > k)
	{
		vector<float> new_array;
		new_array.assign(array.begin(), array.begin() + k);
		return new_array;
	}
	else
	{
		return array;
	}

	
}

void multiScaleMatchTemplate(const Mat &foreground, const Mat &background, const int &num_scales, const float &precision,           //default: num_scale=5, precision=0.1
							float &scaleMaxVal, Point &scaleMaxLoc, Mat &scaleMaxTpl, float &scaleMaxDif)
{
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
	for (int cnt = 0; cnt < num_scales; val += distance, cnt++)
	{
		scales.emplace_back(val);
	}
	scaleMaxVal = -1;
	
	for (float scale : scales)
	{
		// resize the template using a certain scale value
		Mat tpl;
		resize(foreground, tpl, Size(min(int(foreground.size().width * scale), background.size().width),
									min(int(foreground.size().height * scale), background.size().height)));

		Mat result;
		matchTemplate(background, tpl, result, cv::TM_CCORR_NORMED);

		// Good matches are expected to be unique.
		vector<float> result_topk = topk(result, 5);
		float topk_diff = 0;
		for (int i = 1; i < result_topk.size(); ++i)
		{
			topk_diff += (abs(result_topk[i] - result_topk[i - 1]));
		}
		double minVal, maxVal;
		Point minLoc, maxLoc;
		minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
		if (maxVal > scaleMaxVal + eps)
		{
			scaleMaxDif = topk_diff;
			scaleMaxVal = maxVal;
			scaleMaxLoc = maxLoc;
			tpl.copyTo(scaleMaxTpl);
		}
	}
}

vector<int> argsort(const vector<float> &v, int st = -1) // default st = -1, st == -1 dec, st == 0 asc
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

vector<int> nms(const Mat &boxes, const vector<float> &scores, const float &nms_thr)
{
	Mat x1 = boxes.col(0).clone().reshape(1, 1);
	Mat y1 = boxes.col(1).clone().reshape(1, 1);
	Mat x2 = boxes.col(2).clone().reshape(1, 1);
	Mat y2 = boxes.col(3).clone().reshape(1, 1);

	Mat areas = (x2 - x1 + 1).mul(y2 - y1 + 1);

	vector<int> order = argsort(scores, -1);

	map<int, int> mp_id; //map[idx] -> id_of_bbox
	for (int i = 0; i < order.size(); ++i)
	{
		mp_id[i] = order[i];
	}

	vector<int> keep;
	while (mp_id.size() > 0)
	{
		int i = (mp_id.begin()->second);
		keep.emplace_back(i);
		mp_id.erase(mp_id.begin());

		int val_x1 = x1.at<int>(i), val_y1 = y1.at<int>(i), val_x2 = x2.at<int>(i), val_y2 = y2.at<int>(i);
		for (map<int, int>::iterator it = mp_id.begin(); it != mp_id.end(); )
		{
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
			double ovr = (double)inter / (double)(area1 + area2 - inter);
			
			if (ovr > nms_thr + eps)
			{
				it = mp_id.erase(it);
			}
			else
			{
				it++;
			}
		}
	}

	return keep;
}


Mat gen_bboxes_from_edge_nms(const Mat &edge_map)
{
	/*
		Generating a bbox at the center of an edge pixel(`thres` as the threshold).
		The window size is determined by `ws`.
	*/
	int H = edge_map.size().height, W = edge_map.size().width;

	vector<int> indx, indy;
	vector<float> weights;

	for (int i = 0; i < H; ++i)
	{
		for (int j = 0; j < W; ++j)
		{
			if (edge_map.at<uchar>(i, j) > thres)
			{
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
	for (int i = 0; i < nms_indices.size(); ++i)
	{
		bboxes.row(nms_indices[i]).copyTo(bboxes_nms.row(i));
	}
	
	/* hard to code
	# visualization
		if vis_edge_bbox_checked:
			edge_map = np.stack([edge_map, edge_map, edge_map], axis = -1)
		for b in bboxes_nms :
			cv2.rectangle(edge_map, (b[1], b[0]), (b[3], b[2]), (0, 0, 255), 2)

		saveShow(edge_map, 'Edge BBoxes')
		# cv2.waitKey(0)
	*/
	/* for debug*/
	if (vis_edge_bbox_checked)
	{
		Mat edgemap = edge_map.clone();
		vector<Mat> mats(3, edgemap);
		Mat visu;
		merge(mats, visu);
		for (int row = 0; row < bboxes_nms.size().height; row++)
		{
			rectangle(visu, Rect(Point(bboxes_nms.at<int>(row, 1), bboxes_nms.at<int>(row, 0)), Point(bboxes_nms.at<int>(row, 3), bboxes_nms.at<int>(row, 2))),
				Scalar(0, 0, 255), 2);
		}
		showimg(visu);
	}
	return bboxes_nms;
}


vector<int> bbox_scaling(const Mat &bbox, const float &factor, const int &maxH, const int &maxW)
{
	/*
	Scaling the `bbox` according to `factor`, maximum height and width
    are constrained by `maxH` and `maxW`, respectively.
	*/
	int wh = bbox.at<int>(8) * factor, ww = bbox.at<int>(9) * factor;
	int cx = bbox.at<int>(6), cy = bbox.at<int>(7);
	//# 0,1: upper-left corner; 2,3: bottom-right corner; 4,5: edge point; 6,7: center point; 8,9: height, width;
	vector<int> ret(10);
	ret[0] = max(cx - wh / 2, 0), ret[1] = max(cy - ww / 2, 0), ret[2] = min(cx + wh / 2, maxH - 1);
	ret[3] = min(cy + ww / 2, maxW - 1), ret[4] = bbox.at<int>(4);
	ret[5] = bbox.at<int>(5), ret[6] = cx, ret[7] = cy, ret[8] = wh, ret[9] = ww;
	return ret;
}


pair<Mat, Mat> resampling_keypoints(const Mat &edge_map1, const Mat &edge_map2, const Mat &b1, const Mat &b2, const int &maxW = 1, const int &maxH = 1)
{
	/*
	"""
    Sampling key points from edge pixels of registered two bboxes (`b1` and `b2`).

    """
	*/
	int local_ws = 10;

	//# handling the boundary cases
	int shift_x = max(b2.at<int>(2) - maxH, 0);
	int shift_y = max(b2.at<int>(3) - maxW, 0);

	//# obtaining pixel indices of the intersected region
	Mat edge_map1_rect = edge_map1(Range(b1.at<int>(0), b1.at<int>(2) - shift_x), Range(b1.at<int>(1), b1.at<int>(3) - shift_y));
	Mat edge_map2_rect = edge_map2(Range(b2.at<int>(0), b2.at<int>(2) - shift_x), Range(b2.at<int>(1), b2.at<int>(3) - shift_y));

	Mat edge_map1_mask = Mat(edge_map1_rect.size().height, edge_map1_rect.size().width, CV_8UC1);
	threshold(edge_map1_rect, edge_map1_mask, thres, 1, THRESH_BINARY);
	Mat edge_map2_mask = Mat(edge_map2_rect.size().height, edge_map2_rect.size().width, CV_8UC1);
	threshold(edge_map2_rect, edge_map2_mask, thres, 1, THRESH_BINARY);

	Mat logical_and = Mat(min(edge_map1_mask.size().height, edge_map2_mask.size().height),
		min(edge_map1_mask.size().width, edge_map2_mask.size().width), CV_8UC1);
	vector<int> indices[2];
	vector<pair<float, float>> edge_centers;
	for (int i = 0; i < logical_and.size().height; ++i)
	{
		auto mp1_mask_row = edge_map1_mask.ptr<uchar>(i);
		auto mp2_mask_row = edge_map2_mask.ptr<uchar>(i);
		for (int j = 0; j < logical_and.size().width; ++j)
		{
			if (mp1_mask_row[j] + mp2_mask_row[j] == 2)
			{
				indices[0].emplace_back(i);
				indices[1].emplace_back(j);
				edge_centers.emplace_back(make_pair(i, j));
			}
		}
	}

	//# shift the intersection indices w.r.t the whole edge map
	Mat edge_centers1 = Mat(edge_centers.size(), 2, CV_32FC1, Scalar(0));
	for (int i = 0; i < edge_centers.size(); ++i)
	{
		edge_centers1.at<float>(i, 0) = edge_centers[i].first + b1.at<int>(0);
		edge_centers1.at<float>(i, 1) = edge_centers[i].second + b1.at<int>(1);
	}

	Mat edge_centers2 = Mat(edge_centers.size(), 2, CV_32FC1, Scalar(0));
	for (int i = 0; i < edge_centers.size(); ++i)
	{
		edge_centers2.at<float>(i, 0) = edge_centers[i].first + b2.at<int>(0);
		edge_centers2.at<float>(i, 1) = edge_centers[i].second + b2.at<int>(1);
	}

	//# the weights for the NMS
	vector<float> weights;
	for (int i = 0; i < indices[0].size(); ++i)
	{
		int x = indices[0][i], y = indices[1][i];
		weights.emplace_back(((int)edge_map1.at<uchar>(x, y) + (int)edge_map2.at<uchar>(x, y)) / (255. * 2));
	}
	
	Mat edge_bboxes1 = Mat(edge_centers1.size().height, 4, CV_32FC1, Scalar(0));
	for (int i = 0; i < edge_bboxes1.size().height; ++i)
	{
		edge_bboxes1.at<float>(i, 0) = max((float)0, edge_centers1.at<float>(i, 0) - local_ws / 2);
		edge_bboxes1.at<float>(i, 1) = max((float)0, edge_centers1.at<float>(i, 1) - local_ws / 2);
		edge_bboxes1.at<float>(i, 2) = min(max((float)0, edge_centers1.at<float>(i, 0) + local_ws / 2), (float)b1.at<int>(2));
		edge_bboxes1.at<float>(i, 3) = min(max((float)0, edge_centers1.at<float>(i, 1) + local_ws / 2), (float)b1.at<int>(3));
	}
	edge_bboxes1.convertTo(edge_bboxes1, CV_32SC1); 
	//# do NMS for sampled points
	vector<int> nms_indices;
	if (edge_bboxes1.size().height) 
	{
		nms_indices = nms(edge_bboxes1, weights, nms_resampling_thres);
	}

	Mat edge_centers_nms1(nms_indices.size(), 2, CV_32FC1), edge_centers_nms2(nms_indices.size(), 2, CV_32FC1);
	for (int i = 0; i < nms_indices.size(); ++i)
	{
		edge_centers1.row(nms_indices[i]).copyTo(edge_centers_nms1.row(i));
		edge_centers2.row(nms_indices[i]).copyTo(edge_centers_nms2.row(i));
	}

	//# normalize the sampled points
	for (int i = 0; i < edge_centers_nms1.size().height; ++i)
	{
		edge_centers_nms1.at<float>(i, 0) /= maxH;
		edge_centers_nms1.at<float>(i, 1) /= maxW;
		edge_centers_nms2.at<float>(i, 0) /= maxH;
		edge_centers_nms2.at<float>(i, 1) /= maxW;
	}
	return make_pair(edge_centers_nms1, edge_centers_nms2);
}




void tm_registration(Mat &Warped, vector<int> &Rough_match_bbox,
	map<string, vector<vector<float>> > &Matched_points, float &Score)
{
	Mat im_infe = imread(infe_path);
	cvtColor(im_infe, im_infe, COLOR_BGR2RGB);
	Mat im_infe_gray;
	cvtColor(im_infe, im_infe_gray, COLOR_RGB2GRAY);

	Mat im_vise = imread(vise_path);
	cvtColor(im_vise, im_vise, COLOR_BGR2RGB);
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
	multiScaleMatchTemplate(im_infe_gray, background, 5, 0.1, scaleMaxVal, scaleMaxLoc, scaleMaxTpl, scaleMaxDif);
	vector<int> rough_match_bbox;
	rough_match_bbox.emplace_back(scaleMaxLoc.x);
	rough_match_bbox.emplace_back(scaleMaxLoc.y);
	rough_match_bbox.emplace_back(scaleMaxTpl.size().height);
	rough_match_bbox.emplace_back(scaleMaxTpl.size().width);


	int tH = scaleMaxTpl.size().height, tW = scaleMaxTpl.size().width;
	int i_l = max(scaleMaxLoc.y, 0), j_l = max(scaleMaxLoc.x, 0);
	int i_r = min(scaleMaxLoc.y + tH, H), j_r = min(scaleMaxLoc.x + tW, W);
	Mat background_rm = im_vise_gray(Range(i_l, i_r), Range(j_l, j_r));


	// obtaining proposals `bboxes` at the centers of edge of the infrared image
	Mat foreground_e = scaleMaxTpl.clone();
	Mat bboxes = gen_bboxes_from_edge_nms(foreground_e);

	map<string, vector<vector<float>> > matched_points;
	matched_points["inf"] = vector<vector<float>>();
	matched_points["vis"] = vector<vector<float>>();
	int scaleMaxH = background_rm.size().height, scaleMaxW = background_rm.size().width;


	for (int bbox_row = 0; bbox_row < bboxes.size().height; bbox_row++)
	{
		Mat b;
		bboxes.row(bbox_row).copyTo(b);

		Mat tpl = foreground_e(Range(b.at<int>(0), b.at<int>(2)), Range(b.at<int>(1), b.at<int>(3)));
		
		vector<int> bg_bbox = bbox_scaling(b, bbox_scaling_factor, background_rm.size().height, background_rm.size().width);
		Mat bg = background_rm(Range(bg_bbox[0], bg_bbox[2]), Range(bg_bbox[1], bg_bbox[3]));

		//# fine-grained matching between the local cross-domain windows.
		multiScaleMatchTemplate(tpl, bg, 5, 0.05, scaleMaxVal, scaleMaxLoc, scaleMaxTpl, scaleMaxDif);


		if (scaleMaxVal > matching_thres + eps)
		{
			scaleMaxLoc.x = scaleMaxLoc.x + bg_bbox[1];
			scaleMaxLoc.y = scaleMaxLoc.y + bg_bbox[0];
			//# resampling more control points from the matched cross-domain edges, in order to
			//# preserve the structure.

			pair<Mat, Mat> tmpret = resampling_keypoints(foreground_e, background_rm, b.colRange(0, 4),
				(Mat_<int>(1, 4) << scaleMaxLoc.y, scaleMaxLoc.x, scaleMaxLoc.y + b.at<int>(8), scaleMaxLoc.x + b.at<int>(9)),
				scaleMaxW, scaleMaxH);
			Mat sampled_point_inf = tmpret.first, sampled_point_vis = tmpret.second;


			for (int i = 0; i < sampled_point_inf.size().height; ++i)
			{
				vector<float> inf_part, vis_part;
				for (int j = sampled_point_inf.size().width - 1; j >= 0; j--)
				{
					inf_part.emplace_back(sampled_point_inf.at<float>(i, j));
					vis_part.emplace_back(sampled_point_vis.at<float>(i, j));
				}
				matched_points["inf"].emplace_back(inf_part);
				matched_points["vis"].emplace_back(vis_part);
			}
		}
	}

	

	/*
	# the following lines can be removed.
    ## BEGIN
    # warp the image `foreground_e` according to cross-domain matched points.
    warped = warp_image_cv(foreground_e, matched_points['inf'], matched_points['vis'], dshape=(scaleMaxH, scaleMaxW))
    # calculate the score of matched
    score = ccorr_normed(warped, background_rm)
    ## END
    
	*/
	Mat warped;
	float score = 0.;
	Warped = warped.clone();
	Rough_match_bbox = rough_match_bbox;
	Matched_points = matched_points;
	Score = score;
}

Mat matchVisualization(Mat Template, Mat background, pair<int, int> location, string title = NULL, string text = NULL)
{
	/*
	"""
    Visualize the matched template with its background.
    """
	*/
	int tH = Template.size().height, tW = Template.size().width;
	if (Template.channels() == 1)
	{
		Mat zero_like_temp = Mat::zeros(Size(tH, tW), CV_8UC1);
		vector<Mat> mats = (2, zero_like_temp.clone());
		mats.emplace_back(Template);
		merge(mats, Template);
	}
	addWeighted(background(Range(location.second, location.second + tH), Range(location.first, location.first + tW)), 0.5, Template, 0.5, 3, 
		background(Range(location.second, location.second + tH), Range(location.first, location.first + tW)));
	return background;
}

Mat warp_image_cv(Mat img, Mat c_src, Mat c_dst, pair<int, int> dshape = make_pair(-1, -1))
{
	/*
	"""
    Warping `img` following the sets of key points  from `c_src` to `c_dst`.
    """
	*/
	if (dshape.first == -1 && dshape.second == -1)
	{
		dshape.first = img.size().height, dshape.second = img.size().width;
	}
	Mat theta = tps_theta_from_points(c_src, c_dst, tps_lambda, true);
	//    # main bottleneck of the time consuming:  tps_grid
	Mat grid = tps_grid(theta, c_dst, Size(dshape.first, dshape.second));
	Mat mapx, mapy;
	vector<Mat> mats;
	mats.emplace_back(mapx);
	mats.emplace_back(mapy);
	tps_grid_to_remap(grid, mats, img.size().height, img.size().width);
	Mat ret;
	remap(img, ret, mapx, mapy, INTER_CUBIC, IPL_BORDER_REPLICATE);
	return ret;
	
}


int main()
{
	vector<string> inf_dir_listdir;
	listdir(inf_dir, inf_dir_listdir);
	sort(inf_dir_listdir.begin(), inf_dir_listdir.end());
	int id = 0;
	for (string fn : inf_dir_listdir)
	{
		try 
		{
			id++;
			/*test*/
			infe_path = infe_dir + '/' + fn;
			vise_path = vise_dir + '/' + fn;
			cout << id << endl;
			Mat warped_e;
			vector<int> rough_match_bbox;
			map<string, vector<vector<float>> > matched_points;
			float score;
			tm_registration(warped_e, rough_match_bbox, matched_points, score);

			/* for debug*/
			/*Mat combine, infimg = imread(infe_path), visimg = imread(vise_path);
			hconcat(infimg, visimg, combine);
			for (int i = 0; i < matched_points["inf"].size(); ++i)
			{
				
				Point p1(matched_points["inf"][i][0] * infimg.size().width, matched_points["inf"][i][1] * infimg.size().height);
				Point p2(matched_points["vis"][i][0] * visimg.size().width + infimg.size().width, matched_points["vis"][i][1] * visimg.size().height);
				circle(combine, p1, 1, Scalar(255, 0, 0), 2);  
				circle(combine, p2, 1, Scalar(255, 0, 0), 2);
				line(combine, p1, p2, Scalar(0, 0, 255), 1);
			}
			makedir("./test");
			vector <int> compression_params;
			compression_params.emplace_back(IMWRITE_PNG_COMPRESSION);
			compression_params.emplace_back(0);
			string fp = "./test/" + to_string(id) + ".png";
			imwrite(fp, combine, compression_params);*/
			
			/*wait for testing*/
			string inf_path = inf_dir + '/' + fn;
			string vis_path = vis_dir + '/' + fn;

			Mat im_inf = imread(inf_path);
			cvtColor(im_inf, im_inf, CV_BGR2RGB);

			Mat im_vis = imread(vis_path);

			Mat foreground;
			resize(im_inf, foreground, Size(rough_match_bbox[3], rough_match_bbox[2]));
			Mat background = im_vis(Range(rough_match_bbox[1], rough_match_bbox[1] + rough_match_bbox[2]), Range(rough_match_bbox[0], rough_match_bbox[0] + rough_match_bbox[3]));

			int scaleMaxH = background.size().height, scaleMaxW = background.size().width;

			//# Save img params
			string newname = fn;
			int cut = 0;
			while (newname[cut] != '.' && cut < newname.size())
			{
				cut++;
			}
			newname = newname.substr(0, cut) + ".png";
			string dir_path = "results/Seq" + to_string(SeqId) + "_warped_edge/";
			string fp = "./results/Seq" + to_string(SeqId) + "_warped_edge/" + newname;
			vector <int> compression_params;
			compression_params.emplace_back(IMWRITE_PNG_COMPRESSION);
			compression_params.emplace_back(0);
			imwrite(fp, warped_e, compression_params);
			
			
			// # Saving the warped infrared images.		
			Mat warped;
			/*
			warped = warp_image_cv(foreground, matched_points['inf'], matched_points['vis'],dshape=(scaleMaxH, scaleMaxW))
			*/
			Mat matchpoint_inf((int)matched_points["inf"].size(), (int)matched_points["inf"][0].size(), CV_32SC1);
			Mat matchpoint_vis((int)matched_points["vis"].size(), (int)matched_points["vis"][0].size(), CV_32SC1);
			for (int i = 0; i < matched_points["inf"].size(); ++i)
			{
				for (int j = 0; j < matched_points["inf"][0].size(); ++j)
				{
					matchpoint_inf.at<int>(i, j) = matched_points["inf"][i][j];
					matchpoint_vis.at<int>(i, j) = matched_points["vis"][i][j];
				}
			}
			warp_image_cv(foreground, matchpoint_inf, matchpoint_vis, make_pair(scaleMaxH, scaleMaxW));
			resize(warped, warped, Size(480, 360));
			dir_path = "results/Seq" + to_string(SeqId) + "_warped/";
			fp = "./results/Seq" + to_string(SeqId) + "_warped/" + newname;
			makedir(dir_path);
			imwrite(fp, warped, compression_params);

			/*
			# Saving the cropped visible images.
			*/
			resize(background, background, Size(480, 360));
			dir_path = "results/Seq" + to_string(SeqId) + "_bg/";
			fp = "./results/Seq" + to_string(SeqId) + "_bg/" + newname;
			makedir(dir_path);
			imwrite(fp, background, compression_params);

			/*
			# Saving the registered blended infrared-visible images.
			*/
			applyColorMap(warped, warped, COLORMAP_JET);
			background = matchVisualization(warped, background, make_pair(0, 0));
			resize(background, background, Size(480, 360));
			dir_path = "results/Seq" + to_string(SeqId) + "_blended/";
			fp = "./results/Seq" + to_string(SeqId) + "_blended/" + newname;
			makedir(dir_path);
			imwrite(fp, background, compression_params);
		}
		catch(...)
		{
			cout << "Exception :" << fn << endl;
		}
	}

	return 0;
}
