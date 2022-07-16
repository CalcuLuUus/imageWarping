//#include <iostream>  
//#include <opencv2/opencv.hpp>
//#include <opencv2/core/core.hpp>  
//#include <opencv2/highgui/highgui.hpp>  
//#include <filesystem>
//#include <io.h>
//#include <algorithm>
//
//using namespace cv;
//using namespace std;
//
//
//
///*
//global data
//*/
//int SeqId = 5;
////window size
//int window_size = 80;
//
////thres of edge proposals
//int thres = 200;
//
////overlapped len of bbox
//int overlap_len = 25;
//
//float matching_thres = 0.3;
//float nms_thres = 0.2;
//float nms_resampling_thres = 0.1;
//float tps_lambda = 0.2;
//
////the scaling factor for a proposal, when searching in the background
//float bbox_scaling_factor = 1.2;
//
//// if visualize the edge bboxes
//bool vis_edge_bbox_checked = false;
//
//// input cross domain paths
//string infe_dir = "./data/EdgeMaps/Seq{SeqId}/infrared";
//string vise_dir = "./data/EdgeMaps/Seq{SeqId}/visible";
//
//string inf_dir = "./data/Preprocessed/Seq" + to_string(SeqId) + "/infrared";
//string vis_dir = "./data/Preprocessed/Seq" + to_string(SeqId) + "/visible_color";
//
//string infe_path;
//string vise_path;
//
//void listdir(string path, vector<string> &files)
//{
//	intptr_t hFile = 0;
//
//	struct _finddata_t fileinfo;
//	string p;
//	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
//	{
//		do
//		{
//			if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
//			{
//				files.push_back(fileinfo.name);
//			}
//		} while (_findnext(hFile, &fileinfo) == 0);
//		_findclose(hFile);
//	}
//}
//
//bool cmp(float a, float b)
//{
//	return a > b;
//}
//
//vector<float> topk(Mat mat, int k)
//{
//	mat = mat.reshape(1, 1);
//	vector<float> array;
//	for (int i = 0; i < mat.size().width; i++)
//	{
//		array.push_back(mat.at<float>(0, i));
//	}
//
//	sort(array.begin(), array.end(), cmp);
//	if (array.size() > k)
//	{
//		
//		vector<float> new_array;
//		for (int i = 0; i < k; i++)
//		{
//			new_array.push_back(array[i]);
//		}
//		array = new_array;
//	}
//
//	return array;
//}
//
//void multiScaleMatchTemplate(Mat foreground, Mat background, int num_scales, float precision,           //default: num_scale=5, precision=0.1
//							float &scaleMaxVal, Point &scaleMaxLoc, Mat &scaleMaxTpl, float &scaleMaxDif)
//{
//	/*
//	"""
//    Multi-scale template matching.
//
//    Parameters
//    ----------
//    num_scales : the number of scales employing template matching.
//    precision : the step between different scales.
//    ----------
//    """
//	*/
//
//	float scale_start = 1.0 - precision * (num_scales / 2);
//	float scale_end = 1.0 + precision * (num_scales / 2);
//	float distance = (scale_end - scale_start) / (num_scales - 1);
//	float val = scale_start;
//	vector<float> scales;
//	for (int cnt = 0; cnt < num_scales; val += distance, cnt++)
//	{
//		scales.push_back(val);
//	}
//	scaleMaxVal = -1;
//
//	for (float scale : scales)
//	{
//		// resize the template using a certain scale value
//		Mat tpl;
//		resize(foreground, tpl, Size(min(int(foreground.size().width * scale), background.size().width),
//									min(int(foreground.size().height * scale), background.size().height)));
//		Mat result;
//		matchTemplate(background, tpl, result, cv::TM_CCORR_NORMED);
//
//		// Good matches are expected to be unique.
//		vector<float> result_topk = topk(result, 5);
//		float topk_diff = 0;
//		for (int i = 1; i < result_topk.size(); i++)
//		{
//			topk_diff += (abs(result_topk[i] - result_topk[i - 1]));
//		}
//		double minVal, maxVal;
//		Point minLoc, maxLoc;
//		minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
//		if (maxVal > scaleMaxVal)
//		{
//			scaleMaxDif = topk_diff;
//			scaleMaxVal = maxVal;
//			scaleMaxLoc = maxLoc;
//			scaleMaxTpl = tpl.clone();
//		}
//
//	}
//}
//
//vector<int> argsort(vector<float> &v, int st = -1) // default st = -1, st == -1 dec, st == 0 asc
//{
//	vector<pair<float, int>> tmp;
//	for (int i = 0; i < v.size(); i++)
//	{
//		tmp.push_back(make_pair(v[i], i));
//	}
//	sort(tmp.begin(), tmp.end());
//	vector<int> ret;
//	if (st == 0)
//	{
//		for (int i = 0; i < v.size(); i++)
//		{
//			ret.push_back(tmp[i].second);
//		}
//	}
//	else
//	{
//		for (int i = v.size() - 1; i >= 0; i--)
//		{
//			ret.push_back(tmp[i].second);
//		}
//	}
//	return ret;
//}
//
//vector<int> nms(Mat &boxes, vector<float> &scores, float nms_thr)
//{
//	Mat x1 = boxes.col(0).clone().reshape(1, 1);
//	Mat y1 = boxes.col(1).clone().reshape(1, 1);
//	Mat x2 = boxes.col(2).clone().reshape(1, 1);
//	Mat y2 = boxes.col(3).clone().reshape(1, 1);
//
//	Mat areas = (x2 - x1 + 1).mul(y2 - y1 + 1);
//	vector<int> order = argsort(scores, -1);
//
//	vector<int> keep;
//	while (order.size() > 0)
//	{
//		int i = order[0];
//		keep.push_back(i);
//		Mat xx1(1, order.size() - 1, CV_32SC1);
//		Mat yy1(1, order.size() - 1, CV_32SC1);
//		Mat xx2(1, order.size() - 1, CV_32SC1);
//		Mat yy2(1, order.size() - 1, CV_32SC1);
//		for (int j = 0; j < order.size() - 1; j++)
//		{
//			xx1.at<int>(j) = max(x1.at<int>(i), x1.at<int>(order[j + 1]));
//			yy1.at<int>(j) = max(y1.at<int>(i), y1.at<int>(order[j + 1]));
//			xx2.at<int>(j) = min(x2.at<int>(i), x2.at<int>(order[j + 1]));
//			yy2.at<int>(j) = min(y2.at<int>(i), y2.at<int>(order[j + 1]));
//		}
//
//		Mat w = xx2 - xx1 + 1, h = yy2 - yy1 + 1;
//		for (int j = 0; j < w.size().width; j++)
//		{
//			w.at<int>(j) = max(0, w.at<int>(j));
//			h.at<int>(j) = max(0, h.at<int>(j));
//		}
//		Mat inter = w.mul(h);
//		Mat areas_add(inter.size().height, inter.size().width, CV_32SC1, Scalar(areas.at<int>(i)));
//		for (int j = 0; j < inter.size().width; j++)
//		{
//			areas_add.at<int>(j) += areas.at<int>(order[j + 1]);
//		}
//		inter.convertTo(inter, CV_32FC1);
//		areas_add.convertTo(areas_add, CV_32FC1);
//		Mat ovr = inter / (areas_add - inter);
//		vector<int> neworder;
//		for (int j = 0; j < ovr.size().width; j++)
//		{
//			if (ovr.at<float>(j) <= nms_thr)
//			{
//				int ind = j;
//				neworder.push_back(order[ind + 1]);
//			}
//		}
//		order = neworder;
//	}
//
//	return keep;
//}
//
//
//Mat gen_bboxes_from_edge_nms(Mat edge_map)
//{
//	/*
//		Generating a bbox at the center of an edge pixel(`thres` as the threshold).
//		The window size is determined by `ws`.
//	*/
//	int H = edge_map.size().height, W = edge_map.size().width;
//	vector<int> indx, indy;
//	vector<float> weights;
//	for (int i = 0; i < H; i++)
//	{
//		for (int j = 0; j < W; j++)
//		{
//			if (edge_map.at<uchar>(i, j) > thres)
//			{
//				indx.push_back(i);
//				indy.push_back(j);
//				weights.push_back(edge_map.at<uchar>(i, j) / 255.0);
//			}
//		}
//	}
//	
//
//	vector<pair<int, int>> edge_centers;
//	for (int i = 0; i < indx.size(); i++)
//	{
//		edge_centers.push_back(make_pair(indx[i], indy[i]));
//	}
//	
//	Mat bboxes = Mat::zeros(edge_centers.size(), 10, CV_32SC1);
//	for (int i = 0; i < edge_centers.size(); i++)
//	{
//		bboxes.at<int>(i, 0) = max(0, edge_centers[i].first - window_size / 2);
//		bboxes.at<int>(i, 1) = max(0, edge_centers[i].second - window_size / 2);
//		bboxes.at<int>(i, 2) = min(max(0, edge_centers[i].first + window_size / 2), H - 1);
//		bboxes.at<int>(i, 3) = min(max(0, edge_centers[i].second + window_size / 2), W - 1);
//		bboxes.at<int>(i, 4) = edge_centers[i].first;
//		bboxes.at<int>(i, 5) = edge_centers[i].second;
//		bboxes.at<int>(i, 6) = (bboxes.at<int>(i, 0) + bboxes.at<int>(i, 2)) / 2;
//		bboxes.at<int>(i, 7) = (bboxes.at<int>(i, 1) + bboxes.at<int>(i, 3)) / 2;
//		bboxes.at<int>(i, 8) = bboxes.at<int>(i, 2) - bboxes.at<int>(i, 0);
//		bboxes.at<int>(i, 9) = bboxes.at<int>(i, 3) - bboxes.at<int>(i, 1);
//	}
//
//	
//	// do NMS for generated bboxes.
//	vector<int> nms_indices = nms(bboxes, weights, nms_thres);
//	
//	Mat bboxes_nms(nms_indices.size(), 10, CV_32SC1);
//	for (int i = 0; i < nms_indices.size(); i++)
//	{
//		bboxes.row(nms_indices[i]).copyTo(bboxes_nms.row(i));
//	}
//	
//	/* hard to code
//	# visualization
//		if vis_edge_bbox_checked:
//			edge_map = np.stack([edge_map, edge_map, edge_map], axis = -1)
//		for b in bboxes_nms :
//			cv2.rectangle(edge_map, (b[1], b[0]), (b[3], b[2]), (0, 0, 255), 2)
//
//		saveShow(edge_map, 'Edge BBoxes')
//		# cv2.waitKey(0)
//	*/
//	return bboxes_nms;
//}
//
//void showimg(Mat img)
//{
//	imshow("img", img);
//	waitKey(0);
//}
//
//vector<int> bbox_scaling(Mat bbox, float factor, int maxH, int maxW)
//{
//	/*
//	Scaling the `bbox` according to `factor`, maximum height and width
//    are constrained by `maxH` and `maxW`, respectively.
//	*/
//
//	int wh = bbox.at<int>(8) * factor, ww = bbox.at<int>(9) * factor;
//	int cx = bbox.at<int>(6), cy = bbox.at<int>(7);
//	//# 0,1: upper-left corner; 2,3: bottom-right corner; 4,5: edge point; 6,7: center point; 8,9: height, width;
//	vector<int> ret(10);
//	ret[0] = max(cx - wh / 2, 0), ret[1] = max(cy - ww / 2, 0), ret[2] = min(cx + wh / 2, maxH - 1);
//	ret[3] = min(cy + ww / 2, maxW - 1), ret[4] = bbox.at<int>(4);
//	ret[5] = bbox.at<int>(5), ret[6] = cx, ret[7] = cy, ret[8] = wh, ret[9] = ww;
//	return ret;
//}
//
//
//pair<Mat, Mat> resampling_keypoints(Mat edge_map1, Mat edge_map2, Mat b1, Mat b2, int maxW = 1, int maxH = 1)
//{
//	/*
//	"""
//    Sampling key points from edge pixels of registered two bboxes (`b1` and `b2`).
//
//    """
//	*/
//	int local_ws = 10;
//
//	//# handling the boundary cases
//	int shift_x = max(b2.at<int>(2) - maxH, 0);
//	int shift_y = max(b2.at<int>(3) - maxW, 0);
//
//	//# obtaining pixel indices of the intersected region
//	Mat edge_map1_rect = edge_map1(Range(b1.at<int>(0), b1.at<int>(2) - shift_x), Range(b1.at<int>(1), b1.at<int>(3) - shift_y));
//	Mat edge_map2_rect = edge_map2(Range(b2.at<int>(0), b2.at<int>(2)), Range(b2.at<int>(1), b2.at<int>(3)));
//	Mat edge_map1_mask = Mat(edge_map1_rect.size().height, edge_map1_rect.size().width, CV_8UC1);
//	Mat edge_map2_mask = Mat(edge_map2_rect.size().height, edge_map2_rect.size().width, CV_8UC1);
//	for (int i = 0; i < edge_map1_rect.size().height; i++)
//	{
//		for (int j = 0; j < edge_map1_rect.size().width; j++)
//		{
//			edge_map1_mask.at<uchar>(i, j) = (edge_map1_rect.at<uchar>(i, j) > thres ? 1 : 0);
//		}
//	}
//	for (int i = 0; i < edge_map2_rect.size().height; i++)
//	{
//		for (int j = 0; j < edge_map2_rect.size().width; j++)
//		{
//			edge_map2_mask.at<uchar>(i, j) = (edge_map2_rect.at<uchar>(i, j) > thres ? 1 : 0);
//		}
//	}
//
//	Mat logical_and = Mat(min(edge_map1_mask.size().height, edge_map2_mask.size().height),
//		min(edge_map1_mask.size().width, edge_map2_mask.size().width), CV_8UC1);
//	vector<int> indices[2];
//	vector<pair<float, float>> edge_centers;
//	for (int i = 0; i < logical_and.size().height; i++)
//	{
//		for (int j = 0; j < logical_and.size().width; j++)
//		{
//			if (edge_map1_mask.at<uchar>(i, j) + edge_map2_mask.at<uchar>(i, j) == 2)
//			{
//				logical_and.at<uchar>(i, j) = 1;
//				indices[0].push_back(i);
//				indices[1].push_back(j);
//				edge_centers.push_back(make_pair(i, j));
//			}
//			else
//			{
//				logical_and.at<uchar>(i, j) = 0;
//			}
//		}
//	}
//	
//
//	//# shift the intersection indices w.r.t the whole edge map
//	Mat edge_centers1 = Mat(edge_centers.size(), 2, CV_32FC1, Scalar(0));
//	for (int i = 0; i < edge_centers.size(); i++)
//	{
//		edge_centers1.at<float>(i, 0) = edge_centers[i].first + b1.at<int>(0);
//		edge_centers1.at<float>(i, 1) = edge_centers[i].second + b1.at<int>(1);
//	}
//
//	Mat edge_centers2 = Mat(edge_centers.size(), 2, CV_32FC1, Scalar(0));
//	for (int i = 0; i < edge_centers.size(); i++)
//	{
//		edge_centers2.at<float>(i, 0) = edge_centers[i].first + b2.at<int>(0);
//		edge_centers2.at<float>(i, 1) = edge_centers[i].second + b2.at<int>(1);
//	}
//
//
//	//# the weights for the NMS
//	vector<float> weights;
//	for (int i = 0; i < indices[0].size(); i++)
//	{
//		int x = indices[0][i], y = indices[1][i];
//		weights.push_back((edge_map1.at<uchar>(x, y) + edge_map2.at<uchar>(x, y)) / (255. * 2));
//	}
//	
//
//	Mat edge_bboxes1 = Mat(edge_centers1.size().height, 4, CV_32FC1, Scalar(0));
//	for (int i = 0; i < edge_bboxes1.size().height; i++)
//	{
//		edge_bboxes1.at<float>(i, 0) = max((float)0, edge_centers1.at<float>(i, 0) - local_ws / 2);
//		edge_bboxes1.at<float>(i, 1) = max((float)0, edge_centers1.at<float>(i, 1) - local_ws / 2);
//		edge_bboxes1.at<float>(i, 2) = min(max((float)0, edge_centers1.at<float>(i, 0) + local_ws / 2), (float)b1.at<int>(2));
//		edge_bboxes1.at<float>(i, 3) = min(max((float)0, edge_centers1.at<float>(i, 1) + local_ws / 2), (float)b1.at<int>(3));
//	}
//	edge_bboxes1.convertTo(edge_bboxes1, CV_32SC1); 
//
//
//	//# do NMS for sampled points
//	vector<int> nms_indices = nms(edge_bboxes1, weights, nms_resampling_thres);
//	Mat edge_centers_nms1(nms_indices.size(), 2, CV_32FC1), edge_centers_nms2(nms_indices.size(), 2, CV_32FC1);
//	for (int i = 0; i < nms_indices.size(); i++)
//	{
//		edge_centers1.row(nms_indices[i]).copyTo(edge_centers_nms1.row(i));
//		edge_centers2.row(nms_indices[i]).copyTo(edge_centers_nms2.row(i));
//	}
//
//
//	//# normalize the sampled points
//	for (int i = 0; i < edge_centers_nms1.size().height; i++)
//	{
//		edge_centers_nms1.at<float>(i, 0) /= maxH;
//		edge_centers_nms1.at<float>(i, 1) /= maxW;
//		edge_centers_nms2.at<float>(i, 0) /= maxH;
//		edge_centers_nms2.at<float>(i, 1) /= maxW;
//	}
//
//	cout << edge_centers_nms1 << endl;
//	return make_pair(edge_centers_nms1, edge_centers_nms2);
//
//}
//
//int main()
//{
//	//vector<string> inf_dir_listdir;
//	//listdir(inf_dir, inf_dir_listdir);
//	//sort(inf_dir_listdir.begin(), inf_dir_listdir.end());
//
//	//infe_path = "./kkx2.JPG";
//	//vise_path = "./kkx2.jpg";
//
//	//Mat im_infe = imread(infe_path);
//	//cvtColor(im_infe, im_infe, COLOR_BGR2RGB);
//	//Mat im_infe_gray;
//	//cvtColor(im_infe, im_infe_gray, COLOR_RGB2GRAY);
//
//	//Mat im_vise = imread(vise_path);
//	//cvtColor(im_vise, im_vise, COLOR_BGR2RGB);
//	//Mat im_vise_gray;
//	//cvtColor(im_vise, im_vise_gray, COLOR_RGB2GRAY);
//
//	//int W = im_infe.size().width;
//	//int H = im_infe.size().height;
//	//Mat background = im_vise_gray.clone();
//
//	//float scaleMaxVal;
//	//Point scaleMaxLoc;
//	//Mat scaleMaxTpl;
//	//float scaleMaxDif;
//	//multiScaleMatchTemplate(im_infe_gray, background, 5, 0.1, scaleMaxVal, scaleMaxLoc, scaleMaxTpl, scaleMaxDif);
//	//vector<int> rough_match_bbox;
//	//rough_match_bbox.push_back(scaleMaxLoc.x);
//	//rough_match_bbox.push_back(scaleMaxLoc.y);
//	//rough_match_bbox.push_back(scaleMaxTpl.size().height);
//	//rough_match_bbox.push_back(scaleMaxTpl.size().width);
//
//	//int tH = scaleMaxTpl.size().height, tW = scaleMaxTpl.size().width;
//	//int i_l = max(scaleMaxLoc.y, 0), j_l = max(scaleMaxLoc.x, 0);
//	//int i_r = min(scaleMaxLoc.y + tH, H), j_r = min(scaleMaxLoc.x + tW, W);
//	//Mat background_rm = im_vise_gray(Range(i_l, i_r), Range(j_l, j_r));
//	//
//
//	////	// obtaining proposals `bboxes` at the centers of edge of the infrared image
//	//Mat foreground_e = scaleMaxTpl.clone();
//	//Mat bboxes = gen_bboxes_from_edge_nms(foreground_e);
//
//
//	//map<string, vector<vector<float>> > matched_points;
//	//matched_points["inf"] = vector<vector<float>>();
//	//matched_points["vis"] = vector<vector<float>>();
//	//int scaleMaxH = background_rm.size().height, scaleMaxW = background_rm.size().width;
//	//
//	//Mat b;
//	//bboxes.row(9).copyTo(b);
//	//Mat tpl = foreground_e(Range(b.at<int>(0), b.at<int>(2)), Range(b.at<int>(1), b.at<int>(3)));
//	//vector<int> bg_bbox = bbox_scaling(b, bbox_scaling_factor, background_rm.size().height, background_rm.size().width);
//	//Mat bg = background_rm(Range(bg_bbox[0], bg_bbox[2]), Range(bg_bbox[1], bg_bbox[3]));
//
//	////# fine-grained matching between the local cross-domain windows.
//	//multiScaleMatchTemplate(tpl, bg, 5, 0.05, scaleMaxVal, scaleMaxLoc, scaleMaxTpl, scaleMaxDif);
//	//
//
//	//if (scaleMaxVal > matching_thres)
//	//{
//	//	cout << "IN" << endl;
//	//	scaleMaxLoc.x = scaleMaxLoc.x + bg_bbox[1];
//	//	scaleMaxLoc.y = scaleMaxLoc.y + bg_bbox[0];
//	//	//# resampling more control points from the matched cross-domain edges, in order to
//	//	//# preserve the structure.
//	//	pair<Mat, Mat> tmpret = resampling_keypoints(foreground_e, background_rm, b.colRange(0, 4),
//	//		(Mat_<int>(1, 4) << scaleMaxLoc.y, scaleMaxLoc.x, scaleMaxLoc.y + b.at<int>(8), scaleMaxLoc.x + b.at<int>(9)),
//	//		scaleMaxW, scaleMaxH);
//	//	Mat sampled_point_inf = tmpret.first, sampled_point_vis = tmpret.second;
//	//	
//	//	for (int i = 0; i < sampled_point_inf.size().height; i++)
//	//	{
//	//		vector<float> inf_part, vis_part;
//	//		for (int j = sampled_point_inf.size().width - 1; j >= 0; j--)
//	//		{
//	//			inf_part.push_back(sampled_point_inf.at<float>(i, j));
//	//			vis_part.push_back(sampled_point_vis.at<float>(i, j));
//	//		}
//	//		matched_points["inf"].push_back(inf_part);
//	//		matched_points["vis"].push_back(vis_part);
//	//	}
//
//	//	for (int i = 0; i < matched_points["inf"].size(); i++)
//	//	{
//	//		cout << matched_points["inf"][i][0] << ' ' << matched_points["inf"][i][1] << endl;
//	//	}
//
//	//	for (auto x : rough_match_bbox)
//	//	{
//	//		cout << x << ' ';
//	//	}
//	//	cout << endl;
//
//	//		
//	//}
//	Mat a = Mat(2, 3, CV_32SC1);
//	for (int i = 0; i < 2; i++)
//	{
//		for (int j = 0; j < 3; j++)
//		{
//			a.at<int>(i, j) = i * 3 + j;
//		}
//	}
//	//Mat res;
//	//vector<Mat> mats(3, a);
//	//merge(mats, res);
//	//cout << res.size().height << ' ' << res.size().width << endl;
//	//cout << res << endl;
//
//	cout << a << endl;
//	a(Range(0, 2), Range(0, 2)) = a(Range(0, 2), Range(0, 2)) + 1;
//	cout << a << endl;
//
//	return 0;
//}
