#include "knn.h"

#include<algorithm>
using namespace cv;
using namespace std;

#define CLIP(a,b,c) (  (a) = (a)>(c)?(c):((a)<(b)?(b):(a))  )
KNN_BGS::KNN_BGS()
{
	framePixelHistory = NULL;
	frameCnt = 0;
};

KNN_BGS::~KNN_BGS(void)
{
	
}

void KNN_BGS::init()
{
	int gray = 0;
	FGMask.create(IMG_HGT, IMG_WID, CV_8UC1);
	FGMask=Mat::zeros(IMG_HGT,IMG_WID,CV_8UC1);
	frame.create(IMG_HGT,IMG_WID,CV_8UC3);
	frame=Mat::zeros(IMG_HGT,IMG_WID,CV_8UC3);
	FGMask_origin.create(IMG_HGT, IMG_WID, CV_8UC1);
	human_roi.create(IMG_HGT, IMG_WID, CV_8UC1);
	bk_cnt.create(IMG_HGT, IMG_WID, CV_8UC1);
	bk_cnt_cnt.create(IMG_HGT, IMG_WID, CV_8UC1);
	bk_cnt_cnt=Mat::zeros(IMG_HGT,IMG_WID,CV_8UC1);
	bk_cnt=Mat::zeros(IMG_HGT,IMG_WID,CV_8UC1);
	human_roi=Mat::zeros(IMG_HGT,IMG_WID,CV_8UC1);
	bit_and_hotmap_with_diff.create(IMG_HGT, IMG_WID, CV_8UC1);
	DiffMask.create(IMG_HGT, IMG_WID, CV_8UC1);
	process_frame.create(IMG_HGT,IMG_WID,CV_8UC3);
	puzzle_mat=Mat::zeros(IMG_HGT,IMG_WID,CV_8UC3);
	framePixelHistory = (PixelHistory*)malloc(IMG_WID*IMG_HGT * sizeof(PixelHistory));

	for (int i = 0; i < IMG_WID*IMG_HGT; i++)
	{
		framePixelHistory[i].gray = (unsigned char*)malloc((history_num) * sizeof(unsigned char));
		framePixelHistory[i].IsBG = (unsigned char*)malloc((history_num) * sizeof(unsigned char));
		memset(framePixelHistory[i].gray, 0, (history_num ) * sizeof(unsigned char));
		memset(framePixelHistory[i].IsBG, 0, (history_num ) * sizeof(unsigned char));
	}
	hot_map.create(IMG_HGT,IMG_WID,CV_8UC1);
	hot_map=Mat::ones(IMG_HGT,IMG_WID,CV_8UC1)+100;
}

void KNN_BGS::knn_core()
{
	knn_use_box.clear();
	cout<<"knn_core"<<endl;
	cv::cvtColor(frame, fgray, CV_BGR2GRAY);
	cout<<"CV_BGR2GRAY"<<endl;
	FGMask_origin.setTo(Scalar(255));
	int gray = 0;
	for (int i = 0; i < IMG_HGT; i++)
	{
		for (int j = 0; j < IMG_WID; j++)
		{
			bool update_BG = false;
			gray = fgray.at<unsigned char>(i, j);
			int fit = 0;
			int fit_bg = 0;

			for (int n = 0; n < history_num; n++)
			{
				if (fabs((float)gray - framePixelHistory[i*IMG_WID + j].gray[n]) < 20 /* hot_map_thresh.data[i*IMG_WID + j] */  )
				{
						fit++;
					if (framePixelHistory[i*IMG_WID + j].IsBG[n])
					{
						fit_bg++;
					}
				}
			}
			if (fit_bg >= knnv)
			{
				FGMask_origin.at<unsigned char>(i, j) = 0;
			}
			if(hot_map.data[i*IMG_WID + j] == 0 && human_roi.data[i*IMG_WID + j]==0)
			{
				FGMask_origin.at<unsigned char>(i, j) = 0;
				//bk_cnt.at<unsigned char>(i, j)++;
			}

			int index = frameCnt % history_num;
			framePixelHistory[i*IMG_WID + j].gray[index] = gray;
			framePixelHistory[i*IMG_WID + j].IsBG[index] = fit >= knnv ? 1 : 0;

			if(bk_cnt.data[i*IMG_WID + j]>=5)
			{
				if(hot_map.data[i*IMG_WID + j] >=100)
				{
					bk_cnt_cnt.data[i*IMG_WID + j]++;
					hot_map.data[i*IMG_WID + j] = 100;
				}
				if(fit>=2)
				{
					bk_cnt.data[i*IMG_WID + j] = 0;
					bk.data[i*IMG_WID*3 + j*3]= frame.data[i*IMG_WID*3 + j*3];
					bk.data[i*IMG_WID*3 + j*3+1]= frame.data[i*IMG_WID*3 + j*3+1];
					bk.data[i*IMG_WID*3 + j*3+2]= frame.data[i*IMG_WID*3 + j*3+2];
				}
			}
			if(bk_cnt_cnt.data[i*IMG_WID + j]>=2)
			{
				hot_map.data[i*IMG_WID + j] = 0;
				bk_cnt.data[i*IMG_WID + j] = 0;
				human_roi.data[i*IMG_WID + j]=0;
			}

		}
	}
	frameCnt++;
}

void KNN_BGS::postTreatment(Mat &mat)
{
	cv::medianBlur(mat, mat, 3);
	threshold(mat, mat, 2, 255, CV_THRESH_BINARY);
	Mat element = getStructuringElement(MORPH_RECT, Size(dilateRatio, dilateRatio));
	//erode(mat, mat, element);
	dilate(mat, mat, element);
}

bool sortFun(const cv::Rect &p1, const cv::Rect &p2)
{
	return p1.width * p1.height > p2.width * p2.height;
}
bool sortFunY(const REC_BOX &p1, const REC_BOX &p2)
{
	return p1.rec.height > p2.rec.height;
}
void KNN_BGS::getTopRects(vector<Rect> &rects0, vector<REC_BOX> &rects)
{
	for (int i = 0; i<rects0.size(); i++)
	{
		if (i >= useTopRect)
			break;
		REC_BOX tmp;
		tmp.rec = rects0[i];
		tmp.have_box = 0;
		CLIP(tmp.rec.x, 0, (IMG_WID - 1));
		CLIP(tmp.rec.y, 0, (IMG_HGT - 1));
		CLIP(tmp.rec.width, 1, (IMG_WID - 1 - tmp.rec.x));
		CLIP(tmp.rec.height, 1, (IMG_HGT - 1 - tmp.rec.y));
		rects.push_back(tmp);
	}
}

void KNN_BGS::addBoxToRecs()
{
	vector<Box>::iterator it;
	for(it=knn_use_box.begin();it!=knn_use_box.end();)
	{
		it->show_cnt++;
		REC_BOX rect_tmp;
		rect_tmp.rec.x =it->x0;
		rect_tmp.rec.y = it->y0;
		rect_tmp.rec.width = it->x1 - it->x0;
		rect_tmp.rec.height = it->y1 - it->y0;
		rect_tmp.have_box = 0;
		CLIP(rect_tmp.rec.x,0,IMG_WID-1);
		CLIP(rect_tmp.rec.y,0,IMG_HGT-1);
		CLIP(rect_tmp.rec.width,1,IMG_WID-1-rect_tmp.rec.x);
		CLIP(rect_tmp.rec.height,1,IMG_HGT-1-rect_tmp.rec.y);
		
		if((it->show_cnt>=knn_box_exist_cnt)||(rect_tmp.rec.width>IMG_WID/2 || rect_tmp.rec.height >IMG_HGT/2))
			it=knn_use_box.erase(it);
		else
		{
			boundRect.push_back(rect_tmp);
			it++;
		}
			
	}

}

void KNN_BGS::add_diff_in_box_to_mask(vector<Box> &box)
{

	for (int i = 0; i<box.size(); i++) {

		box[i].show_cnt =0;
		knn_use_box.push_back(box[i]);
		int x0 = box[i].x0-2*padSize;
		int y0 = box[i].y0-2*padSize;
		int w0 = box[i].x1-box[i].x0+4*padSize;
		int h0 = box[i].y1-box[i].y0+4*padSize;
		CLIP(x0,0,IMG_WID-1);
		CLIP(y0,0,IMG_HGT-1);
		CLIP(w0,1,IMG_WID-x0 -1);
		CLIP(h0,1,IMG_HGT-y0 -1);
		if(w0>IMG_WID/2 || h0>IMG_HGT/2)
			continue;
		//make hot map
		Mat	tmp= hot_map(cv::Rect(x0,y0,w0,h0));
		tmp.convertTo(tmp, tmp.type(), 1, 255);	
		tmp= human_roi(cv::Rect(x0,y0,w0,h0));
		tmp.convertTo(tmp, tmp.type(), 1, 255);	
		
		tmp= bk_cnt(cv::Rect(x0,y0,w0,h0));
		tmp.convertTo(tmp, tmp.type(), 1, -255);	
		tmp= bk_cnt_cnt(cv::Rect(x0,y0,w0,h0));
		tmp.convertTo(tmp, tmp.type(), 1, -255);	
	}
	hot_map_noraml = hot_map -100;
	hot_map_noraml+=254;
	hot_map_noraml-=254;
	hot_map_noraml*=255;
     
	bitwise_and(DiffMask,hot_map_noraml,bit_and_hotmap_with_diff);
	bitwise_or(FGMask_origin,bit_and_hotmap_with_diff,FGMask_origin);

	
	bitwise_and(human_roi,hot_map,senser_roi_down100);

	senser_roi_down100 = 100 -senser_roi_down100;
	bitwise_and(senser_roi_down100,human_roi,senser_roi_down100);
	senser_roi_down100*=255;

	bitwise_not(senser_roi_down100,senser_roi_down100_not);
	bitwise_and(hot_map,senser_roi_down100_not,senser_roi_down100_not);
	senser_roi_down100-=100;
	hot_map = senser_roi_down100_not+senser_roi_down100;

	senser_roi_down100-=154;
	bk_cnt+=senser_roi_down100;

}
void KNN_BGS::processRects(vector<Box> &box)
{
	add_diff_in_box_to_mask(box);
	FGMask = FGMask_origin.clone();
	 postTreatment(FGMask);

	std::vector<cv::Rect> boundRectTmp;
	vector<vector<Point>> contours;
	vector<Vec4i> hierarcy;
	findContours(FGMask.clone(), contours, hierarcy, CV_RETR_EXTERNAL, CHAIN_APPROX_NONE);
	int rec_nums = buildRecsFromContors(contours, boundRectTmp);
	std::sort(boundRectTmp.begin(), boundRectTmp.end(), sortFun);
	getTopRects(boundRectTmp, boundRect);
	addBoxToRecs();

	//mergeRecs(boundRect, knn_over_percent);

#if 1
		vector<REC_BOX>::iterator it;
	for(it=boundRect.begin();it!=boundRect.end();it++)
	{
		//up
		while(1)
		{
			int x0,y0,w0,h0;
			x0 = it->rec.x;
			y0 = it->rec.y-1;
			w0 = it->rec.width;
			h0 = it->rec.height;
			if(y0<=0) break;
			Mat tmp=DiffMask(Rect(x0,y0,w0,1))/255;
			int zero_cnt = countNonZero(tmp);
			//cout<<"uzero_cnt "<<zero_cnt<<" w0/3 "<<w0/3<<endl;
			if(zero_cnt==0) break;
			it->rec.y -=1;

		}
		//down
		while(1)
		{
			int x0,y0,w0,h0;
			x0 = it->rec.x;
			y0 = it->rec.y;
			w0 = it->rec.width;
			h0 = it->rec.height+1;
			if(y0+h0>=IMG_HGT-2) break;
			Mat tmp=DiffMask(Rect(x0,y0+h0,w0,1))/255;
			int zero_cnt = countNonZero(tmp);
			//cout<<"dzero_cnt "<<zero_cnt<<" w0/3 "<<w0/3<<endl;
			if(zero_cnt==0) break;
			it->rec.height +=1;
		}
		//left
		while(1)
		{
			int x0,y0,w0,h0;
			x0 = it->rec.x-1;
			y0 = it->rec.y;
			w0 = it->rec.width;
			h0 = it->rec.height;
			if(x0<=0) break;
			Mat tmp=DiffMask(Rect(x0,y0,1,h0))/255;
			int zero_cnt = countNonZero(tmp);
			//cout<<"lzero_cnt "<<zero_cnt<<" w0/3 "<<w0/3<<endl;
			if(zero_cnt==0) break;
			it->rec.x -=1;
		}
		//right
		while(1)
		{
			int x0,y0,w0,h0;
			x0 = it->rec.x;
			y0 = it->rec.y;
			w0 = it->rec.width+1;
			h0 = it->rec.height;
			if(x0+w0>=IMG_WID-2) break;
			Mat tmp=DiffMask(Rect(x0+w0,y0,1,h0))/255;
			int zero_cnt = countNonZero(tmp);
			//cout<<"rzero_cnt "<<zero_cnt<<" w0/3 "<<w0/3<<endl;
			if(zero_cnt==0) break;
			it->rec.width +=1;
		}
		
	}
	
#endif
	clearSmallRecs();
	paddingRecs(boundRect, padSize);
	mergeRecs(boundRect, knn_over_percent);
	clearStrangeRecs();
	boundRectTmp.shrink_to_fit();
	contours.shrink_to_fit();
	hierarcy.shrink_to_fit();

}
void KNN_BGS::clearStrangeRecs()
{
	vector<REC_BOX>::iterator it;
	for(it=boundRect.begin();it!=boundRect.end();)
	{
		if(2*it->rec.width > 3*it->rec.height)
			it=boundRect.erase(it);
		else
			it++;
	}
}
void KNN_BGS::clearSmallRecs()
{
	vector<REC_BOX>::iterator it;
	for(it=boundRect.begin();it!=boundRect.end();)
	{
		if((it->rec.width<tooSmalltoDrop || it->rec.height<tooSmalltoDrop*2) ||(it->rec.width>IMG_WID/2 || it->rec.height>IMG_HGT/2) )
			it=boundRect.erase(it);
		else
			it++;
	}
}
int KNN_BGS::buildRecsFromContors(vector<vector<Point>> &contours, vector<Rect> &rects)
{
	int x0 = 0, y0 = 0, w0 = 0, h0 = 0;
	int  cnt = 0;
	for (int i = 0; i<contours.size(); i++)
	{
		Rect rect_tmp = boundingRect((Mat)contours[i]);
		rects.push_back(rect_tmp);
		cnt++;
	}
	return cnt;
}
void KNN_BGS::knn_puzzle(Mat &frame)
{
	pos_x_in_rec_box.clear();
	int pos_rec = 0;
	int p_x=0,p_y=0;
	int total_wid=0,total_hgt=0;
	Mat out_tmp = Mat::zeros(IMG_HGT,IMG_WID,CV_8UC3);
	std::sort(boundRect.begin(), boundRect.end(), sortFunY);
	if(boundRect.size()>0)
	{
		Mat roi = frame(boundRect[0].rec);
		Mat tmp = out_tmp(cv::Rect(p_x, p_y, boundRect[0].rec.width, boundRect[0].rec.height));
		roi.copyTo(tmp);
		p_x = boundRect[0].rec.width;
		p_y=0;
		total_wid = boundRect[0].rec.width;
		total_hgt = boundRect[0].rec.height;
		pos_rec++;
		pos_x_in_rec_box.push_back(0);
	}
	//顶对其的第二块
	if(boundRect.size()>1)
	{
		Mat roi = frame(boundRect[1].rec);
		Mat tmp = out_tmp(cv::Rect(p_x, p_y, boundRect[1].rec.width, boundRect[1].rec.height));
		roi.copyTo(tmp);
		total_wid += boundRect[1].rec.width;
		pos_rec++;
		pos_x_in_rec_box.push_back(p_x);
	}
	while(pos_rec<=3&&boundRect.size()>pos_rec)
	{
		// if(boundRect.size()>pos_rec)
		// {
		// 	cout<<"boundRect.size() "<<boundRect.size()<<" >pos_rec "<<pos_rec<<endl;
		// 	int pos = find_son_block(boundRect,pos_rec,boundRect[pos_rec-1].width,boundRect[0].height-boundRect[pos_rec-1].height);
		// 	if(pos!=0)
		// 	{
		// 		p_y = boundRect[pos_rec-1].height;
		// 		Mat roi = frame(boundRect[pos]);
		// 		Mat tmp = out_tmp(cv::Rect(p_x, p_y, boundRect[pos].width, boundRect[pos].height));
		// 		roi.copyTo(tmp);
		// 		pos_rec++;
		// 	}
		// }
		// else
		// 	break;
		//顶对其
		if(boundRect.size()>pos_rec)
		{
			p_y = 0;
			p_x+=boundRect[pos_rec-1].rec.width;
			Mat roi = frame(boundRect[pos_rec].rec);
			Mat tmp = out_tmp(cv::Rect(p_x, p_y, boundRect[pos_rec].rec.width, boundRect[pos_rec].rec.height));
			roi.copyTo(tmp);
			total_wid += boundRect[pos_rec].rec.width;
			pos_rec++;
			pos_x_in_rec_box.push_back(p_x);
		}
		else
			break;
		
		
	}
	if(total_wid==0&&total_hgt==0)
	{
		puzzle_mat=Mat::zeros(IMG_HGT,IMG_WID,CV_8UC3);
	}
	else
	{
		Mat tmp = out_tmp(cv::Rect(0,0,total_wid,total_hgt));
		puzzle_mat.create(tmp.size(),CV_8UC3);
		tmp.copyTo(puzzle_mat);
	}

}
int KNN_BGS::find_son_block(vector<Rect> &rects,int begin,int max_wid,int max_hgt)
{
	int pos = begin;
	bool is_found = false;
	vector<Rect> rest;
	vector<Rect>::iterator it;
	Rect found;
	int aaa=0;
	for(it=rects.begin()+begin;it!=rects.end();it++)
	{
		if(!is_found&&(it->width<=max_wid)&&(it->height<=max_hgt))
		{
			found = *it;
			is_found = true;
			continue;
		}
		else
		{
			rest.push_back(*it);
			pos++;
		}
	}
	if(is_found&&pos+1<rects.size())
	{
		Rect tmp = rects[pos];
		rects[pos] = found;
		int res_t = 0;
		for(it=rects.begin()+pos+1;res_t<rest.size()&&it!=rects.end();it++)
		{
			*it = rest[res_t++];
		}
	}
	return is_found?pos:0;
}
void KNN_BGS::mergeRecs(vector<REC_BOX> &rects, float percent)
{
	int len = rects.size();
	int new_len = 0;
	int ptr = 0;
	Rect tmp;

	for (;;)
	{
		if (ptr >= len)
			break;

		for (int i = 0; i < len; i++)
		{
			if (ptr < 0 || ptr >= rects.size() || i < 0 || i >= rects.size())
				break;
			if (i == ptr)
				continue;
			if (DecideOverlap(rects[ptr].rec, rects[i].rec, tmp) >= percent)
			{
				rects[ptr].rec = tmp;
				if (rects.begin() + i <= rects.end())
				{
					rects.erase(rects.begin() + i);
					i--;
				}
				else
					break;
			}

		}
		ptr++;
		len = rects.size();
	}

}

void KNN_BGS::drawRecs(Mat & img, vector<Rect> &rects, const Scalar& color)
{
	int x0 = 0, y0 = 0, w0 = 0, h0 = 0;
	for (int i = 0; i< rects.size(); i++)
	{
		x0 = rects[i].x; 
		y0 = rects[i].y; 
		w0 = rects[i].width; 
		h0 = rects[i].height; 

		if (w0 <= tooSmalltoDrop || h0 <= tooSmalltoDrop)
			continue;
		rectangle(img, Point(x0, y0), Point(x0 + w0, y0 + h0), color, 2, 8);
	}
}

float KNN_BGS::DecideOverlap(const Rect &r1, const Rect &r2, Rect &r3)
{
	//r3 = r1;
	int x1 = r1.x;
	int y1 = r1.y;
	int width1 = r1.width;
	int height1 = r1.height;

	int x2 = r2.x;
	int y2 = r2.y;
	int width2 = r2.width;
	int height2 = r2.height;

	int endx = max(x1 + width1, x2 + width2);
	int startx = min(x1, x2);
	int width = width1 + width2 - (endx - startx);

	int endy = max(y1 + height1, y2 + height2);
	int starty = min(y1, y2);
	int height = height1 + height2 - (endy - starty);

	float ratio = 0.0f;
	float Area, Area1, Area2;

	if (width <= 0 || height <= 0)
		return 0.0f;
	else
	{
		Area = width*height;
		Area1 = width1*height1;
		Area2 = width2*height2;
		ratio = max(Area / (float)Area1, Area / (float)Area2);
		r3.x = startx;
		r3.y = starty;
		r3.width = endx - startx;
		r3.height = endy - starty;
	}

	return ratio;
}

void KNN_BGS::paddingRecs(vector<REC_BOX> &rects, int size)
{
	for (int i = 0; i<rects.size(); i++)
	{
		rects[i].rec.x = min(max(rects[i].rec.x - size, 0), IMG_WID-1);
		rects[i].rec.y = min(max(rects[i].rec.y - size, 0), IMG_HGT-1);
		rects[i].rec.width = rects[i].rec.x + rects[i].rec.width + 2 * size > IMG_WID ? IMG_WID - rects[i].rec.x : rects[i].rec.width + 2 * size;
		rects[i].rec.height = rects[i].rec.y + rects[i].rec.height + 2 * size > IMG_HGT ? IMG_HGT - rects[i].rec.y : rects[i].rec.height + 2 * size;
	}
}

void KNN_BGS::insideDilate(Mat & bimg, Mat & bout, int win_size, int scale)
{
	for (int w = 0 + win_size; w < IMG_WID - win_size; w++)
	{
		for (int h = 0 + win_size; h < IMG_HGT - win_size; h++)
		{
			Point curr(w, h);
			Point refer;
			Mat tmp;
			int l = 0, r = 0, u = 0, d = 0;
			Mat roi_u(bimg, Rect(w, h - win_size, 1, win_size));
			Mat roi_d(bimg, Rect(w, h, 1, win_size));
			Mat roi_l(bimg, Rect(w - win_size, h, win_size, 1));
			Mat roi_r(bimg, Rect(w, h, win_size, 1));
			u = countNonZero(roi_u);
			d = countNonZero(roi_d);
			l = countNonZero(roi_l);
			r = countNonZero(roi_r);
			uchar* data = bout.ptr<uchar>(h);

			if ((u >= scale && d >= scale) || (l >= scale && r >= scale))
				data[w] = 255;
			else
			{
				if(u+d+r+l <= win_size)
					data[w] = 0;
			}
				
		}
	}
}

void KNN_BGS::set(int *conf)
{
	history_num = conf[0];
	knnv = conf[1];
	padSize = conf[2];
}

void KNN_BGS::saveROI()
{
	int x0 = 0, y0 = 0, w0 = 0, h0 = 0;
	for (int i = 0; i< boundRect.size(); i++)
	{
		x0 = boundRect[i].rec.x; 
		y0 = boundRect[i].rec.y; 
		w0 = boundRect[i].rec.width; 
		h0 = boundRect[i].rec.height;

		if (w0 <= tooSmalltoDrop || h0 <= tooSmalltoDrop)
			continue;
		cv::Mat src_roi = frame(cv::Rect(x0, y0, w0, h0));
		std::stringstream ss;
		
		ss << saveAdress << pos << "_"<< i << ".bmp" ;
		std::string s = ss.str();
		imwrite(s, src_roi);
	}
}
void KNN_BGS::diff2(Mat &cur,Mat &las)
{
	Mat cur_gray,las_gray;
	cv::cvtColor(cur,cur_gray, CV_BGR2GRAY);
	cv::cvtColor(las,las_gray, CV_BGR2GRAY);
	absdiff(cur_gray,las_gray,DiffMask);
	threshold( DiffMask, DiffMask, 30, 255 , 0 );
	medianBlur(DiffMask,DiffMask,5);    
	Mat element = getStructuringElement(MORPH_RECT, Size(dilateRatio, dilateRatio));
	//erode(DiffMask, DiffMask, element);
	dilate(DiffMask, DiffMask, element);
	//normalize(out, out, 255, 0, NORM_MINMAX);
}