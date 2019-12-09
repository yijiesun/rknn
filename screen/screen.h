
#ifndef SCREEN_H
#define SCREEN_H
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <getopt.h> /* getopt_long() */
#include <fcntl.h> /* low-level i/o */
#include <unistd.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <linux/fb.h>
#include <linux/videodev2.h>
#include <opencv2/video.hpp>
#include "opencv2/opencv.hpp"  
#include<vector>

using namespace std;
using namespace cv;
struct draw_box
{
	Point lu;
	Point rd;
	int cnt;
};
class SCREEN {
    public :
        SCREEN();
        ~SCREEN();
        int fb = -1;
        int ret = -1;
        int img_width;
        int img_hgt;
        vector<draw_box>v_draw;
        vector<draw_box>v_draw_[2];
        int draw_box_max_cnt;
        char dev_name[200];
        unsigned int * pfb;
        struct fb_fix_screeninfo finfo;
        struct fb_var_screeninfo vinfo;
        int init(char *dev,int wid,int hgt);
        void uninit();
        void show_bgr_mat_at_screen(Mat &in,int pos_x,int pos_y);
        void draw_line(unsigned int *buf,int x0,int y0,int x1,int y1);
        void refresh_draw_box(unsigned int *buf,unsigned int pos_x,unsigned int pos_y);
        void refresh_draw_box_(unsigned int *buf,unsigned int pos_x,unsigned int pos_y,int pingpong);
};

#endif