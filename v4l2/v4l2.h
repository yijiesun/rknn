#ifndef V4L2_H
#define V4L2_H
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
#include <linux/videodev2.h>
#include <opencv2/video.hpp>
#include "opencv2/opencv.hpp"  

using namespace std;
using namespace cv;
struct buffer {
	void *start;
	size_t length;
};

class V4L2{
public :
	V4L2();
	~V4L2();
	unsigned int width;
	unsigned int height;
	char dev_name[100];
	char bufSec[100];
	int fd; 
	struct buffer *buffers;
	unsigned int n_buffers;
	void init(const char *dev,int wid,int hgt,int box_max_cnt = 100);
	void errno_exit(const char *s);
	int xioctl(int fh, int request, void *arg);
	int read_frame_argb(unsigned int *out,Mat &rgb,unsigned int screen_wid,unsigned int pos_x,unsigned int pos_y);
	int read_frame(Mat &out);
	void stop_capturing(void);
	void start_capturing(void);
	void uninit_device(void);
	void init_mmap(void);
	void init_device(void);
	void open_device(void);
	void close_device(void);
	void yuyv_to_bgr(unsigned char* yuv,unsigned char* rgb,int width, int height );
	void yuyv_to_rgb_screen(unsigned char* yuv,unsigned int* argb,unsigned char* rgb,unsigned int width, unsigned int height,unsigned int res_wid,unsigned int pos_x,unsigned int pos_y );
	void mat_to_argb(unsigned char *bgr,unsigned int *argb,unsigned int width, unsigned int height,unsigned int res_wid,unsigned int pos_x,unsigned int pos_y);
};
#endif