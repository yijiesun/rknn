#include "screen.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <iomanip>

using namespace std;
using namespace cv;

#define CLIP( a, b,c) ( (a)=(a)<(b)?(b):((a)>(c)?(c):(a))   ) 

SCREEN::SCREEN()
{
        draw_box_max_cnt = 100;
        fb = -1;
        ret = -1;
        pfb = NULL;
}
SCREEN::~SCREEN()
{

}

void SCREEN::uninit()
{
	unsigned int i;
	munmap(pfb, finfo.smem_len);
	free(pfb);
}

int SCREEN::init(char *dev,int wid,int hgt)
{
        img_width = wid;
        img_hgt = hgt;
        strncpy(dev_name,dev,200);
        fb = open(dev_name, O_RDWR);
        if (fb < 0)
        {
            perror("open");
            return -1;
        }
        printf("open %s success \n", dev_name);

        ret = ioctl(fb, FBIOGET_FSCREENINFO, &finfo);

        if (ret < 0)
        {
        perror("ioctl");
            return -1;
        }

        ret = ioctl(fb, FBIOGET_VSCREENINFO, &vinfo);
        if (ret < 0)
        {
            perror("ioctl");
            return -1;
        }
        pfb = (unsigned int *)malloc(finfo.smem_len);
        pfb = (unsigned int *)mmap(NULL, finfo.smem_len, PROT_READ | PROT_WRITE, MAP_SHARED, fb, 0);

        printf("smem_len: %ld", finfo.smem_len);
        if (NULL == pfb)
        {
            perror("mmap");
            return -1;
        }
       
        printf("pfb :0x%x \n", *pfb);
        std::cout << "height: " << vinfo.yres << "weight: "<< vinfo.xres << std::endl;
        std::cout << "xres_virtual: " << vinfo.xres_virtual << "yres_virtual: "<< vinfo.yres_virtual << std::endl;
}

void SCREEN::show_bgr_mat_at_screen(Mat &in,int pos_x,int pos_y)
{
    uint32_t color = 0;
    for (int h=0; h < in.rows; h++){
        for(int w=0;w <in.cols; w++){
            color = (0xff000000) | ((in.data[h*in.cols*3+w*3+2] << 16) & 0x00ff0000) | ((in.data[h*in.cols*3+w*3+1] << 8) & 0x0000ff00) | ((in.data[h*in.cols*3+w*3]&0x000000ff));
            *(pfb+(h+pos_y)*vinfo.xres_virtual+w+pos_x)  = color;
   
        }
    }
}

void SCREEN::draw_line(unsigned int *buf, int x0,int y0,int x1,int y1)
{
    uint32_t color = 0xffff0000;
        for (int h=y0; h <= y1; h++){
            for(int w=x0;w <=x1; w++){
                *(buf+h*vinfo.xres_virtual+w)  = color;
            }
    }
}

void SCREEN::refresh_draw_box(unsigned int *buf,unsigned int pos_x,unsigned int pos_y)
{
	for (vector<draw_box>::iterator it = v_draw.begin(); it != v_draw.end();)
	{
        int  lux = it->lu.x + pos_x;
        int luy = it->lu.y+pos_y;
        int  rdx = it->rd.x + pos_x;
        int rdy = it->rd.y+pos_y;

        CLIP(lux,pos_x+2,pos_x+img_width-2);
        CLIP(luy,pos_y+2,pos_y+img_hgt-2);
        CLIP(rdx,pos_x+2,pos_x+img_width-2);
        CLIP(rdy,pos_y+2,pos_y+img_hgt-2);
        draw_line(buf,lux,luy,rdx,luy); //lu-ru
        draw_line(buf,rdx,luy,rdx,rdy);//ru-rd
        draw_line(buf,lux,rdy,rdx,rdy);//ld-rd
        draw_line(buf,lux,luy,lux,rdy);//lu-ld
        it++;
	}
}
void SCREEN::refresh_draw_box_(unsigned int *buf,unsigned int pos_x,unsigned int pos_y,int pingpong)
{
	for (vector<draw_box>::iterator it = v_draw_[pingpong].begin(); it != v_draw_[pingpong].end();)
	{
        int  lux = it->lu.x + pos_x;
        int luy = it->lu.y+pos_y;
        int  rdx = it->rd.x + pos_x;
        int rdy = it->rd.y+pos_y;

        CLIP(lux,pos_x+2,pos_x+img_width-2);
        CLIP(luy,pos_y+2,pos_y+img_hgt-2);
        CLIP(rdx,pos_x+2,pos_x+img_width-2);
        CLIP(rdy,pos_y+2,pos_y+img_hgt-2);
        draw_line(buf,lux,luy,rdx,luy); //lu-ru
        draw_line(buf,rdx,luy,rdx,rdy);//ru-rd
        draw_line(buf,lux,rdy,rdx,rdy);//ld-rd
        draw_line(buf,lux,luy,lux,rdy);//lu-ld
        it++;
	}
}