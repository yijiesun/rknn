#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <fstream>
#include <atomic>
#include <queue>
#include <thread>
#include <mutex>
#include <chrono>
#include <sys/time.h>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>
#include "v4l2/v4l2.h" 
#include "screen/screen.h"
#include "knn/knn.h"
#include "rknn_api.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>

#include <unistd.h>
#include <sys/syscall.h>
inline pid_t gettid() 
{
  return syscall(__NR_gettid);
}

using namespace std;
using namespace cv;
using namespace std::chrono;

#define QUEUE_SIZE 2
#define NUM_RESULTS         1917
#define NUM_CLASSES         91

#define Y_SCALE  10.0f
#define X_SCALE  10.0f
#define H_SCALE  5.0f
#define W_SCALE  5.0f

#define CLIP(a,b,c) (  (a) = (a)>(c)?(c):((a)<(b)?(b):(a))  )
#define __AVE_TIC__(tag) static int ____##tag##_total_time=0; \
        static int ____##tag##_total_conut=0;\
        timeval ____##tag##_start_time, ____##tag##_end_time;\
        gettimeofday(&____##tag##_start_time, 0);

#define __AVE_TOC__(tag) gettimeofday(&____##tag##_end_time, 0); \
        ____##tag##_total_conut++; \
        ____##tag##_total_time+=((int)____##tag##_end_time.tv_sec-(int)____##tag##_start_time.tv_sec)*1000000+((int)____##tag##_end_time.tv_usec-(int)____##tag##_start_time.tv_usec); \
        std::fprintf(stderr,  #tag ": %d us\n", ____##tag##_total_time/____##tag##_total_conut);

#define __TIC__(tag) timeval ____##tag##_start_time, ____##tag##_end_time;\
        gettimeofday(&____##tag##_start_time, 0);

#define __TOC__(tag) gettimeofday(&____##tag##_end_time, 0); \
        int ____##tag##_total_time=((int)____##tag##_end_time.tv_sec-(int)____##tag##_start_time.tv_sec)*1000000+((int)____##tag##_end_time.tv_usec-(int)____##tag##_start_time.tv_usec); \
        std::fprintf(stderr,  #tag ": %f ms\n", ____##tag##_total_time/1000.0);


bool first_init_knn = true;
int  IMG_WID=640;
int  IMG_HGT=480;
int knn_conf[5] = { 2, 1, 5, 5, 10};
vector<Box>	boxes_all; 
vector<Box>	boxes[2]; 
V4L2 v4l2_;
KNN_BGS knn_bgs;
SCREEN screen_;
unsigned int screen_pos_x,screen_pos_y;
unsigned int * pfb;

int idxInputImage = 0;  // image index of input video
int idxShowImage = 0;   // next frame index to be display
bool bReading = true;   // flag of input
chrono::system_clock::time_point start_time;

typedef pair<int, Mat> imagePair;
class paircomp {
public:
    bool operator()(const imagePair &n1, const imagePair &n2) const {
        if (n1.first == n2.first) return n1.first > n2.first;
        return n1.first > n2.first;
    }
};
mutex mtxBoxes;
mutex mtxKnn;
mutex mtxShowImg;
mutex mtxQueueInput;               // mutex of input queue
mutex mtxQueueShow;                // mutex of display queue
queue<Mat> queueInput;  // input queue
queue<Mat> queueInput_knn;  // input queue knn
queue<Mat> queuePuzzle;

priority_queue<imagePair, vector<imagePair>, paircomp>
        queueShow;  // display queue
Mat show_img;
#ifdef SHOWTIME
#define _T(func)                                                              \
    {                                                                         \
        auto _start = system_clock::now();                                    \
        func;                                                                 \
        auto _end = system_clock::now();                                      \
        auto duration = (duration_cast<microseconds>(_end - _start)).count(); \
        string tmp = #func;                                                   \
        tmp = tmp.substr(0, tmp.find('('));                                   \
        cout << "[TimeTest]" << left << setw(30) << tmp;                      \
        cout << left << setw(10) << duration << "us" << endl;                 \
    }
#else
#define _T(func) func;
#endif

Scalar colorArray[10] = {
        Scalar(139,   0,   0, 255),
        Scalar(139,   0, 139, 255),
        Scalar(  0,   0, 139, 255),
        Scalar(  0, 100,   0, 255),
        Scalar(139, 139,   0, 255),
        Scalar(209, 206,   0, 255),
        Scalar(  0, 127, 255, 255),
        Scalar(139,  61,  72, 255),
        Scalar(  0, 255,   0, 255),
        Scalar(255,   0,   0, 255),
};

float MIN_SCORE = 0.4f;

float NMS_THRESHOLD = 0.45f;

int multi_npu_process_initialized[2] = {0, 0};

int loadLabelName(string locationFilename, string* labels) {
    ifstream fin(locationFilename);
    string line;
    int lineNum = 0;
    while(getline(fin, line))
    {
        labels[lineNum] = line;
        lineNum++;
    }
    return 0;
}

int loadCoderOptions(string locationFilename, float (*boxPriors)[NUM_RESULTS])
{
    const char *d = ", ";
    ifstream fin(locationFilename);
    string line;
    int lineNum = 0;
    while(getline(fin, line))
    {
        char *line_str = const_cast<char *>(line.c_str());
        char *p;
        p = strtok(line_str, d);
        int priorIndex = 0;
        while (p) {
            float number = static_cast<float>(atof(p));
            boxPriors[lineNum][priorIndex++] = number;
            p=strtok(nullptr, d);
        }
        if (priorIndex != NUM_RESULTS) {
            return -1;
        }
        lineNum++;
    }
    return 0;

}

float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1, float ymax1) {
    float w = max(0.f, min(xmax0, xmax1) - max(xmin0, xmin1));
    float h = max(0.f, min(ymax0, ymax1) - max(ymin0, ymin1));
    float i = w * h;
    float u = (xmax0 - xmin0) * (ymax0 - ymin0) + (xmax1 - xmin1) * (ymax1 - ymin1) - i;
    return u <= 0.f ? 0.f : (i / u);
}

float expit(float x) {
    return (float) (1.0 / (1.0 + exp(-x)));
}

void decodeCenterSizeBoxes(float* predictions, float (*boxPriors)[NUM_RESULTS]) {

    for (int i = 0; i < NUM_RESULTS; ++i) {
        float ycenter = predictions[i*4+0] / Y_SCALE * boxPriors[2][i] + boxPriors[0][i];
        float xcenter = predictions[i*4+1] / X_SCALE * boxPriors[3][i] + boxPriors[1][i];
        float h = (float) exp(predictions[i*4 + 2] / H_SCALE) * boxPriors[2][i];
        float w = (float) exp(predictions[i*4 + 3] / W_SCALE) * boxPriors[3][i];

        float ymin = ycenter - h / 2.0f;
        float xmin = xcenter - w / 2.0f;
        float ymax = ycenter + h / 2.0f;
        float xmax = xcenter + w / 2.0f;

        predictions[i*4 + 0] = ymin;
        predictions[i*4 + 1] = xmin;
        predictions[i*4 + 2] = ymax;
        predictions[i*4 + 3] = xmax;
    }
}

int scaleToInputSize(float * outputClasses, int (*output)[NUM_RESULTS], int numClasses)
{
    int validCount = 0;
    // Scale them back to the input size.
    for (int i = 0; i < NUM_RESULTS; ++i) {
        float topClassScore = static_cast<float>(-1000.0);
        int topClassScoreIndex = -1;

        // Skip the first catch-all class.
        for (int j = 1; j < numClasses; ++j) {
            float score = expit(outputClasses[i*numClasses+j]);
            if (score > topClassScore) {
                topClassScoreIndex = j;
                topClassScore = score;
            }
        }

        if (topClassScore >= MIN_SCORE) {
            output[0][validCount] = i;
            output[1][validCount] = topClassScoreIndex;
            ++validCount;
        }
    }

    return validCount;
}

int nms(int validCount, float* outputLocations, int (*output)[NUM_RESULTS])
{
    for (int i=0; i < validCount; ++i) {
        if (output[0][i] == -1) {
            continue;
        }
        int n = output[0][i];
        for (int j=i + 1; j<validCount; ++j) {
            int m = output[0][j];
            if (m == -1) {
                continue;
            }
            float xmin0 = outputLocations[n*4 + 1];
            float ymin0 = outputLocations[n*4 + 0];
            float xmax0 = outputLocations[n*4 + 3];
            float ymax0 = outputLocations[n*4 + 2];

            float xmin1 = outputLocations[m*4 + 1];
            float ymin1 = outputLocations[m*4 + 0];
            float xmax1 = outputLocations[m*4 + 3];
            float ymax1 = outputLocations[m*4 + 2];

            float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

            if (iou >= NMS_THRESHOLD) {
                output[0][j] = -1;
            }
        }
    }

    return 0;
}

void draw_img(Mat &img)
{
    int line_width=300*0.002;
    for(int i=0;i<(int)boxes_all.size();i++)
    {
        Box box=boxes_all[i];
        cv::rectangle(img, cv::Rect(box.x0, box.y0,(box.x1-box.x0),(box.y1-box.y0)),cv::Scalar(255, 255, 0),line_width);
        cout<<"draw_img "<<box.x0<<","<<box.y0<<","<<box.x1<<","<<box.y1<<endl;
        // std::ostringstream score_str;
        // score_str<<box.score;
        // std::string label = "person";
        // int baseLine = 0;
        // cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        // cv::rectangle(img, cv::Rect(cv::Point(box.x0,box.y0- label_size.height),
        //                           cv::Size(label_size.width, label_size.height + baseLine)),
        //               cv::Scalar(255, 255, 0), -1);
        // cv::putText(img, label, cv::Point(box.x0, box.y0),
        //             cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

}

int box_in_which_rec(Box &b0,vector<REC_BOX> &knn_recs_)
{
    int pos = 0;
    float percent = 0;
    vector<REC_BOX>::iterator it;

    Rect tmp;
    for(int i=0;i<knn_recs_.size();i++)
    {
        Rect rbox(b0.x0-knn_bgs.pos_x_in_rec_box[i]+knn_recs_[i].rec.x,b0.y0+knn_recs_[i].rec.y,b0.x1-b0.x0,b0.y1-b0.y0);
        cout<<"box_in_which_rec "<<"boundRect["<<i<<"]:"<<knn_recs_[i].rec.x<<","<<knn_recs_[i].rec.y<<","<<knn_recs_[i].rec.height<<","<<knn_recs_[i].rec.width<<endl;
        float p=knn_bgs.DecideOverlap(rbox,knn_recs_[i].rec,tmp);
        if(p>percent)
        {
            pos = i;
            cout<<"box_in_which_rec "<<"p>percent "<<p<<","<<percent<<"  pos:"<<pos<<endl;
            percent = p;
        }
        if(percent>0.5)
        {
            b0.x0 -= knn_bgs.pos_x_in_rec_box[pos];
            return pos;
        }
        else  
            return -1;
    }
    return -1;
}

void togetherAllBox(int th_num,vector<Box> &b0,vector<Box> &b_all ,vector<REC_BOX> &knn_recs_)
{
  cout<<"togetherAllBox th_num "<<th_num<<", rec size:"<<b0.size()<<endl;
  boxes_all.clear();
    if(th_num==0)
    {
        for (int i = 0; i<b0.size(); i++) {
            float bx0 = b0[i].x0, by0 = b0[i].y0, bx1= b0[i].x1, by1 = b0[i].y1;

            int pos = box_in_which_rec(b0[i],knn_recs_);
            cout<<"togetherAllBox "<<"["<<th_num<<"] x0y0x1y1,pos:"<<bx0<<","<<by0<<","<<bx1<<","<<by1<<"          "<<pos<<endl;
            
            if(pos>=0)
            {
                knn_recs_[pos].have_box = 1;
                cout<<"togetherAllBox "<<"["<<pos<<"] wyhw:"<<knn_recs_[pos].rec.x<<","<<knn_recs_[pos].rec.y<<","<<knn_recs_[pos].rec.height<<","<<knn_recs_[pos].rec.width<<endl;
                b0[i].x0= bx0 + knn_recs_[pos].rec.x;
                b0[i].y0 = by0 + knn_recs_[pos].rec.y;
                b0[i].x1 = bx1 + knn_recs_[pos].rec.x;
                b0[i].y1 = by1 + knn_recs_[pos].rec.y;
                CLIP(b0[i].x0,0,IMG_WID-1);
                CLIP(b0[i].y0,0,IMG_HGT-1);
                CLIP(b0[i].x1,0,IMG_WID-1);
                CLIP(b0[i].y1,0,IMG_HGT-1);
                boxes_all.push_back(b0[i]);
            }
	    }
        for(int i=0;i<knn_recs_.size();i++)
        {

            if(knn_recs_[i].have_box)
            {
                Mat	tmp_hotmap = knn_bgs.hot_map(knn_recs_[i].rec);
                tmp_hotmap.convertTo(tmp_hotmap, tmp_hotmap.type(), 1, 10);
            }
            else
            {
                Mat	tmp_hotmap = knn_bgs.hot_map(knn_recs_[i].rec);
                tmp_hotmap.convertTo(tmp_hotmap, tmp_hotmap.type(), 1, -10);
            }
        }
    }
    else
    {
        for (int i = 0; i<b0.size(); i++) {
            boxes_all.push_back(b0[i]);
        }
    }

}

void knnProcess()
{
  cpu_set_t mask;
  int cpuid = 1;

  CPU_ZERO(&mask);
  CPU_SET(cpuid, &mask);

  if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0)
    cerr << "set thread affinity failed" << endl;

  std::printf("Bind knn process to CPU %d\n", cpuid); 

  while (true) {
    if (first_init_knn)
      break;
    sleep(1);
  }

  while (true) {
    bool do_knn = false;
    Mat pairIndexImage;
    usleep(20000);
    /*********************calc knn*********************/
    mtxQueueInput.lock();
    if (!queueInput.empty()) {
      do_knn = true;
      pairIndexImage = queueInput.front();
      knn_bgs.frame = pairIndexImage.clone();
      queueInput.pop();
    }
    mtxQueueInput.unlock();
    if(do_knn)
    {
      mtxKnn.lock();
      knn_bgs.pos ++;
      knn_bgs.boundRect.clear();
      knn_bgs.diff2(knn_bgs.frame, knn_bgs.bk);
      knn_bgs.knn_core();
      mtxBoxes.lock();
      knn_bgs.processRects(boxes_all);
      mtxBoxes.unlock();

      knn_bgs.knn_puzzle(knn_bgs.frame);

      
      queuePuzzle.push(knn_bgs.puzzle_mat);
      mtxKnn.unlock();
    }

    /*********************calc knn*********************/
  }

}
void cameraRead() 
{
  int initialization_finished = 1;
  cpu_set_t mask;
  int cpuid = 2;

  CPU_ZERO(&mask);
  CPU_SET(cpuid, &mask);

  if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0)
    cerr << "set thread affinity failed" << endl;

  std::printf("Bind CameraCapture process to CPU %d\n", cpuid); 

  while (true) {
    initialization_finished = 1;
    usleep(20000);
    Mat img;
    img.create(480,640,CV_8UC3);
    img = Mat::zeros(480,640,CV_8UC3);
    v4l2_.read_frame(img);
    for (int i = 0; i < sizeof(multi_npu_process_initialized) / sizeof(int); i++) {
      if (multi_npu_process_initialized[i] == 0) {
        initialization_finished = 0;
      }
    }

    if (initialization_finished)
      break;

    sleep(1);
  }

  start_time = chrono::system_clock::now();
        while (true) {
                // read function
                usleep(20000);
                Mat img;
                img.create(480,640,CV_8UC3);
                img = Mat::zeros(480,640,CV_8UC3);
                v4l2_.read_frame(img);
                // camera >> img;
                if (img.empty()) {
                        cerr << "Fail to read image from camera!" << endl;
                        break;
                }
                if(first_init_knn)
                {
                  knn_bgs.bk = img.clone();
                  first_init_knn = false;
                }
                mtxShowImg.lock();
                show_img = img.clone();
                mtxShowImg.unlock();
                mtxQueueInput.lock();
                if(queueInput.size() < QUEUE_SIZE)
                {
                  cout<<"queueInput.push "<<idxInputImage<<endl;
                  queueInput.push( img);
                }
                mtxQueueInput.unlock();

  }
}

void displayImage() {
  // Mat img;
  cpu_set_t mask;
  int cpuid = 3;

  CPU_ZERO(&mask);
  CPU_SET(cpuid, &mask);

  if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0)
    cerr << "set thread affinity failed" << endl;

  std::printf("Bind Display process to CPU %d\n", cpuid); 

    while (true) {
        Mat img_show;
        mtxShowImg.lock();
        img_show = show_img.clone();
        mtxShowImg.unlock();

        mtxBoxes.lock();
        draw_img(img_show);
        // boxes_all.clear();
        mtxBoxes.unlock();

        mtxKnn.lock();
        if(!knn_bgs.bk.empty())
        {
          Mat out,hot_map_color,hot_map_color2,hot_map_thresh_color,hot_map,pm;
          hot_map = knn_bgs.hot_map;

          cv::cvtColor(knn_bgs.FGMask, hot_map_color2, CV_GRAY2BGR);  
          cv::cvtColor(knn_bgs.hot_map, hot_map_thresh_color, CV_GRAY2BGR);  
          hconcat(img_show,knn_bgs.bk,out);
          // resize(knn_bgs.puzzle_mat, pm, img_show.size(), 0, 0,  cv::INTER_LINEAR); 
          // hconcat(img_show,pm,out);
          mtxKnn.unlock();
          hconcat(hot_map_color2,hot_map_thresh_color,hot_map_color2);
          vconcat(out,hot_map_color2,out);
      
          // string no=to_string(frame_cnt);
          // Point siteNo;
          // siteNo.x = 25;
          // siteNo.y = 25;
          // putText( out, no, siteNo, 2,1,Scalar( 255, 0, 0 ), 4);
          resize(out, img_show, img_show.size(), 0, 0,  cv::INTER_LINEAR); 

          v4l2_. mat_to_argb(img_show.data,pfb,640,480,screen_.vinfo.xres_virtual,0,0);
          memcpy(screen_.pfb,pfb,screen_.finfo.smem_len);
        }
        else
        {
          mtxKnn.unlock();
        }
        

    }
}

void run_process(int thread_id, void *pmodel, int model_len) 
{
  #if 1
  __TIC__(NPU_INIT);
  const char *label_path = "/home/firefly/RKNN/rknn-api/Linux/tmp/coco_labels_list.txt";
  const char *box_priors_path = "/home/firefly/RKNN/rknn-api/Linux/tmp/box_priors.txt";
  const int img_width = 300;
  const int img_height = 300;
  const int img_channels = 3;
  const int input_index = 0;      // node name "reprocessor/sub"

  const int output_elems1 = NUM_RESULTS * 4;
  const uint32_t output_size1 = output_elems1 * sizeof(float);
  const int output_index1 = 0;    // node name "concat"

  const int output_elems2 = NUM_RESULTS * NUM_CLASSES;
  const uint32_t output_size2 = output_elems2 * sizeof(float);
  const int output_index2 = 1;    // node name "concat_1"

  cv::Mat resimg;
  Mat	img_roi;
  cpu_set_t mask;
  int cpuid = 0;

  if (thread_id == 0)
  {
   cpuid = 4;
  }
  else if (thread_id == 1)
  {
   cpuid = 5;
  }
  else
    cpuid = 0;

  CPU_ZERO(&mask);
  CPU_SET(cpuid, &mask);

  if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0)
    cerr << "set thread affinity failed" << endl;

  std::printf("Bind NPU process(%d) to CPU %d\n", thread_id, cpuid); 

  // Start Inference
  rknn_input inputs[1];
  rknn_output outputs[2];
  rknn_tensor_attr outputs_attr[2];

  int ret = 0;
  rknn_context ctx = 0;

  ret = rknn_init(&ctx, pmodel, model_len, RKNN_FLAG_PRIOR_MEDIUM);
  if(ret < 0) {
    std::printf("rknn_init fail! ret=%d\n", ret);
    return;
  }

  outputs_attr[0].index = 0;
  ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(outputs_attr[0]), sizeof(outputs_attr[0]));
  if(ret < 0) {
    std::printf("rknn_query fail! ret=%d\n", ret);
    return;
  }

  outputs_attr[1].index = 1;
  ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(outputs_attr[1]), sizeof(outputs_attr[1]));
  if(ret < 0) {
    std::printf("rknn_query fail! ret=%d\n", ret);
    return;
  }

  if (thread_id > sizeof(multi_npu_process_initialized) / sizeof(int) - 1)
    return;

  multi_npu_process_initialized[thread_id] = 1;
  std::printf("The initialization of NPU Process %d has been completed.\n", thread_id);
  __TOC__(NPU_INIT);
#endif
  while (true) {
    __TIC__(thread_id);
      if (thread_id == 0)
      {
          mtxKnn.lock();
          if(!queuePuzzle.empty())
          {
            img_roi = queuePuzzle.front().clone();
            queuePuzzle.pop();
            mtxKnn.unlock();
            for(int i=0;i<knn_bgs.boundRect.size();i++)
            {
              mtxShowImg.lock();
              Mat	img_show = show_img(knn_bgs.boundRect[i].rec);
              img_show.convertTo(img_show, img_show.type(), 1, 30);
              mtxShowImg.unlock();
            }
          }
          else
          {
            mtxKnn.unlock();
            continue;
          }
      }
      else if (thread_id == 1)
      {
        mtxKnn.lock();
        if(!queueInput.empty())
        {
          img_roi = queueInput.front().clone();
          mtxKnn.unlock();
        }
        else
        {
          mtxKnn.unlock();
          continue;
        }
      }
    // // cout<<"process 0"<<endl;
    // pair<int, Mat> pairIndexImage;
    // mtxQueueInput.lock();
    // // cout<<"mtxQueueInput.lock()"<<endl;
    // if (queueInput.empty()) {
    //   mtxQueueInput.unlock();
    //   usleep(20000);
    //   if (bReading)
    //     continue;
    //   else
    //     break;
    //   } else {
    //     // Get an image from input queue
    //     pairIndexImage = queueInput.front();
    //     queueInput.pop();
    //     // mtxQueueInput.unlock();
    //   }
    //   mtxQueueInput.unlock();
      #if 1
      cv::resize(img_roi, resimg, cv::Size(img_width, img_height), (0, 0), (0, 0), cv::INTER_LINEAR);

      inputs[0].index = input_index;
      inputs[0].buf = resimg.data;
      inputs[0].size = img_width * img_height * img_channels;
      inputs[0].pass_through = false;
      inputs[0].type = RKNN_TENSOR_UINT8;
      inputs[0].fmt = RKNN_TENSOR_NHWC;
      ret = rknn_inputs_set(ctx, 1, inputs);
      if(ret < 0) {
          std::printf("rknn_input_set fail! ret=%d\n", ret);
          return;
      }

      ret = rknn_run(ctx, nullptr);
      if(ret < 0) {
          std::printf("rknn_run fail! ret=%d\n", ret);
          return;
      }

      outputs[0].want_float = true;
      outputs[0].is_prealloc = false;
      outputs[1].want_float = true;
      outputs[1].is_prealloc = false;
      ret = rknn_outputs_get(ctx, 2, outputs, nullptr);
      if(ret < 0) {
          std::printf("rknn_outputs_get fail! ret=%d\n", ret);
          return;
      }

      boxes[thread_id].clear();
      if(outputs[0].size == outputs_attr[0].n_elems*sizeof(float) && outputs[1].size == outputs_attr[1].n_elems*sizeof(float))
      {
          float boxPriors[4][NUM_RESULTS];
          string labels[91];

          /* load label and boxPriors */
          loadLabelName(label_path, labels);
          loadCoderOptions(box_priors_path, boxPriors);

          float* predictions = (float*)outputs[0].buf;
          float* outputClasses = (float*)outputs[1].buf;

          int output[2][NUM_RESULTS];

          /* transform */
          decodeCenterSizeBoxes(predictions, boxPriors);

          int validCount = scaleToInputSize(outputClasses, output, NUM_CLASSES);
          //std::printf("validCount: %d\n", validCount);

          if (validCount < 100) {
            /* detect nest box */
            nms(validCount, predictions, output);

            /* box valid detect target */
            for (int i = 0; i < validCount; ++i) {
                if (output[0][i] == -1) {
                    continue;
                }
                int n = output[0][i];
                int topClassScoreIndex = output[1][i];
                if(topClassScoreIndex == 1)
                {
                    Box box;
                    box.x0 = static_cast<float>(predictions[n * 4 + 1] * IMG_WID);
                    box.y0 = static_cast<float>(predictions[n * 4 + 0] * IMG_HGT);
                    box.x1 = static_cast<float>(predictions[n * 4 + 3] * IMG_WID);
                    box.y1 = static_cast<float>(predictions[n * 4 + 2] * IMG_HGT);
                   cout<<"ok "<<box.x0<<","<<box.y0<<","<<box.x1<<","<<box.y1<<endl;
                    CLIP(box.x0,0,IMG_WID-1);
                    CLIP(box.y0,0,IMG_HGT-1);
                    CLIP(box.x1,0,IMG_WID-1);
                    CLIP(box.y1,0,IMG_HGT-1);
                    boxes[thread_id].push_back(box);
                    // string label = labels[topClassScoreIndex];
                    // rectangle(pairIndexImage.second, Point(x1, y1), Point(x2, y2), colorArray[topClassScoreIndex%10], 3);
                    // putText(pairIndexImage.second, label, Point(x1, y1 - 12), 1, 2, Scalar(0, 255, 0, 255));
                }

            }
          } else {
              std::printf("validCount too much!\n");
          }
      }
      else
      {
          std::printf("rknn_outputs_get fail! get outputs_size = [%d, %d], but expect [%lu, %lu]!\n",
              outputs[0].size, outputs[1].size, outputs_attr[0].n_elems*sizeof(float), outputs_attr[1].n_elems*sizeof(float));
      }
      vector<REC_BOX> knn_recs;
      knn_recs.clear();
      mtxKnn.lock();
      knn_recs.assign(knn_bgs.boundRect.begin(),knn_bgs.boundRect.end());
      mtxKnn.unlock();
      mtxBoxes.lock();
      togetherAllBox(thread_id, boxes[thread_id], boxes_all,knn_recs);
      mtxBoxes.unlock();
      rknn_outputs_release(ctx, 2, outputs);
#endif
      // mtxQueueShow.lock();
      // cout<<"mtxQueueShow input "<<queueShow.size()<<endl;
      // Put the processed iamge to show queue
      // queueShow.push(pairIndexImage);
      // mtxQueueShow.unlock();
      __TOC__(thread_id);
  }
}

int main(const int argc, const char** argv) 
{
  screen_pos_x = 0;
	screen_pos_y = 0;
	screen_.init((char *)"/dev/fb0",640,480);
  v4l2_.init("/dev/video8",640,480);
  v4l2_.open_device();
	v4l2_.init_device();
	v4l2_.start_capturing();
  pfb = (unsigned int *)malloc(screen_.finfo.smem_len);
  show_img.create(IMG_HGT,IMG_WID,CV_8UC3);
  show_img = Mat::zeros(IMG_HGT,IMG_WID,CV_8UC3);
  knn_bgs.IMG_WID = IMG_WID;
  knn_bgs.IMG_HGT = IMG_HGT;
  knn_bgs.set(knn_conf);
  knn_bgs.pos = 0;
  knn_bgs.knn_box_exist_cnt = 5;
  knn_bgs.useTopRect = knn_conf[3];
  knn_bgs.knn_over_percent = 0.001f;
  knn_bgs.tooSmalltoDrop = knn_conf[4];
  knn_bgs.dilateRatio =  knn_bgs.IMG_WID  / 320 * 5;
  knn_bgs.init();
  boxes_all.clear();

  int i, cpus = 0;
  int camera_index;
  cpu_set_t mask;
  cpu_set_t get;
  array<thread, 5> threads;

  const char *model_path = "/home/firefly/RKNN/rknn-api/Linux/tmp/mobilenet_ssd.rknn";
  FILE *fp = fopen(model_path, "rb");
  if(fp == NULL) {
    std::printf("fopen %s fail!\n", model_path);
    return -1;
  }
  fseek(fp, 0, SEEK_END);

  int model_len = ftell(fp);
  void *model = malloc(model_len);
  fseek(fp, 0, SEEK_SET);
  if(model_len != fread(model, 1, model_len, fp)) {
    std::printf("fread %s fail!\n", model_path);
    free(model);
    return -1;
  }

  cpus = sysconf(_SC_NPROCESSORS_CONF);
  std::printf("This system has %d processor(s).\n", cpus);

  // string mode = argv[1];
  // if (mode == "c") {
    // camera_index = atoi(argv[2]);
    threads = {thread(cameraRead),
                                thread(knnProcess),
                                thread(displayImage),
                                thread(run_process, 0, model, model_len),
                                thread(run_process, 1, model, model_len)
                                };
  // } 
  // else if (mode == "v") {
  //   threads = {thread(videoRead, argv[2]),
  //                               thread(displayImage),
  //                               thread(run_process, 0, model, model_len),
  //                               thread(run_process, 1, model, model_len)};
  // } 
  // else {
  //   return -1;
  // }

  for (int i = 0; i < 5; i++)
    threads[i].join();

  return 0;
}
