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

#define __AVE_TIC__(tag) static int ____##tag##_total_time=0; \
        static int ____##tag##_total_conut=0;\
        timeval ____##tag##_start_time, ____##tag##_end_time;\
        gettimeofday(&____##tag##_start_time, 0);

#define __AVE_TOC__(tag) gettimeofday(&____##tag##_end_time, 0); \
        ____##tag##_total_conut++; \
        ____##tag##_total_time+=((int)____##tag##_end_time.tv_sec-(int)____##tag##_start_time.tv_sec)*1000000+((int)____##tag##_end_time.tv_usec-(int)____##tag##_start_time.tv_usec); \
        fprintf(stderr,  #tag ": %d us\n", ____##tag##_total_time/____##tag##_total_conut);

#define __TIC__(tag) timeval ____##tag##_start_time, ____##tag##_end_time;\
        gettimeofday(&____##tag##_start_time, 0);

#define __TOC__(tag) gettimeofday(&____##tag##_end_time, 0); \
        int ____##tag##_total_time=((int)____##tag##_end_time.tv_sec-(int)____##tag##_start_time.tv_sec)*1000000+((int)____##tag##_end_time.tv_usec-(int)____##tag##_start_time.tv_usec); \
        fprintf(stderr,  #tag ": %d us\n", ____##tag##_total_time);



V4L2 v4l2_;
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

mutex mtxQueueInput;               // mutex of input queue
mutex mtxQueueShow;                // mutex of display queue
queue<pair<int, Mat>> queueInput;  // input queue
priority_queue<imagePair, vector<imagePair>, paircomp>
        queueShow;  // display queue

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

void cameraRead() 
{
  int i = 0;
  int initialization_finished = 1;
  cpu_set_t mask;
  int cpuid = 2;

  CPU_ZERO(&mask);
  CPU_SET(cpuid, &mask);

  if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0)
    cerr << "set thread affinity failed" << endl;

  printf("Bind CameraCapture process to CPU %d\n", cpuid); 

        // cv::VideoCapture camera(index);
        // if (!camera.isOpened()) {
        //         cerr << "Open camera error!" << endl;
        //         exit(-1);
        // }

  // camera.set(3, 640);
  // camera.set(4, 480);
  // camera.set(5, 30.0);

  while (true) {
    initialization_finished = 1;
  
    for (int i = 0; i < sizeof(multi_npu_process_initialized) / sizeof(int); i++) {
      //cout << i << " " << multi_npu_process_initialized << endl;
      if (multi_npu_process_initialized[i] == 0) {
        initialization_finished = 0;
        //break;
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
                mtxQueueInput.lock();
                if(queueInput.size() <= QUEUE_SIZE)
                    queueInput.push(make_pair(idxInputImage++, img));
                // if (queueInput.size() > QUEUE_SIZE) {
                //         mtxQueueInput.unlock();
                //         cout << "[Warning]input queue size is " << queueInput.size() << endl;
                //         sleep(1);
                // } else {
                        mtxQueueInput.unlock();
                // }
  }
}

// void videoRead(const char *video_name) 
// {
//         int i = 0;
//   int initialization_finished = 1;
//   int cpuid = 2;
//   cpu_set_t mask;

//   CPU_ZERO(&mask);
//   CPU_SET(cpuid, &mask);

//   if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0)
//     cerr << "set thread affinity failed" << endl;

//   printf("Bind VideoCapture process to CPU %d\n", cpuid); 

//         cv::VideoCapture video;

//         if (!video.open(video_name)) {
//                 cout << "Fail to open " << video_name << endl;
//                 return;
//         }

//   int frame_cnt = video.get(CV_CAP_PROP_FRAME_COUNT);

//   while (true) {
//     initialization_finished = 1;

//     for (int i = 0; i < sizeof(multi_npu_process_initialized) / sizeof(int); i++) {
//       //cout << i << " " << multi_npu_process_initialized << endl;
//       if (multi_npu_process_initialized[i] == 0) {
//         initialization_finished = 0;
//         //break;
//       }
//     }

//     if (initialization_finished)
//       break;

//     sleep(1);
//   }

//   start_time = chrono::system_clock::now();
//         while (true) 
//   {  
//           usleep(1000);
//     Mat img;

//                 if (queueInput.size() < 30) {
//       if (!video.read(img)) {
//         cout << "read video stream failed!" << endl;
//         return;
//       }
//       mtxQueueInput.lock();
//       queueInput.push(make_pair(idxInputImage++, img));
//                         mtxQueueInput.unlock();

//       if (idxInputImage >= frame_cnt)
//         break;
//                 } else {
//                         usleep(10);
//                 }
//   }
// }

void displayImage() {
  // Mat img;
  cpu_set_t mask;
  int cpuid = 3;

  CPU_ZERO(&mask);
  CPU_SET(cpuid, &mask);

  if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0)
    cerr << "set thread affinity failed" << endl;

  printf("Bind Display process to CPU %d\n", cpuid); 

    while (true) {
        mtxQueueShow.lock();
        if (queueShow.empty()) {
            // mtxQueueShow.unlock();
            usleep(10);
        } else if (idxShowImage == queueShow.top().first) {
            auto show_time = chrono::system_clock::now();
            stringstream buffer;
            Mat img = queueShow.top().second;
            auto dura = (duration_cast<microseconds>(show_time - start_time)).count();
            buffer << fixed << setprecision(2)
                   << (float)queueShow.top().first / (dura / 1000000.f);
            string a = buffer.str() + "FPS";
            cv::putText(img, a, cv::Point(15, 15), 1, 1, cv::Scalar{0, 0, 255},2);

            if(!img.empty())
            {
              v4l2_. mat_to_argb(img.data,pfb,640,480,screen_.vinfo.xres_virtual,0,0);
              memcpy(screen_.pfb,pfb,screen_.finfo.smem_len);
            }


            // cv::imshow("RK3399Pro", img);  // display image
            idxShowImage++;
            queueShow.pop();
            
            // if (waitKey(1) == 'q') {
            //     bReading = false;
            //     exit(0);
            // }
        } else {
            // mtxQueueShow.unlock();
        }
        mtxQueueShow.unlock();
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

  cpu_set_t mask;
  int cpuid = 0;

  if (thread_id == 0)
    cpuid = 4;
  else if (thread_id == 1)
    cpuid = 5;
  else
    cpuid = 0;

  CPU_ZERO(&mask);
  CPU_SET(cpuid, &mask);

  if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0)
    cerr << "set thread affinity failed" << endl;

  printf("Bind NPU process(%d) to CPU %d\n", thread_id, cpuid); 

  // Start Inference
  rknn_input inputs[1];
  rknn_output outputs[2];
  rknn_tensor_attr outputs_attr[2];

  int ret = 0;
  rknn_context ctx = 0;

  ret = rknn_init(&ctx, pmodel, model_len, RKNN_FLAG_PRIOR_MEDIUM);
  if(ret < 0) {
    printf("rknn_init fail! ret=%d\n", ret);
    return;
  }

  outputs_attr[0].index = 0;
  ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(outputs_attr[0]), sizeof(outputs_attr[0]));
  if(ret < 0) {
    printf("rknn_query fail! ret=%d\n", ret);
    return;
  }

  outputs_attr[1].index = 1;
  ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(outputs_attr[1]), sizeof(outputs_attr[1]));
  if(ret < 0) {
    printf("rknn_query fail! ret=%d\n", ret);
    return;
  }

  if (thread_id > sizeof(multi_npu_process_initialized) / sizeof(int) - 1)
    return;

  multi_npu_process_initialized[thread_id] = 1;
  printf("The initialization of NPU Process %d has been completed.\n", thread_id);
  __TOC__(NPU_INIT);
#endif
  while (true) {
    __TIC__(thread_id);
    // cout<<"process 0"<<endl;
    pair<int, Mat> pairIndexImage;
    mtxQueueInput.lock();
    // cout<<"mtxQueueInput.lock()"<<endl;
    if (queueInput.empty()) {
      mtxQueueInput.unlock();
      usleep(20000);
      if (bReading)
        continue;
      else
        break;
      } else {
        // Get an image from input queue
        pairIndexImage = queueInput.front();
        queueInput.pop();
        // mtxQueueInput.unlock();
      }
      mtxQueueInput.unlock();
      #if 1
      cv::resize(pairIndexImage.second, resimg, cv::Size(img_width, img_height), (0, 0), (0, 0), cv::INTER_LINEAR);

      inputs[0].index = input_index;
      inputs[0].buf = resimg.data;
      inputs[0].size = img_width * img_height * img_channels;
      inputs[0].pass_through = false;
      inputs[0].type = RKNN_TENSOR_UINT8;
      inputs[0].fmt = RKNN_TENSOR_NHWC;
      ret = rknn_inputs_set(ctx, 1, inputs);
      if(ret < 0) {
          printf("rknn_input_set fail! ret=%d\n", ret);
          return;
      }

      ret = rknn_run(ctx, nullptr);
      if(ret < 0) {
          printf("rknn_run fail! ret=%d\n", ret);
          return;
      }

      outputs[0].want_float = true;
      outputs[0].is_prealloc = false;
      outputs[1].want_float = true;
      outputs[1].is_prealloc = false;
      ret = rknn_outputs_get(ctx, 2, outputs, nullptr);
      if(ret < 0) {
          printf("rknn_outputs_get fail! ret=%d\n", ret);
          return;
      }
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
          //printf("validCount: %d\n", validCount);

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

                int x1 = static_cast<int>(predictions[n * 4 + 1] * pairIndexImage.second.cols);
                int y1 = static_cast<int>(predictions[n * 4 + 0] * pairIndexImage.second.rows);
                int x2 = static_cast<int>(predictions[n * 4 + 3] * pairIndexImage.second.cols);
                int y2 = static_cast<int>(predictions[n * 4 + 2] * pairIndexImage.second.rows);

                string label = labels[topClassScoreIndex];

                //std::cout << label << "\t@ (" << x1 << ", " << y1 << ") (" << x2 << ", " << y2 << ")" << "\n";

                rectangle(pairIndexImage.second, Point(x1, y1), Point(x2, y2), colorArray[topClassScoreIndex%10], 3);
                putText(pairIndexImage.second, label, Point(x1, y1 - 12), 1, 2, Scalar(0, 255, 0, 255));
            }
          } else {
              printf("validCount too much!\n");
          }
      }
      else
      {
          printf("rknn_outputs_get fail! get outputs_size = [%d, %d], but expect [%lu, %lu]!\n",
              outputs[0].size, outputs[1].size, outputs_attr[0].n_elems*sizeof(float), outputs_attr[1].n_elems*sizeof(float));
      }
      rknn_outputs_release(ctx, 2, outputs);
#endif
      mtxQueueShow.lock();
      // cout<<"mtxQueueShow input "<<queueShow.size()<<endl;
      // Put the processed iamge to show queue
      queueShow.push(pairIndexImage);
      mtxQueueShow.unlock();
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

  int i, cpus = 0;
  int camera_index;
  cpu_set_t mask;
  cpu_set_t get;
  array<thread, 4> threads;

  const char *model_path = "/home/firefly/RKNN/rknn-api/Linux/tmp/mobilenet_ssd.rknn";
  FILE *fp = fopen(model_path, "rb");
  if(fp == NULL) {
    printf("fopen %s fail!\n", model_path);
    return -1;
  }
  fseek(fp, 0, SEEK_END);

  int model_len = ftell(fp);
  void *model = malloc(model_len);
  fseek(fp, 0, SEEK_SET);
  if(model_len != fread(model, 1, model_len, fp)) {
    printf("fread %s fail!\n", model_path);
    free(model);
    return -1;
  }

  cpus = sysconf(_SC_NPROCESSORS_CONF);
  printf("This system has %d processor(s).\n", cpus);

  // string mode = argv[1];
  // if (mode == "c") {
    // camera_index = atoi(argv[2]);
    threads = {thread(cameraRead),
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

  for (int i = 0; i < 3; i++)
    threads[i].join();

  return 0;
}
