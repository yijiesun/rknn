#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <sys/time.h>
#include <fcntl.h>
#include <string.h>
#include <sys/ioctl.h>
#include <errno.h>
#include <sys/mman.h>
#include <signal.h>
#include <iomanip>
#include <string>
#include <vector>
#include "config.h"
#include "screen/screen.h"
#include "v4l2/v4l2.h" 
#include <unistd.h>  
#include <linux/fb.h>
#include <linux/videodev2.h>
#include "rknn_api.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"


using namespace std;
using namespace cv;

#define NUM_RESULTS         1917
#define NUM_CLASSES         91

#define Y_SCALE  10.0f
#define X_SCALE  10.0f
#define H_SCALE  5.0f
#define W_SCALE  5.0f

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

int IMG_WID;
int IMG_HGT;
cv::Mat frame;
cv::Mat final_img;
V4L2 v4l2_;
SCREEN screen_;
unsigned int screen_pos_x,screen_pos_y;
unsigned int * pfb;
cv::Mat rgb;
bool quit;
pthread_mutex_t  mutex_knn_bgs_frame;
pthread_mutex_t  mutex_show_img;
void *v4l2_thread(void *threadarg);
void my_handler(int s);

float MIN_SCORE = 0.4f;

float NMS_THRESHOLD = 0.45f;


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

int main(int argc, char** argv)
{
    // pthread_mutex_init(&mutex_knn_bgs_frame, NULL);
    // pthread_mutex_init(&mutex_show_img, NULL);
    IMG_WID = 640;
    IMG_HGT = 480;
    screen_.init((char *)"/dev/fb0",640,480);
    Mat frame;
    struct timeval t0_, t1_;
    float total_time = 0.f;
    float mytime = 0;
    quit = false;
    frame.create(IMG_HGT,IMG_WID,CV_8UC3);
    frame = Mat::zeros(IMG_HGT,IMG_WID,CV_8UC3);
    final_img.create(IMG_HGT,IMG_WID,CV_8UC3);
    final_img = Mat::zeros(IMG_HGT,IMG_WID,CV_8UC3);
    struct sigaction sigIntHandler;
 
   sigIntHandler.sa_handler = my_handler;
   sigemptyset(&sigIntHandler.sa_mask);
   sigIntHandler.sa_flags = 0;
 
   sigaction(SIGINT, &sigIntHandler, NULL);

    rgb.create(480,640,CV_8UC3);
    std::string dev_num,imgfld,video_fld;
    get_param_mms_V4L2(dev_num);
    get_captrue_save_data_floder(imgfld,video_fld);
    std::cout<<"open "<<dev_num<<std::endl;

    v4l2_.init(dev_num.c_str(),640,480);
    cout<<"11 "<<endl;
    v4l2_.open_device();
    cout<<"22 "<<endl;
	v4l2_.init_device();
    cout<<"33 "<<endl;
	v4l2_.start_capturing();
    cout<<"save screen_.finfo.smem_len: "<<screen_.finfo.smem_len<<endl;
    pfb = (unsigned int *)malloc(screen_.finfo.smem_len);

	pthread_t threads_v4l2;
	int rc = pthread_create(&threads_v4l2, NULL, v4l2_thread, NULL);

    pthread_join(threads_v4l2,NULL);
    screen_.uninit();
	v4l2_.stop_capturing();
	v4l2_.uninit_device();
	v4l2_.close_device();
    #if 0
    const char *img_path = "/home/firefly/RKNN/rknn-api/Linux/tmp/road.bmp";
    const char *img_path1 = "/home/firefly/RKNN/rknn-api/Linux/tmp/dog.jpg";
    const char *model_path = "/home/firefly/RKNN/rknn-api/Linux/tmp/mobilenet_ssd.rknn";
    const char *label_path = "/home/firefly/RKNN/rknn-api/Linux/tmp/coco_labels_list.txt";
    const char *box_priors_path = "/home/firefly/RKNN/rknn-api/Linux/tmp/box_priors.txt";

    const int img_width = 300;
    const int img_height = 300;
    const int img_channels = 3;
    const int input_index = 0;      // node name "Preprocessor/sub"

    const int output_elems1 = NUM_RESULTS * 4;
    const uint32_t output_size1 = output_elems1 * sizeof(float);
    const int output_index1 = 0;    // node name "concat"

    const int output_elems2 = NUM_RESULTS * NUM_CLASSES;
    const uint32_t output_size2 = output_elems2 * sizeof(float);
    const int output_index2 = 1;    // node name "concat_1"

    // Load image
    cv::Mat img = cv::imread(img_path, 1);
    if(!img.data) {
        printf("cv::imread %s fail!\n", img_path);
        return -1;
    }
        cv::Mat img1 = cv::imread(img_path1, 1);
    if(!img.data) {
        printf("cv::imread %s fail!\n", img_path1);
        return -1;
    }
    if(img.cols != img_width || img.rows != img_height)
        cv::resize(img, img, cv::Size(img_width, img_height), (0, 0), (0, 0), cv::INTER_LINEAR);
    if(img1.cols != img_width || img1.rows != img_height)
        cv::resize(img1, img1, cv::Size(img_width, img_height), (0, 0), (0, 0), cv::INTER_LINEAR);
    // Load model
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

    // Start Inference
    rknn_input inputs[1];
    rknn_output outputs[2];
    rknn_tensor_attr outputs_attr[2];

    int ret = 0;
    rknn_context ctx = 0;


    ret = rknn_init(&ctx, model, model_len, RKNN_FLAG_PRIOR_MEDIUM);
    if(ret < 0) {
        printf("rknn_init fail! ret=%d\n", ret);
        // goto Error;
    }

    outputs_attr[0].index = 0;
    ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(outputs_attr[0]), sizeof(outputs_attr[0]));
    if(ret < 0) {
        printf("rknn_query fail! ret=%d\n", ret);
        // goto Error;
    }

    outputs_attr[1].index = 1;
    ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(outputs_attr[1]), sizeof(outputs_attr[1]));
    if(ret < 0) {
        printf("rknn_query fail! ret=%d\n", ret);
        // goto Error;
    }

while(1)
{
    gettimeofday(&t0_, NULL);
    inputs[0].index = input_index;
    pthread_mutex_lock(&mutex_knn_bgs_frame);
    inputs[0].buf = frame.data;
    pthread_mutex_unlock(&mutex_knn_bgs_frame);

    inputs[0].size = img_width * img_height * img_channels;
    inputs[0].pass_through = false;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    ret = rknn_inputs_set(ctx, 1, inputs);
    if(ret < 0) {
        printf("rknn_input_set fail! ret=%d\n", ret);
        // goto Error;
    }

    ret = rknn_run(ctx, nullptr);
    if(ret < 0) {
        printf("rknn_run fail! ret=%d\n", ret);
        // goto Error;
    }

    outputs[0].want_float = true;
    outputs[0].is_prealloc = false;
    outputs[1].want_float = true;
    outputs[1].is_prealloc = false;
    ret = rknn_outputs_get(ctx, 2, outputs, nullptr);
    if(ret < 0) {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        // goto Error;
    }

        // Process output
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
        printf("validCount: %d\n", validCount);

        if (validCount < 100) {
            /* detect nest box */
            nms(validCount, predictions, output);
            Mat rgba = frame.clone();
            //Mat rgba = imread(img_path, CV_LOAD_IMAGE_UNCHANGED);
            cv::resize(rgba, rgba, cv::Size(1200, 1200), (0, 0), (0, 0), cv::INTER_LINEAR);

            /* box valid detect target */
            for (int i = 0; i < validCount; ++i) {
                if (output[0][i] == -1) {
                    continue;
                }
                int n = output[0][i];
                int topClassScoreIndex = output[1][i];

                int x1 = static_cast<int>(predictions[n * 4 + 1] * rgba.cols);
                int y1 = static_cast<int>(predictions[n * 4 + 0] * rgba.rows);
                int x2 = static_cast<int>(predictions[n * 4 + 3] * rgba.cols);
                int y2 = static_cast<int>(predictions[n * 4 + 2] * rgba.rows);

                string label = labels[topClassScoreIndex];

                std::cout << label << "\t@ (" << x1 << ", " << y1 << ") (" << x2 << ", " << y2 << ")" << "\n";

                rectangle(rgba, Point(x1, y1), Point(x2, y2), colorArray[topClassScoreIndex%10], 3);
                putText(rgba, label, Point(x1, y1 - 12), 1, 2, Scalar(0, 255, 0, 255));
            }
            pthread_mutex_lock(&mutex_show_img);
            final_img = rgba.clone();
            pthread_mutex_unlock(&mutex_show_img);
            // imwrite("out.jpg", rgba);
            // printf("write out.jpg succ!\n");
        } else {
            printf("validCount too much!\n");
        }
    }
    else
    {
        printf("rknn_outputs_get fail! get outputs_size = [%d, %d], but expect [%lu, %lu]!\n",
            outputs[0].size, outputs[1].size, outputs_attr[0].n_elems*sizeof(float), outputs_attr[1].n_elems*sizeof(float));
    }


        gettimeofday(&t1_, NULL);
        mytime = ( float )((t1_.tv_sec * 1000000 + t1_.tv_usec) - (t0_.tv_sec * 1000000 + t0_.tv_usec)) / 1000;
        std::cout <<"thread_done"  << " times  " << mytime << "ms\n";
}




    rknn_outputs_release(ctx, 2, outputs);

Error:
    if(ctx)             rknn_destroy(ctx);
    if(model)           free(model);
    if(fp)              fclose(fp);
#endif
    return 0;
}

void *v4l2_thread(void *threadarg)
{
    
	while (1)
	{
            cout<<"v4l2_thread in"<<endl;
            // pthread_mutex_lock(&mutex_knn_bgs_frame);
            v4l2_.read_frame(frame);
            // pthread_mutex_unlock(&mutex_knn_bgs_frame);
            cout<<"read_frame out"<<endl;
            pthread_mutex_lock(&mutex_show_img);
            v4l2_. mat_to_argb(final_img.data,pfb,640,480,screen_.vinfo.xres_virtual,0,0);
            // pthread_mutex_unlock(&mutex_show_img);
            cout<<"mat_to_argb out"<<endl;
            // memcpy(screen_.pfb,pfb,screen_.finfo.smem_len);
            cout<<"v4l2_thread out"<<endl;
    //    screen_.draw_line(200,200,600,200);

        sleep(0.01); 
        if (quit)
            pthread_exit(NULL);
    }
}


void my_handler(int s)
{
            quit = true;
            cout<<"Caught signal "<<s<<" quit="<<quit<<endl;
}