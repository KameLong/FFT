#include <iostream>
#include<complex>
#include<math.h>
#include <CL/opencl.h>
#include<chrono>
#include <fstream>
#include<string>
using namespace std;

#define PLATFORM_MAX 4
#define DEVICE_MAX 4
#define M_PI 3.14159265

cl_int err = CL_SUCCESS;
int* bitRev;


size_t Q=0;
int O=0;
void EC(cl_int result, const char *title)
{
    if (result != CL_SUCCESS) {
        std::cout << "Error: " << title << "(" << result << ")\n";
    }
}
void EC2(const char *title)
{
    if (err != CL_SUCCESS) {
        std::cout << "Error: " << title << "(" << err << ")\n";
    }
//    err = CL_SUCCESS;
}



float* data1;
float* data2;
float* data3;
float* data4;
float* data5;

float* W;

cl_kernel QEU1;
cl_kernel QEU2;
cl_kernel QEU3;
cl_kernel QEU4;
cl_program  program;
cl_platform_id platforms[PLATFORM_MAX];
cl_device_id devices[DEVICE_MAX];
cl_context ctx;

int init() {





    // プラットフォーム一覧を取得
    cl_uint platformCount;
    EC(clGetPlatformIDs(PLATFORM_MAX, platforms, &platformCount), "clGetPlatformIDs");
    if (platformCount == 0) {
        std::cerr << "No platform.\n";
        return EXIT_FAILURE;
    }

    // 見つかったプラットフォームの情報を印字
    for (int i = 0; i < platformCount; i++) {
        char vendor[100] = { 0 };
        char version[100] = { 0 };
        EC(clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(vendor), vendor, nullptr), "clGetPlatformInfo");
        EC(clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, sizeof(version), version, nullptr), "clGetPlatformInfo");
        std::cout << "Platform id: " << platforms[i] << ", Vendor: " << vendor << ", Version: " << version << "\n";
    }

    // デバイス一覧を取得
    cl_uint deviceCount;
    EC(clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, DEVICE_MAX, devices, &deviceCount), "clGetDeviceIDs");
    if (deviceCount == 0) {
        std::cerr << "No device.\n";
        return EXIT_FAILURE;
    }

    // 見つかったデバイスの情報を印字
    std::cout << deviceCount << " device(s) found.\n";
    for (int i = 0; i < deviceCount; i++) {
        char name[100] = { 0 };
        size_t len;
        EC(clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(name), name, &len), "clGetDeviceInfo");
        std::cout << "Device id: " << i << ", Name: " << name << "\n";
    }

    // コンテキストの作成
    ctx = clCreateContext(nullptr, 1, devices, nullptr, nullptr, &err);
    EC2("clCreateContext");
    // clプログラムの読み込み
    ifstream ifs("kernel.cl");
    string str;
    string source = "";
    while (getline(ifs, str)) {
        source += str + "\n";
    }
    const char* source_str = source.c_str();
    size_t source_size = strlen(source_str);
    const char* source_list[] = { source_str };

    program = clCreateProgramWithSource(ctx, 1, source_list, &source_size, &err);
    EC2("clCreateProgramWithBinary");

    // プログラムのビルド
    EC(clBuildProgram(program, 1, devices, nullptr, nullptr, nullptr), "clBuildProgram");

    // カーネルの作成
    QEU1 = clCreateKernel(program, "QEU", &err);
    QEU2 = clCreateKernel(program, "QEU", &err);
    QEU3 = clCreateKernel(program, "QEU", &err);
    QEU4 = clCreateKernel(program, "QEU", &err);
    EC2("clCreateKernel");

    data1 = (float*)clSVMAlloc(ctx, CL_MEM_READ_WRITE, sizeof(float) * 1024 * 1024 * 2, 0);
    data2 = (float*)clSVMAlloc(ctx, CL_MEM_READ_WRITE, sizeof(float) * 1024 * 1024 * 2, 0);
    data3 = (float*)clSVMAlloc(ctx, CL_MEM_READ_WRITE, sizeof(float) * 1024 * 1024 * 2, 0);
    data4 = (float*)clSVMAlloc(ctx, CL_MEM_READ_WRITE, sizeof(float) * 1024 * 1024 * 2, 0);
    data5 = (float*)clSVMAlloc(ctx, CL_MEM_READ_WRITE, sizeof(float) * 1024 * 1024 * 2, 0);




    bitRev = (int*)clSVMAlloc(ctx, CL_MEM_READ_WRITE, sizeof(int) * 1024, 0);
    W = (float*)clSVMAlloc(ctx, CL_MEM_READ_WRITE, sizeof(float) * 1024*2, 0);
    //bitリバースを作ります
    for (int i = 0; i < 1024; i++) {
        int x = i;
        x = ((x & 0x00ff00ff) << 8) | ((x >> 8) & 0x00ff00ff);
        x = ((x & 0x0f0f0f0f) << 4) | ((x >> 4) & 0x0f0f0f0f);
        x = ((x & 0x33333333) << 2) | ((x >> 2) & 0x33333333);
        x = ((x & 0x55555555) << 1) | ((x >> 1) & 0x55555555);
        x = x >> 6;
        bitRev[i] = x;
    }
    //複素平面状の単位円の座標を用意します
    for (int i = 0; i < 10; i++) {
        int A = pow(2, i);
        for (int j = 0; j < pow(2, i); j++) {
            W[(A + j) * 2] = cos(M_PI * j / A);
            W[(A + j) * 2 + 1] = sin(M_PI * j / A);
        }

    }

    EC(clSetKernelArgSVMPointer(QEU1, 0, data1), "clSetKernelArg");
    EC(clSetKernelArgSVMPointer(QEU1, 1, data2), "clSetKernelArg");
    EC(clSetKernelArgSVMPointer(QEU1, 2, W), "clSetKernelArg");
    EC(clSetKernelArgSVMPointer(QEU1, 3, bitRev), "clSetKernelArg");
    EC(clSetKernelArgSVMPointer(QEU2, 0, data2), "clSetKernelArg");
    EC(clSetKernelArgSVMPointer(QEU2, 1, data3), "clSetKernelArg");
    EC(clSetKernelArgSVMPointer(QEU2, 2, W), "clSetKernelArg");
    EC(clSetKernelArgSVMPointer(QEU2, 3, bitRev), "clSetKernelArg");

    EC(clSetKernelArgSVMPointer(QEU3, 0, data3), "clSetKernelArg");
    EC(clSetKernelArgSVMPointer(QEU3, 1, data4), "clSetKernelArg");
    EC(clSetKernelArgSVMPointer(QEU3, 2, W), "clSetKernelArg");
    EC(clSetKernelArgSVMPointer(QEU3, 3, bitRev), "clSetKernelArg");
    EC(clSetKernelArgSVMPointer(QEU4, 0, data4), "clSetKernelArg");
    EC(clSetKernelArgSVMPointer(QEU4, 1, data5), "clSetKernelArg");
    EC(clSetKernelArgSVMPointer(QEU4, 2, W), "clSetKernelArg");
    EC(clSetKernelArgSVMPointer(QEU4, 3, bitRev), "clSetKernelArg");

}
void close(){






    // カーネルの解放
    EC(clReleaseKernel(QEU1), "clReleaseKernel");
    EC(clReleaseKernel(QEU2), "clReleaseKernel");
    EC(clReleaseKernel(QEU3), "clReleaseKernel");
    EC(clReleaseKernel(QEU4), "clReleaseKernel");

    // プログラムの解放
    EC(clReleaseProgram(program), "clReleaseProgram");

    // コンテキストの解放
    EC(clReleaseContext(ctx), "clReleaseContext");

}
void task(string input,string output) {
    auto mid1Time = std::chrono::system_clock::now();
    auto end1Time = std::chrono::system_clock::now();
    cout << std::chrono::duration_cast<std::chrono::milliseconds>(end1Time - mid1Time).count() << endl;

    cl_command_queue q = clCreateCommandQueue(ctx, devices[0], 0, &err);

    ifstream in(input, ios::binary);
    unsigned char tmp[1024];
    in.read((char*)tmp, 54);
    for (int i = 0; i < 1024 * 1024; i++) {
        in.read((char*)tmp, 3);
        data1[2 * i] = tmp[1] / 255.0;
    }

    // カーネルの実行
    size_t global[2] = { 128,1024};

    size_t local[2] = { 128,1 };
    EC(clEnqueueNDRangeKernel(q, QEU1, 2, nullptr, global, local, 0, nullptr, nullptr), "clEnqueueNDRangeKernel");
    clFinish(q);
    EC(clEnqueueNDRangeKernel(q, QEU2, 2, nullptr, global, local, 0, nullptr, nullptr), "clEnqueueNDRangeKernel");
    clFinish(q);
    auto mid2Time = std::chrono::system_clock::now();



    const int SMOOZE = 1000;
    for (int i = 0; i < 1024; i++) {
        for (int j = 0; j < 1024; j++) {
            int y = (i + 512) % 1024 - 512;
            int x = (j + 512) % 1024 - 512;
            data3[2 * 1024 * i + 2 * j] *= exp(-((x * x + y * y) / SMOOZE));
            data3[2 * 1024 * i + 2 * j + 1] *= exp(-((x * x + y * y) / SMOOZE));
        }
    }
    EC(clEnqueueNDRangeKernel(q, QEU3, 2, nullptr, global, local, 0, nullptr, nullptr), "clEnqueueNDRangeKernel");
    clFinish(q);
    EC(clEnqueueNDRangeKernel(q, QEU4, 2, nullptr, global, local, 0, nullptr, nullptr), "clEnqueueNDRangeKernel");
    clFinish(q);



    ofstream ofs(output, ios::binary);
    ifstream ifs2("ori.bmp", ios::binary);
    char t;
    while (ifs2.read(&t, 1)) {
        ofs.write(&t, 1);
    }
    float max = 0;
    float* outputData = data5;
    for (int i = 0; i < 1024 * 1024; i++) {
        float b = sqrt(outputData[2 * i] * outputData[2 * i] + outputData[2 * i + 1] * outputData[2 * i + 1]);
        if (b > max) {
            max = b;
        }
    }
    cout << max << endl;
    for (int i = 0; i < 1024 * 1024; i++) {
        int revI = 1024 * 1024 - i - 1;
        float b = sqrt(outputData[2 * revI] * outputData[2 * revI] + outputData[2 * revI + 1] * outputData[2 * revI + 1]) * 255 / max;
        b = abs(b - data1[i * 2] * 255);
        unsigned char c = b;
        if (b > 255) {
            c = 255;
        }
        ofs.write((char*) & c, 1);
        ofs.write((char*)&c, 1);
        ofs.write((char*)&c, 1);
    }
    ofs.close();


    for (int i = 0; i < 1024; i++) {
        for (int j = 0; j < 1024; j++) {
            int y = (i + 512) % 1024 - 512;
            int x = (j + 512) % 1024 - 512;
            if (x * x + y * y > 100) {
                data3[2 * 1024 * i + 2 * j] = 0;
                data3[2 * 1024 * i + 2 * j + 1] = 0;
            }
        }
    }
    EC(clReleaseCommandQueue(q), "clReleaseCommandQueue");

}
int main() {
    init();

    auto mid1Time = std::chrono::system_clock::now();
    task("in.bmp","res.bmp");
    auto end1Time = std::chrono::system_clock::now();
    cout << std::chrono::duration_cast<std::chrono::milliseconds>(end1Time - mid1Time).count() << endl;
    close();



}

