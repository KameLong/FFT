#pragma OPENCL EXTENSION cl_khr_fp16 : enable
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

typedef int32_t s32;
typedef int16_t s16;
typedef uint32_t u32;
typedef uint16_t u16;
typedef uint8_t u8;

union float32_converter
{
    s32 n;
    float f;
};


// 16-bit float
struct float16
{
    // --- constructors

    float16() {}
    float16(s16 n) { from_float((float)n); }
    float16(s32 n) { from_float((float)n); }
    float16(float n) { from_float(n); }
    float16(double n) { from_float((float)n); }

    // build from a float
    void from_float(float f) { *this = to_float16(f); }

    // --- implicit converters

    operator s32() const { return (s32)to_float(*this); }
    operator float() const { return to_float(*this); }
    operator double() const { return double(to_float(*this)); }

    // --- operators

    float16 operator += (float16 rhs) { from_float(to_float(*this) + to_float(rhs)); return *this; }
    float16 operator -= (float16 rhs) { from_float(to_float(*this) - to_float(rhs)); return *this; }
    float16 operator *= (float16 rhs) { from_float(to_float(*this) * to_float(rhs)); return *this; }
    float16 operator /= (float16 rhs) { from_float(to_float(*this) / to_float(rhs)); return *this; }
    float16 operator + (float16 rhs) const { return float16(*this) += rhs; }
    float16 operator - (float16 rhs) const { return float16(*this) -= rhs; }
    float16 operator * (float16 rhs) const { return float16(*this) *= rhs; }
    float16 operator / (float16 rhs) const { return float16(*this) /= rhs; }
    float16 operator - () const { return float16(-to_float(*this)); }
    bool operator == (float16 rhs) const { return this->v_ == rhs.v_; }
    bool operator != (float16 rhs) const { return !(*this == rhs); }

private:

    // --- entity

    u16 v_;

    // --- conversion between float and float16

    static float16 to_float16(float f)
    {
        if (f<0.00001&&f>-0.00001) {
            float16 f_;
            f_.v_ = 0;
            return f_;

        }

        float32_converter c;
        c.f = f;
        u32 n = c.n;

        // The sign bit is MSB in common.
        u16 sign_bit = (n >> 16) & 0x8000;

        // The exponent of IEEE 754's float 32 is biased +127 , so we change this bias into +15 and limited to 5-bit.
        u16 exponent = (((n >> 23) - 127 + 15) & 0x1f) << 10;

        // The fraction is limited to 10-bit.
        u16 fraction = (n >> (23 - 10)) & 0x3ff;

        float16 f_;
        f_.v_ = sign_bit | exponent | fraction;
        return f_;
    }

    static float to_float(float16 v)
    {
        u32 sign_bit = (v.v_ & 0x8000) << 16;
        u32 exponent = ((((v.v_ >> 10) & 0x1f) - 15 + 127) & 0xff) << 23;
        u32 fraction = (v.v_ & 0x3ff) << (23 - 10);

        float32_converter c;
        c.n = sign_bit | exponent | fraction;
        return c.f;
    }

};

u8* data1;
float16* data2;
float16* data3;
float16* data4;
float* data5;

float16* W;

cl_kernel QEU1;
cl_kernel QEU2;
cl_kernel QEU3;
cl_kernel QEU4;
cl_program  program;
cl_platform_id platforms[PLATFORM_MAX];
cl_device_id devices[DEVICE_MAX];
cl_context ctx;
cl_command_queue q;

int init() {

    float16 a = float(0);



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
    char data[1024];
    size_t size = 1024;
    clGetDeviceInfo(devices[0], CL_DEVICE_EXTENSIONS, 1024, data,0);
    cout << data << endl;

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
    QEU1 = clCreateKernel(program, "K1", &err);
    QEU2 = clCreateKernel(program, "K2", &err);
    QEU3 = clCreateKernel(program, "K3", &err);
    QEU4 = clCreateKernel(program, "K4", &err);
    EC2("clCreateKernel");

    data1 = (u8*)clSVMAlloc(ctx, CL_MEM_READ_WRITE, sizeof(u8) * 1024 * 1024 * 2, 0);
    data2 = (float16*)clSVMAlloc(ctx, CL_MEM_READ_WRITE, sizeof(float16) * 1024 * 1024 * 2, 0);
    data3 = (float16*)clSVMAlloc(ctx, CL_MEM_READ_WRITE, sizeof(float16) * 1024 * 1024 * 2, 0);
    data4 = (float16*)clSVMAlloc(ctx, CL_MEM_READ_WRITE, sizeof(float16) * 1024 * 1024 * 2, 0);
    data5 = (float*)clSVMAlloc(ctx, CL_MEM_READ_WRITE, sizeof(float) * 1024 * 1024 * 2, 0);




    bitRev = (int*)clSVMAlloc(ctx, CL_MEM_READ_WRITE, sizeof(int) * 1024, 0);
    W = (float16*)clSVMAlloc(ctx, CL_MEM_READ_WRITE, sizeof(float16) * 1024*2, 0);
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

    q = clCreateCommandQueue(ctx, devices[0], 0, &err);

}
void close(){



    EC(clReleaseCommandQueue(q), "clReleaseCommandQueue");



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
    cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - mid1Time).count() << endl;


    ifstream in(input, ios::binary);
    unsigned char tmp[1024];
    in.read((char*)tmp, 54);
    for (int i = 0; i < 1024 * 1024; i++) {
        in.read((char*)tmp, 3);
        data1[2 * i] = tmp[0];
    }

    // カーネルの実行
    size_t global[2] = { 128,1024};

    size_t local[2] = { 128,1 };
    cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - mid1Time).count() << endl;

    EC(clEnqueueNDRangeKernel(q, QEU1, 2, nullptr, global, local, 0, nullptr, nullptr), "clEnqueueNDRangeKernel");
    clFinish(q);
    EC(clEnqueueNDRangeKernel(q, QEU2, 2, nullptr, global, local, 0, nullptr, nullptr), "clEnqueueNDRangeKernel");
    clFinish(q);
    cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - mid1Time).count() << endl;
    auto mid2Time = std::chrono::system_clock::now();




    cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - mid1Time).count() << endl;
    EC(clEnqueueNDRangeKernel(q, QEU3, 2, nullptr, global, local, 0, nullptr, nullptr), "clEnqueueNDRangeKernel");
    clFinish(q);
    cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - mid1Time).count() << endl;
    EC(clEnqueueNDRangeKernel(q, QEU4, 2, nullptr, global, local, 0, nullptr, nullptr), "clEnqueueNDRangeKernel");
    clFinish(q);
    cout << "fin" << endl;
    cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - mid1Time).count() << endl;



    ofstream ofs(output, ios::binary);
    ifstream ifs2("ori.bmp", ios::binary);
    char t;
    while (ifs2.read(&t, 1)) {
        ofs.write(&t, 1);
    }
    float max = 0;
    float* outputData = data5;
    outputData[0] = 0;
    for (int i = 0; i < 1024 * 1024; i++) {
        int revI = 1024 * 1024 - i - 1;
        float b = sqrt(float(outputData[2 * revI]) * float(outputData[2 * revI]) + float(outputData[2 * revI + 1]) * float(outputData[2 * revI + 1])) ;
        unsigned char c = b;
        if (b > 255) {
            c = 255;
        }
        ofs.write((char*) & c, 1);
        ofs.write((char*)&c, 1);
        ofs.write((char*)&c, 1);
    }
    ofs.close();
    cout << "ofs Close" << endl;
    cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - mid1Time).count() << endl;


}
int main() {
    init();

    auto mid1Time = std::chrono::system_clock::now();
    task("in.bmp","res.bmp");
    task("in.bmp", "res.bmp");
    auto end1Time = std::chrono::system_clock::now(); 

    cout << std::chrono::duration_cast<std::chrono::milliseconds>(end1Time - mid1Time).count() << endl;
    close();



}

