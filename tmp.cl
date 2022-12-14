__kernel
void FFT1024(__global float* input,__global const  float* W,__global  const int* rev){
    const int x=get_global_id(0);
    const int y=get_global_id(1);//　0-1023が入る
    const int x2=get_local_id(0);//  0-127が入る
    __local float data[1024*2];
    barrier(CLK_LOCAL_MEM_FENCE);
    //データをSVMにキャッシュ
    for(int i=0;i<8;i++){
        int t=128*i+x2;
        int t2=rev[t];
        data[t*2]=input[t2*2+1024*2*y];
        data[t*2+1]=input[t2*2+1+1024*2*y];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int pow2[11]={1,2,4,8,16,32,64,128,256,512,1024};
    int pow2_2[11]={0,1,3,7,15,31,63,127,255,511,1023};
    for(int i=0;i<10;i++){
        int startX=8*x2-((4*x2)&pow2_2[i]);
        //バタフライ演算をします
        //x2*8からx2*8+7までの要素の計算を行います
        //i==0の時は2項目
        for(int j=0;j<4;j++){
            int p1=startX+j*2-(j&pow2_2[i]);
            int p2=p1+pow2[i];
            int w=pow2[i]+(p1&pow2_2[i]);
            float tR=data[p2*2]*W[w*2]-data[p2*2+1]*W[w*2+1];
            float tI=data[p2*2]*W[w*2+1]+data[p2*2+1]*W[w*2];
            float tmp1R=data[p1*2+0]+tR;
            float tmp2R=data[p1*2+0]-tR;
            float tmp1I=data[p1*2+1]+tI;
            float tmp2I=data[p1*2+1]-tI;
            data[p1*2]=tmp1R;
            data[p1*2+1]=tmp1I;
            data[p2*2]=tmp2R;
            data[p2*2+1]=tmp2I;

        }
        barrier(CLK_LOCAL_MEM_FENCE);


    }
    barrier(CLK_LOCAL_MEM_FENCE);
    //データをSVMにキャッシュ
    for(int i=0;i<16;i++){
        input[128*i+x2+1024*2*y]=data[128*i+x2]/1024;

    }


}
