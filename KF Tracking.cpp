#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/background_segm.hpp"
#include<vector>
using namespace cv;
using namespace std;

bool cmp(vector<Point> &v1,vector<Point> &v2)//将检测到的轮廓按面积大小进行排序
{
	return contourArea(v1)>contourArea(v2);
}

const float scale=0.5;//缩放系数
const int invscale=(int)1/scale;//导数是在追踪到目标后将举行画到原始图像帧中

int main()
{
	//VideoCapture cap("C:/Users/lenovo/Desktop/【重要】视觉程序/追踪程序/cv_walker.mp4");//打开视频
	VideoCapture cap(0);//相机
	if(!cap.isOpened())//判断是否成功打开
	{
		cout<<"打开视频/相机失败！\n";
		return -1;
	}

	Mat frame,fgmask;//两个变量分别为原图像、提取前景（二值形式：前景为1，背景为0）
	int frameNum=0;//帧数
	BackgroundSubtractorMOG2 mog;//高斯背景模型对象
	Mat element=getStructuringElement(MORPH_RECT,Size(5,5));//形态学滤波结构元
	Rect rect;//轮廓外部矩形边界

	while(1)//开始处理图像
	{
		Begin:
		cap>>frame;//读取帧
		if( frame.empty())//是否读取到帧
		{
			cout<<"未采集到图像！\n";
			break;
		}
		imshow("原始视频",frame);//显示原视频

		Mat halfframe;
		resize(frame,halfframe,Size(),scale,scale);//将原始帧尺寸转换为原来的一半
		imshow("halfframe",halfframe);

		mog(halfframe,fgmask,0.01);//提取前景，0.01为学习率
		imshow("前景",fgmask);//显示前景

		/*对前景进行处理*/
		medianBlur(fgmask,fgmask,5);//中值滤波
		imshow("前景滤波",fgmask);//显示前景
		morphologyEx(fgmask,fgmask,MORPH_DILATE,element);//膨胀处理
		imshow("前景膨胀",fgmask);//显示前景
		morphologyEx(fgmask,fgmask,MORPH_ERODE,element);//腐蚀处理
		imshow("前景腐蚀",fgmask);//显示前景

		/*查找前景的轮廓*/
		vector<vector<Point>>contours;//定义函数参数
		vector<Vec4i>hierarchy;

		findContours(fgmask,contours,hierarchy,RETR_EXTERNAL,CHAIN_APPROX_SIMPLE);//查找轮廓
		if(contours.size()<1)//没有找到轮廓时重新采集图像
		{
			frameNum++;
			goto Begin;
		}
		sort(contours.begin(),contours.end(),cmp);//轮廓按面积从大到小进行排序

		for(size_t i=0;i<contours.size();++i)
		{
			if(contourArea(contours[i])<contourArea(contours[0])/5)//删除小轮廓
				break;
			rect=boundingRect(contours[i]);//矩形外部边界	

			/*从这里开始换成CamShift算法，在CamShift算法中也是上面那句程序以后的换成Kalman滤波器*/
			float a=float(rect.x+rect.width/2);   //这里是用于获取目标矩形框的中心的坐标用于追踪；最后再返回到和原框一样的坐标
			float b=float(rect.y+rect.height/2);

			const int stateNum=4;//状态个数（矩形中心坐标x,y和坐标的变化值dx,dy）
			const int measureNum=2;//要测量的状态的个数
			KalmanFilter KF(stateNum, measureNum, 0);
			Mat state(stateNum, 1, CV_32F); //四个状态：x,y,detax,detay(x,y是矩形中心坐标)
			Mat processNoise(stateNum, 1, CV_32F);
			Mat measurement = Mat::zeros(measureNum, 1, CV_32F);//测量x,y
				
			state.at<float>(0)=a;//初始化滤波器的状态
			state.at<float>(1)=b;
				
			KF.transitionMatrix = *(Mat_<float>(4, 4) << 1,0,1,0,0,1,0,1,0,0,1,0,0, 0, 0, 1);//转移矩阵

			setIdentity(KF.measurementMatrix);
			setIdentity(KF.processNoiseCov, Scalar::all(1e-5));
			setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));
			setIdentity(KF.errorCovPost, Scalar::all(1));

			randn(KF.statePost, Scalar::all(0), Scalar::all(0.1));//这句可以不要

			Mat prediction = KF.predict();
			prediction.at<float>(0)=a;//根据这两个值预测;---这里一定要，不然不会更新，结果完全错误
			prediction.at<float>(1)=b;
			float precs1=prediction.at<float>(0);//预测后的状态
			float precs2=prediction.at<float>(1);

			//randn( measurement, Scalar::all(0), Scalar::all(KF.measurementNoiseCov.at<float>(0)));//这个也可以不要

			//// 获得测量矩阵；不用到measurement的时，这两句可以不要，因为只用到预测后的状态
			/*measurement += KF.measurementMatrix*state;
			float meass = measurement.at<float>(0);*/

			//由预测值计算新矩形
			int x=int(precs1)-rect.width/2;
			int y=int(precs2)-rect.height/2;
			Rect preRect(x*invscale,y*invscale,rect.width*invscale,rect.height*invscale);
			rectangle(frame,preRect,Scalar(0,0,255),3);

			KF.correct(measurement);

			randn( processNoise, Scalar(0), Scalar::all(sqrt(KF.processNoiseCov.at<float>(0, 0))));
			state = KF.transitionMatrix*state + processNoise;

			imshow("追踪结果",frame);
			/*到这里都可以换成CamShift算法，在CamShift算法中到这里换成Kalman滤波器**/
		}//end for

		char c=(char)waitKey(10);
		if(c==(char)27||c=='q'||c=='Q')  
			break; 
	}//end while();
	return 0;
}
