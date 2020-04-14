# YOLO

安装  torch-1.2.0-cp36-cp36m-win_amd64.whl
torchvision-0.4.0-cp36-cp36m-win_amd64.whl

CUDA 10.1

## 该文件夹存放使用pytorch实现的代码版本
**model.py**： 是模型文件  
**train.py**： 是调用模型训练的文件    
**predict.py**： 是调用模型进行预测的文件  
**class_indices.json**： 是训练数据集对应的标签文件   


* （1）在data_set文件夹下创建新文件夹"flower_data"
* （2）打开flower_link.txt文档，复制网址到浏览器会自动进行下载花分类数据集
* （3）解压数据集到flower_data文件夹下
* （4）执行"split_data.py"脚本自动将数据集划分成训练集train和验证集val    
  （不要重复使用该脚本，否则训练集和验证集会混在一起，flower_data文件夹结构如下）   
  |—— flower_data   
  |———— flower_photos（解压的数据集文件夹，3670个样本）  
  |———— train（生成的训练集，3306个样本）  
  |———— val（生成的验证集，364个样本） 
     
根据YOLO的result文件，将置信度<0.5的目标裁剪出来，裁剪算法用yanzhen的或者飞飞的裁剪那2个其中一个，MATLAB文件。裁剪出来的图片再送进VGG检测，检测生成的文件再使用飞飞的search文件，和原来的RESULT文件进行合并
