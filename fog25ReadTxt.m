% fid=fopen('comp4_det_test_bicycle.txt','r'); %得到文件号
% [f,count]=fscanf(fid,'%f_%f_%f_%f %f %f',[49372,6]);
% %把文件号1的数据读到f中。其中f是[12 90]的矩阵
% %这里'%f %f'表示读取数据的形势，他是按原始数据型读出
% fclose(fid);
filename = 'C:\Users\Administrator.SC-201903012023\Desktop\pcb1_result_test\comp4_det_test_missing_hole.txt';
[data1,data2,data3,data4,data5,data6]=textread(filename,'%s%f%f%f%f%f',130);
data2=num2cell(data2);
data3=num2cell(data3);
data4=num2cell(data4);
data5=num2cell(data5);
data6=num2cell(data6);
missing_train=cat(2,data1,data2,data3,data4,data5,data6);