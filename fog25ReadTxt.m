% fid=fopen('comp4_det_test_bicycle.txt','r'); %�õ��ļ���
% [f,count]=fscanf(fid,'%f_%f_%f_%f %f %f',[49372,6]);
% %���ļ���1�����ݶ���f�С�����f��[12 90]�ľ���
% %����'%f %f'��ʾ��ȡ���ݵ����ƣ����ǰ�ԭʼ�����Ͷ���
% fclose(fid);
filename = 'C:\Users\Administrator.SC-201903012023\Desktop\pcb1_result_test\comp4_det_test_missing_hole.txt';
[data1,data2,data3,data4,data5,data6]=textread(filename,'%s%f%f%f%f%f',130);
data2=num2cell(data2);
data3=num2cell(data3);
data4=num2cell(data4);
data5=num2cell(data5);
data6=num2cell(data6);
missing_train=cat(2,data1,data2,data3,data4,data5,data6);