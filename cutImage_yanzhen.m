fog25ReadTxt;
% x1=pcb1_test{1,2};
% y1=pcb1_test{1,3};
% w=pcb1_test{1,4};
% h=pcb1_test{1,5};
count=0;
R=[];  
for i=1:1:130
z=missing_train{i,2};  
x1=missing_train{i,3};
y1=missing_train{i,4};
w=missing_train{i,5};
h=missing_train{i,6}; 

%²Ã¼ô
a='C:\Users\Administrator.SC-201903012023\Desktop\cut_little_aim\JPEGImages\';
b=data1{i,1};
c='.jpg';
d=[a b c];
e='C:\Users\Administrator.SC-201903012023\Desktop\cut_little_aim\missing_hole\';
if z<0.5
    A=imread(d);
rect=[x1-w/2 y1-h/2 w h];
A1=imcrop(A,rect);
%imshow(A1);

%³ß´ç64*64
B=imresize(A1,[64,64]);
count = count+1;
filename = [e i c];
imwrite(B,filename);

%×ª»Ò¶ÈÍ¼Ïñ
C=rgb2gray(B);
%×ª¾ØÕó
D = reshape(C, 1, prod(size(C))); % prodÊÇÀÛ³Ë
F=im2double(D);
R=[R;F];
%imshow(R)
end
end
count
