#这个代码是飞飞帮忙写的，把VGG跑出来的结果覆盖回YOLO的result文件
f1 = open("test_missinghole.txt","r")
lines1 = f1.readlines()
f2= open("comp4_det_test_Missing_hole.txt","r")
lines2 = f2.readlines()


content = []
for line2 in lines2:
    flag =1
    for line1 in lines1:
        # print(line1)
        line1_list = line1.strip().split('\t')
        line2_list = line2.strip().split(' ')
        if (line1_list[0] == line2_list[0]) and (float(line2_list[1]) <= float(line1_list[2])):
                # line2_list[2] = line1_list[2]
                txt = line2_list[0] + ' ' + line1_list[2] + line2_list[2]  + ' ' + line2_list[3] + ' ' + line2_list[4]+ ' ' + line2_list[5] + '\n'
                content.append(txt)
                flag = 0
    if flag:
        content.append(line2)
# print(content)
print(len(content))

for txt in content:
    with open(".//result/comp4_det_test_Missing_hole.txt", "a") as f:
        f.write(txt)

# f1.close()

# t = content.replace("hello","hi")
# with open("hello.txt","w") as f2:
#     f2.write(t)
#
#
#
# f = open(filename,'r')
# line = f.readline()
# while line:
#     index = line.find(_str)
#     if index!=-1:
#         print(index)
#         print(line[index:index+lenth])
#         line=line.replace(_str,_str_new)
#         print(line)
#     all_line.append(line)
#     line = f.readline()
# #print(all_line)
# f.close()
# f1 =open(filename,'w')
# for line in all_line:
#     f1.write(line)
#
# f2 = open("./image/abc.txt","r")
# lines = f2.readlines()
# for line3 in lines:
#     print line3