clc;clear;close all;
%%
%定义参数
number = input("请输入需要生成的样本数据：\n");
f_up = input("请输入上行频段中心频率（例如3GHz输入3e9）：\n");
f_down = input("请输入下行频段中心频率（例如28GHz输入28e9）：\n");
type = input("请输入信道类型(可选项为CDL-A,B,C,D,E): \n" , 's')
Vc = [4.8, 24, 40, 60];    %用户移动速度km/h
%type = 'CDL-A'

NRB = 6;    %资源块数
carrnum = NRB*12;      %子载波数 = 资源块数*12
OFDMNum = 14;       %OFDM符号数14
%分别定义基站和用户端的天线阵列,求出基站和用户的天线数
BSAtNum = 64;
UEAtNum = 4;

switch type
    case 'CDL-A'
        DelayS = 129e-9;
    case 'CDL-B'
        DelayS = 634e-9;
    case 'CDL-C'
        DelayS = 634e-9;
    otherwise
        DelayS = 65e-9;
end


%将number个样本分batch处理
%batch = 1000;
batch = input("请输入每次生成的batch数，建议根据内存动态调整: \n")
epoch = number/batch;

%合并后数据保存路径
filename2 = "../data/"+type+"_"+num2str(carrnum)+"_"+num2str(BSAtNum)+"_"+num2str(UEAtNum)+"_"+num2str(number)+".mat" %保存路径

%%
raseed_total = randperm(number);
index_total = randi(length(Vc),1,number);

%生成速度矩阵
for p = 1:length(index_total)
    Vmatrix(p) = Vc(index_total(p));
end
%分批次生成数据
for j = 1:epoch
    %生成随机种子序列
    raseed_arr = raseed_total((j-1)*batch+1:j*batch);
    index_arr = index_total((j-1)*batch+1:j*batch);
    mimo_uplink = zeros(batch,carrnum,OFDMNum,BSAtNum,UEAtNum);%上行信道，用户——基站
    mimo_downlink = zeros(batch,carrnum,OFDMNum,UEAtNum,BSAtNum);%下行信道，基站——用户
    %每个epoch进行保存
    filename = "../data/"+type+"_"+num2str(carrnum)+"_"+num2str(BSAtNum)+"_"+num2str(UEAtNum)+"_"...
    +num2str(number)+"_ep"+num2str(j)+"of"+num2str(epoch)+".mat" %保存路径

    for i = 1:batch
        raseed = raseed_arr(i);
        index = index_arr(i);
        mimo_uplink(i,:,:,:,:) = PUSCH(type,NRB,f_up,Vc(index),UEAtNum,BSAtNum,DelayS,raseed); %上行频段2.0GHz
        mimo_downlink(i,:,:,:,:) = PDSCH(type,NRB,f_down,Vc(index),BSAtNum,UEAtNum,DelayS,raseed); %下行频段2.1GHz
        fprintf("epoch:%d/%d,第%d组数据已生成\n",j,epoch,i)
        if(mod(i,batch/2) ==0)
            %压缩保存数据
            save(filename,'mimo_uplink','mimo_downlink','Vmatrix','-v7.3')
            %不压缩保存数据
            %save(filename,'mimo_uplink','mimo_downlink','Vmatrix','-v7.3','-nocompression')
            fprintf("保存提示：前%d组数据已保存\n",i)
        end
    end
    
    clear mimo_uplink mimo_downlink;
   
        
end
%%
%合并数据
mimo_uplink = [];
mimo_downlink = [];
%filename2 = "../data/mimo_"+num2str(carrnum)+"_"+num2str(BSAtNum)+"_"+num2str(UEAtNum)+"_"+num2str(number)+".mat" %保存路径
save(filename2,'mimo_uplink','mimo_downlink','Vmatrix','-v7.3')
m = matfile(filename2,'Writable',true)

i= 1;
 filename1 = "../data/"+type+"_"+num2str(carrnum)+"_"+num2str(BSAtNum)+"_"+num2str(UEAtNum)+"_"...
    +num2str(number)+"_ep"+num2str(i)+"of"+num2str(epoch)+".mat" %保存路径
file = matfile(filename1);
m.mimo_uplink = file.mimo_uplink;
m.mimo_downlink = file.mimo_downlink;
m.Vmatrix = file.Vmatrix;

fprintf("准备开始合并数据\n")
for i = 1:epoch
    filename1 = "../data/"+type+"_"+num2str(carrnum)+"_"+num2str(BSAtNum)+"_"+num2str(UEAtNum)+"_"...
    +num2str(number)+"_ep"+num2str(i)+"of"+num2str(epoch)+".mat"; %保存路径
    file = matfile(filename1);
    m.mimo_uplink(batch*(i-1)+1:batch*i,:,:,:,:) = file.mimo_uplink;
    m.mimo_downlink(batch*(i-1)+1:batch*i,:,:,:,:) = file.mimo_downlink;
    fprintf("第%d epoch数据已合并\n",i)
end
% filename = "../data/mimo_"+num2str(BSAtNum)+"_"+num2str(UEAtNum)+"_"+num2str(number)+".mat" %保存路径
% save(filename,'mimo_uplink','mimo_downlink','-v7.3')
% fprintf("保存提示：所有数据均已保存\n")
