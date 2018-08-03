% Copyright 2018 Suguru KANOGA <s.kanouga@aist.go.jp>
%
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
%
%     http://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.

%検証用のプログラム
clear all

% main_folder
code_folder = 'C:\Users\sdxwh\Desktop\ADAMECS_matlab';
cd(code_folder);
data_folder = 'C:\Users\sdxwh\Desktop\ADAMECS_matlab\Debug';
addpath(data_folder);

%Day30に対して通常学習を行う
label_list = ["HO","WF","WE","RD","UD","FP","FS","HC", ...
    "FP_WF","FP_WE","FP_RD","FP_UD", ...
    "FS_WF","FS_WE","FS_RD","FS_UD", ...
    "HC_WF","HC_WE","HC_RD","HC_UD", ...
    "HC_FP","HC_FS"];

feature_lib = [];
final_label = [];

Fs_emg = 200;
dim = 2;
ch_num = 8;    % the number of electrodes
mov_num = 22;  % 22 movements
model_num = 5; % 5 LDA models
win_size = 50; % 250ms window
win_inc = 10;  % 50ms overlap
SampWin = 8;   % window size for modified sample entropy (40 ms)
SampTh = 0.4;  % SampEn threshold

%ラベルの作成
% duplicate class label for features
class_labels = 1:mov_num;
class_labels = repmat(class_labels,1,1);
class_labels = class_labels(:);

% translate label to -1 and 1 for each classifier model
labels = NaN(size(class_labels,1),model_num);

for i = 1:size(class_labels,1)
    % resting(-1)・actual movement(1)
    if class_labels(i,1) == 1
        labels(i,1) = 1;
    else
        labels(i,1) = -1;
    end
    
    % flexion(1)・extension(-1)
    if (class_labels(i,1) == 2) || (class_labels(i,1) == 9) || (class_labels(i,1) == 13) || (class_labels(i,1) == 17)
        labels(i,2) = 1;
    else
        if (class_labels(i,1) == 3) || (class_labels(i,1) == 10) || (class_labels(i,1) == 14) || (class_labels(i,1) == 18)
            labels(i,2) = -1;
        end
    end
    
    % radial deviation(1)・ulnar deviation(-1)
    if (class_labels(i,1) == 4) || (class_labels(i,1) == 11) || (class_labels(i,1) == 15) || (class_labels(i,1) == 19)
        labels(i,3) = 1;
    else
        if (class_labels(i,1) == 5) || (class_labels(i,1) == 12) || (class_labels(i,1) == 16) || (class_labels(i,1) == 20)
            labels(i,3) = -1;
        end
    end
    
    % pronation(1)・supination(-1)
    if (class_labels(i,1) == 6) || (class_labels(i,1) == 9) || (class_labels(i,1) == 10) || (class_labels(i,1) == 11) || (class_labels(i,1) == 12) || (class_labels(i,1) == 21)
        labels(i,4) = 1;
    else
        if (class_labels(i,1) == 7) || (class_labels(i,1) == 13) || (class_labels(i,1) == 14) || (class_labels(i,1) == 15) || (class_labels(i,1) == 16) || (class_labels(i,1) == 22)
            labels(i,4) = -1;
        end
    end
    
    % hand open(-1)・hand close(1)
    if (class_labels(i,1) == 8) || (class_labels(i,1) == 17) || (class_labels(i,1) == 18) || (class_labels(i,1) == 19) || (class_labels(i,1) == 20) || (class_labels(i,1) == 21) || (class_labels(i,1) == 22)
        labels(i,5) = -1;
    else
        if class_labels(i,1) == 1
            labels(i,5) = 1;
        end
    end
end

epoch = [];

for motion=1:mov_num
    %%%%%%%%%%%%%
    % loda data %
    %%%%%%%%%%%%%
    Logs.emg = csvread(label_list(motion)+'.csv');

    %%%%%%%%%%%%%%%%%%%%%%%%%%
    % apply high-pass filter %
    %%%%%%%%%%%%%%%%%%%%%%%%%%

    pre_emg = zeros(size(Logs.emg));

    a_emg = [1,-3.4789, 5.0098,-3.6995,1.3942,-0.2138];
    b_emg = [0.4624, -2.3119, 4.6238, -4.6238, 2.3119, -0.4624];

    for i = 1:ch_num
       pre_emg(:,i) = filtfilt(b_emg,a_emg,Logs.emg(:,i)); %250msのレンジにかけてもいいが，考慮必要
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % extract 1.5 s epochs of muscle activation %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %オンセットは1～3秒目から取り出す
    emg = pre_emg(1+Fs_emg*1:Fs_emg+Fs_emg*2,:);
    
    % Sample entropyでonsetを特定
    SampEn = zeros(ch_num, size(emg,1)-SampWin);
    
    for channel = 1:ch_num
        for n = 1:size(emg,1)-SampWin %SampWin ms毎にSampEnを計算　（学習にのみ使用すれば良い）  オンラインはSampEnいらないけど，レンジ250msで50msシフト　50msごとに結果出したい
            %if n~=47
            %   continue 
            %end
            data = emg(1+(n-1):SampWin+(n-1),channel); % 5ms shifting 
            r = 0.25 * std(data);                      % tolerance

            correl = zeros(1,2);
            dataMat = zeros(dim+1,SampWin-dim);

            for i = 1:dim+1
                dataMat(i,:) = data(i:SampWin-dim+i-1);
            end

            for m = dim:dim+1
               count = zeros(1,SampWin-dim);
               tempMat = dataMat(1:m,:);

               for i = 1:SampWin-m
                   % calculate Chebyshev distance extcuding
                   % self-matching case
                   dist = max(abs(tempMat(:,i+1:SampWin-dim) - repmat(tempMat(:,i),1,SampWin-dim-i)));

                   % calculate Heaviside function of the distance
                   %D = (dist < r);
                   D = 1./(1 + exp((dist-0.5)/r)); % Sigmoid function (mSampEn)　%0 or 1 

                   count(i) = sum(D)/(SampWin-dim);
               end
               correl(m-dim+1) = sum(count)/(SampWin-dim);
            end

            SampEn(channel,n) = log(correl(1)/correl(2));
        end
    end

    if(label_list(motion)=="HO")
        epoch = pre_emg(1+Fs_emg:Fs_emg+Fs_emg*1.5,:);
    else
        [maxvec, rowvec] = max(SampEn);
        [maxval, column] = max(maxvec);
        max_ch = find(SampEn(:,column)==maxval);
        TempSamp = SampEn(max_ch,:);
        % 正規化
        % もともとSampEnが最大のチャネル正規化後に，スレッショルドを超えてるチャネルからオンセットを決定
        [A B] = max(TempSamp,[],2);
        TempSamp = TempSamp./repmat(A,1,size(TempSamp,2)); % NaNは無視

        [row, col] = find(abs(TempSamp) > SampTh); % このcol(1,1)がもっともはやいactivated muscle
        if(size(col,1)==0)
           col(1,1)=1; 
        end
        epoch = pre_emg(col(1,1)+Fs_emg:Fs_emg*1.5+col(1,1)+(Fs_emg-1),:);
    end
    
    input = epoch;
    feature = extract_feature(input',win_size,win_inc);
    feature_lib = [feature_lib; feature];
    for k=1:size(feature,1)
        final_label = [final_label; labels(motion,:)];
    end
end

n_param = [];

for imodel=1:1:model_num
    training_data = feature_lib;
    training_labels = final_label(:,imodel);
    
    % ラベル変換をかませる 
    if imodel == 5
        training_labels(isnan(training_labels))=1;
    else
        training_labels(isnan(training_labels))=0;
    end
    
    % determine size of input data
    [n_tr,m] = size(training_data);

    % find and count unique class labels (in thic case k=2)
    class_label = unique(training_labels);
    k = length(class_label);

    % initialization
    n_group_tr = NaN(k,1);        % Group counts
    group_mean_tr = NaN(k,m);     % Group sample means
    pooled_cov_tr = zeros(m,m);   % Pooled covariance
    w = NaN(k,m+1);           % model coefficients

    % loop over classes to perform intermediate calculations
    for i = 1:k
         % establish location and size of each class
         group_tr = (training_labels==class_label(i));
         n_group_tr(i) = sum(double(group_tr));

        % calculate group mean vectors
        group_mean_tr(i,:) = mean(training_data(group_tr,:));

        % accumulate pooled covariance infromtion
        pooled_cov_tr = pooled_cov_tr + ((n_group_tr(i)-1) / (n_tr-k)).*cov(training_data(group_tr,:));
    end
    
    if imodel==2
       check_cov = pooled_cov_tr 
    end
    prior_prob = n_group_tr/n_tr;
    
    % loop over classes to calculate linear discriminant coefficients
    for i = 1:k
        % intermidiate calculation for efficiency
        % this replaces: group_mean(g,:) * inv(pooled_cov)
        temp = group_mean_tr(i,:) / pooled_cov_tr;

        % constant
        w(i,1) = -0.5 * temp * group_mean_tr(i,:)' + log(prior_prob(i));

        % linear
        w(i,2:end) = temp;
        
        if imodel==5
            if i==1
                temp
            end
        end
    end
    n_param = [n_param;w];
end      

%転移学習

tau = 0.4;
lambda = 0.6;

c_param = [];
new_param = [];
c_feature_lib = [];
c_final_label = [];

%WE,WF,RDをそれぞれ転移学習させる
for motion=2:1:4
    feature = []
    Logs.emg = csvread(label_list(motion)+'_t.csv');
       
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    % apply high-pass filter %
    %%%%%%%%%%%%%%%%%%%%%%%%%%

    pre_emg = zeros(size(Logs.emg));

    a_emg = [1,-3.4789, 5.0098,-3.6995,1.3942,-0.2138];
    b_emg = [0.4624, -2.3119, 4.6238, -4.6238, 2.3119, -0.4624];

    for i = 1:ch_num
       pre_emg(:,i) = filtfilt(b_emg,a_emg,Logs.emg(:,i)); %250msのレンジにかけてもいいが，考慮必要
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % extract 1.5 s epochs of muscle activation %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %オンセットは1～3秒目から取り出す
    emg = pre_emg(1+Fs_emg*1:Fs_emg+Fs_emg*2,:);
    
    % Sample entropyでonsetを特定
    SampEn = zeros(ch_num, size(emg,1)-SampWin);
    
    for channel = 1:ch_num
        for n = 1:size(emg,1)-SampWin %SampWin ms毎にSampEnを計算　（学習にのみ使用すれば良い）  オンラインはSampEnいらないけど，レンジ250msで50msシフト　50msごとに結果出したい
            %if n~=47
            %   continue 
            %end
            data = emg(1+(n-1):SampWin+(n-1),channel); % 5ms shifting 
            r = 0.25 * std(data);                      % tolerance

            correl = zeros(1,2);
            dataMat = zeros(dim+1,SampWin-dim);

            for i = 1:dim+1
                dataMat(i,:) = data(i:SampWin-dim+i-1);
            end

            for m = dim:dim+1
               count = zeros(1,SampWin-dim);
               tempMat = dataMat(1:m,:);

               for i = 1:SampWin-m
                   % calculate Chebyshev distance extcuding
                   % self-matching case
                   dist = max(abs(tempMat(:,i+1:SampWin-dim) - repmat(tempMat(:,i),1,SampWin-dim-i)));

                   % calculate Heaviside function of the distance
                   %D = (dist < r);
                   D = 1./(1 + exp((dist-0.5)/r)); % Sigmoid function (mSampEn)　%0 or 1 

                   count(i) = sum(D)/(SampWin-dim);
               end
               correl(m-dim+1) = sum(count)/(SampWin-dim);
            end

            SampEn(channel,n) = log(correl(1)/correl(2));
        end
    end

    if(label_list(motion)=="HO")
        epoch = pre_emg(1+Fs_emg:Fs_emg+Fs_emg*1.5,:);
    else
        [maxvec, rowvec] = max(SampEn);
        [maxval, column] = max(maxvec);
        max_ch = find(SampEn(:,column)==maxval);
        TempSamp = SampEn(max_ch,:);
        [A B] = max(TempSamp,[],2);
        TempSamp = TempSamp./repmat(A,1,size(TempSamp,2)); % NaNは無視

        [row, col] = find(abs(TempSamp) > SampTh); % このcol(1,1)がもっともはやいactivated muscle
        if(size(col,1)==0)
           col(1,1)=1; 
        end
        epoch = pre_emg(col(1,1)+Fs_emg:Fs_emg*1.5+col(1,1)+(Fs_emg-1),:);
    end
    
    input = epoch;
    feature = extract_feature(input',win_size,win_inc);
    c_feature_lib = [c_feature_lib; feature];
    for k=1:size(feature,1)
        c_final_label = [c_final_label; labels(motion,:)];
    end
end

result_w = [];

for imodel = 1:model_num
    training_data = feature_lib;
    training_labels = final_label(:,imodel);

    calibration_data = c_feature_lib;
    calibration_labels = c_final_label(:,imodel);    
   
    % ラベル変換をかませる 
    if imodel == 5
        training_labels(isnan(training_labels))=1;
        calibration_labels(isnan(calibration_labels))=1;
    else
        training_labels(isnan(training_labels))=0;
        calibration_labels(isnan(calibration_labels))=0;
    end
    
    % determine size of input data
    [n_tr,m] = size(training_data);
    [n_cal,~] = size(calibration_data);

    % find and count unique class labels (in thic case k=2)
    class_label = unique(training_labels);
    k = length(class_label);

    % initialization
    n_group_tr = NaN(k,1);        % Group counts
    group_mean_tr = NaN(k,m);     % Group sample means
    pooled_cov_tr = zeros(m,m);   % Pooled covariance
    w = NaN(k,m+1);           % model coefficients

    n_group_cal = NaN(k,1);
    group_mean_cal = NaN(k,m);
    pooled_cov_cal = zeros(m,m); 

    %全クラスの総数を数えておく
    m_and_n = 0
    
    for i = 1:k
        % establish location and size of each class
        group_tr = (training_labels==class_label(i));
        n_group_tr(i) = sum(double(group_tr));

        group_cal = (calibration_labels==class_label(i));
        n_group_cal(i) = sum(double(group_cal));   
        % もしキャリブレーションデータにそのクラスのデータがなければ学習データのを使う
        if n_group_cal(i) == 0
            m_and_n = m_and_n + size(training_data(group_tr,:),1)
        else
            m_and_n = m_and_n + size(calibration_data(group_cal,:),1)
        end 
    end
    
    % loop over classes to perform intermediate calculations
    for i = 1:k
        % establish location and size of each class
        group_tr = (training_labels==class_label(i));
        n_group_tr(i) = sum(double(group_tr));

        group_cal = (calibration_labels==class_label(i));
        n_group_cal(i) = sum(double(group_cal));     
        
        % クラス平均の計算
        % もしキャリブレーションデータにそのクラスのデータがなければ学習データのを使う
        if n_group_cal(i) == 0
            % calculate group mean vectors
            group_mean_tr(i,:) = mean(training_data(group_tr,:));
            group_mean_cal(i,:) = group_mean_tr(i,:);     

            % accumulate pooled covariance infromtion
            pooled_cov_tr = pooled_cov_tr + ((n_group_tr(i)-1) / (n_tr-k)).*cov(training_data(group_tr,:));
            pooled_cov_cal = pooled_cov_cal + ((n_group_tr(i)-1) / (m_and_n-k)).*cov(training_data(group_tr,:));

        else
            % calculate group mean vectors
            group_mean_tr(i,:) = mean(training_data(group_tr,:));
            group_mean_cal(i,:) = mean(calibration_data(group_cal,:));

            % accumulate pooled covariance infromtion
            pooled_cov_tr = pooled_cov_tr + ((n_group_tr(i)-1) / (n_tr-k)).*cov(training_data(group_tr,:));
            pooled_cov_cal = pooled_cov_cal + ((n_group_cal(i)-1) / (m_and_n-k)).*cov(calibration_data(group_cal,:));

        end     
    end
    
    % 学習データ内の各クラスのデータ数が変わってしまうことが予想されるので、可変にした
    prior_prob = (n_group_tr + n_group_cal) / (n_tr+n_cal);

    % loop over classes to calculate linear discriminant coefficients
    for i = 1:k
        % パラメータの転移
        new_group_mean = (1-tau).*group_mean_tr(i,:) + tau.*group_mean_cal(i,:);
        new_pooled_cov = (1-lambda).*pooled_cov_tr + lambda.*pooled_cov_cal;

        % intermidiate calculation for efficiency
        % this replaces: group_mean(g,:) * inv(pooled_cov)
        temp = new_group_mean / new_pooled_cov;

        % constant
        w(i,1) = -0.5 * temp * new_group_mean' + log(prior_prob(i));

        % linear
        
        w(i,2:end) = temp;
    end
    
    if imodel==2
       new_check_cov = new_pooled_cov 
    end

        
    result_w = [result_w;w]
end

%予測
k = [2,3,3,3,2];
w_index = 1;
final_result = [];
for model_ind =1:model_num

    testing_data = calibration_data;
    
    w = result_w(w_index:w_index+k(model_ind)-1,:);

    [w_index, w_index+k(model_ind)-1]
    
    w_index = w_index + k(model_ind);

    
    % 尤度計算
    L = [ones(size(testing_data,1),1) testing_data] * w';
    p = exp(L) ./ repmat(sum(exp(L),2),[1 size(L,2)]);

    % 3クラスとして予測
    predict = zeros(size(testing_data,1),1);
    for i = 1:size(testing_data,1)
        [~, predict(i,:)] = max(p(i,:));
    end

    % -1,0,1に変換
    if size(L,2) == 2
        predict(predict==1,:) = -1;
        predict(predict==2,:) = 1;
    else
        predict(predict==1,:) = -1;
        predict(predict==2,:) = 0;
        predict(predict==3,:) = 1;
    end
    
    final_result = [final_result,predict];
end
