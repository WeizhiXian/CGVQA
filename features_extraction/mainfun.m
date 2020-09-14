function Result = mainfun(VideoName,Fps,MbSize)

tic;

FramesPath = '';
interval = 50;

Obj = VideoReader(strcat(FramesPath,VideoName));
NumFrames = Obj.NumberOfFrames;

% write frames
for i = 1:NumFrames
    frame = read(Obj,i);
    imwrite(frame,strcat(FramesPath,num2str(i),'.jpg'),'jpg');
end

k = 1;
Entropy = zeros(1,NumFrames-1);
MSE = zeros(1,NumFrames-1);
Blur = zeros(1,NumFrames);
for i = 1:NumFrames-1
    First = imread([FramesPath,num2str(i),'.jpg']);
    Second = imread([FramesPath,num2str(i+1),'.jpg']);
    % DiffI = abs(Second - First);
    DiffI = abs(rgb2gray(Second - First));
    Entropy(i) = entropy(DiffI);% Entropy %
    MSE(i) = immse(First,Second);
    Blur(i) = JNBM_compute(First);
    if MSE(i)<5000
        SpatialFrequencyPhi = SpatialFrequency([FramesPath,num2str(i),'.jpg'],MbSize);
        [MV_X,MV_Y] = motionEstES(rgb2gray(First),rgb2gray(Second), MbSize, 7);
        MV_Norm=sqrt(MV_X.^2+MV_Y.^2);
        MV_Norm=rot90(MV_Norm);
        SpatialFrequencyPhi = flipud(SpatialFrequencyPhi)*1720*pi/180;
        [MotionSense(k), MotionSenseMax(k)]= Kelly(SpatialFrequencyPhi, MV_Norm, Fps);
        k=k+1;
    end
end
Blur(NumFrames) = JNBM_compute(Second);

Contrast = zeros(floor(NumFrames/interval),2);
alpha_ROI = zeros(1,floor(NumFrames/interval));
overallstd_ROI= zeros(1,floor(NumFrames/interval));
alpha_NROI= zeros(1,floor(NumFrames/interval));
overallstd_NROI = zeros(1,floor(NumFrames/interval));

k = 1;
for i = 1:interval:NumFrames
    Image = imread([FramesPath,num2str(i),'.jpg']);
    Contrast(k,:) = Contrast8(Image);
    [alpha_ROI(k),overallstd_ROI(k),alpha_NROI(k),overallstd_NROI(k)] = estimateROI(Image);
    k = k+1;
end

Result = [];

% feature1
ValMean_Entropy = mean(Entropy);
Result = [Result, ValMean_Entropy];

Sum = 0;
N = floor((NumFrames-1)/Fps);
for i = 1:N
    Sum = Sum + sum((Entropy(((i-1)*Fps+1):i*Fps)-mean(Entropy(((i-1)*Fps+1):i*Fps))).^2)/Fps;
end

% feature2
ValDev_Entropy = sqrt(Sum/N);
Result = [Result, ValDev_Entropy];

% feature3
ValMeanMotionSense=mean(MotionSense);
Result = [Result, ValMeanMotionSense];

% feature4
ValMaxMotionSense=max(MotionSenseMax);
Result = [Result, ValMaxMotionSense];

% feature5
ValMeanBlur=mean(Blur);
Result = [Result, ValMeanBlur];

% feature6
ValMaxBlur=max(Blur);
Result = [Result, ValMaxBlur];

% feature7
ValMeanContrast=mean(Contrast(:,1));
Result = [Result, ValMeanContrast];

% feature8
ValMaxContrast=max(Contrast(:,2));
Result = [Result, ValMaxContrast];

% feature9
ValMeanAlphaROI=mean(alpha_ROI);
Result = [Result, ValMeanAlphaROI];

% feature10
ValMeanOverallstdROI=mean(overallstd_ROI);
Result = [Result, ValMeanOverallstdROI];

% feature11
ValMeanAlphaNROI =mean(alpha_NROI);
Result = [Result, ValMeanAlphaNROI];

% feature12
ValMeanOverallstdNROI=mean(overallstd_NROI);
Result = [Result, ValMeanOverallstdNROI];

toc
end