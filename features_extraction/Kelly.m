function [MotionSense, MotionSenseMax]= Kelly(SpatialFrequencyPhi, MV_Norm, Fps)
MotionSenseSum = 0;
MotionSenseMax = 0;
phi = SpatialFrequencyPhi.*MV_Norm*Fps;
MCSF_Kelly=(6.1+7.3*abs(log(phi./(3*SpatialFrequencyPhi))).^3).*4*pi^2....
    .*phi.*SpatialFrequencyPhi.*exp(-4*pi*(2*SpatialFrequencyPhi+phi)/45.9);

% [x,y]=find((isnan(MCSF_Kelly))==0);
% for i=1:length(x)
%     MotionSenseSum = MotionSenseSum+MCSF_Kelly(x(i),y(i));
%     MotionSenseMax = max(MotionSenseMax,MCSF_Kelly(x(i),y(i)));
% end
% MotionSense = MotionSenseSum/length(x);

[x,y]=find((isnan(MCSF_Kelly))==1);
for i=1:length(x)
    MCSF_Kelly(x(i),y(i))=0;
end
MotionSense = mean(mean(MCSF_Kelly));
MotionSenseMax = max(max(MCSF_Kelly));

end