function [alpha_ROI,overallstd_ROI,alpha_NROI,overallstd_NROI]= estimateROI(Image)

Image = rgb2gray(Image);

Image1 = double(Image);
window = fspecial('gaussian',7,7/6);
window = window/sum(sum(window));

mu = filter2(window, Image1, 'same');
mu_sq = mu.*mu;
sigma = sqrt(abs(filter2(window, Image1.*Image1, 'same') - mu_sq));
structdis = (Image1-mu)./(sigma+1);


if (size(Image,3)==1) %If gray-scale
    Image=repmat(Image,[1 1 3]);
end

HDPTH = .99;
DROPOFF = 10;
APERTURE = 1;
[fine,coarse] = computeSaliencyMap(Image,HDPTH,DROPOFF,APERTURE);

HDPTH = [.99 .98 .97];
DROPOFF = [10 20 100];
APERTURE = [1 2 0];
[multi] = computeSaliencyMap(Image,HDPTH,DROPOFF,APERTURE);

for i = 1:size(Image,1)
    for j = 1:size(Image,2)
        tt(i,j)=0;
        for k = 1:3
            yy(i,j,k)=Image(i,j,k)*(coarse(i,j)+fine(i,j)+multi(i,j,k))/3.0;
            tt(i,j)=tt(i,j)+yy(i,j,k)/3.0;
        end
        if tt(i,j)>20
            tt(i,j)=255;
        else
            tt(i,j)=0;
        end
    end
end

[x0,y0] = find(tt==255);
[x1,y1] = find(tt==0);

vec0 = zeros(1,length(x0));
for i =1:length(x0)
    vec0(i) = structdis(x0(i),y0(i));
end
[alpha_ROI,overallstd_ROI] = estimateGGDparam(vec0);

vec1 = zeros(1,length(x1));
for i =1:length(x1)
    vec1(i) = structdis(x1(i),y1(i));
end
[alpha_NROI,overallstd_NROI] = estimateGGDparam(vec1);

end