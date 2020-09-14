function SF = SpatialFrequency(filename_image,MbSize)

RGB = imread(filename_image);

% convert image to greyscale if its not b&w already
if size(RGB,3) ~= 1
    I = rgb2gray(RGB);
else
    I=RGB;
end

% convert image to double type. necesarry for sqrt function
I2 = im2double(I);

% M=number of rows; N=number of columns in the image
M= size(I2,1);
N= size(I2,2);
X = floor(M/MbSize);
Y = floor(N/MbSize);

for x = 1:X
    for y = 1:Y
        
        % calculate Horizontal Frequency HF
        SumHF=0;
        for i=((x-1)*MbSize+1):(x*MbSize)
            for j=((y-1)*MbSize+2):(y*MbSize)
                SumHF = SumHF + (I2(i,j)-I2(i,j-1))^2;
            end
        end
        HF=sqrt(SumHF/(MbSize*(MbSize-1)));
        
        % calculate Vertical Frequency VF
        SumVF=0;
        for i=((x-1)*MbSize+2):(x*MbSize)
            for j=((y-1)*MbSize+1):(y*MbSize)
                SumVF = SumVF + (I2(i,j)-I2(i-1,j))^2;
            end
        end
        VF=sqrt(SumVF/(MbSize*(MbSize-1)));
        
        % calculate diagonal Frequency DF
        SumDF1=0;
        SumDF2=0;
        for i=((x-1)*MbSize+2):(x*MbSize)
            for j=((y-1)*MbSize+2):(y*MbSize)
                SumDF1 = SumDF1 + (I2(i,j)-I2(i-1,j-1))^2;
                SumDF2 = SumDF2 + (I2(i-1,j)-I2(i,j-1))^2;
            end
        end
        DF=sqrt(SumDF1/((MbSize-1)*(MbSize-1)))+sqrt(SumDF1/((MbSize-1)*(MbSize-1)));
        
        % calculate Spatial Frequency SF output
        SF(x,y)=sqrt(HF^2+VF^2+DF^2);
    end
end

