function [varargout] = computeSaliencyMap(I,HDPTH,DROPOFF,APERTURE)
% Input variables:
% I - UINT8 RGB image
% HDPTH - (0-1) percentage of points to be selected as non-HDP. i.e. 0.99
% will select 0.01 of the points as HDP.
% DROPOFF - (positive number) dropoff rate of reciprocity function
% APERTURE - (non-negative) aperture size of PDF functions. To disable
% select 0.

%USAGE:

% [fine coarse] = computeSaliencyMap(IM,HDPTH,DROPOFF,APERTURE) :
% HDPTH,DROPOFF,APERTURE are all scalar values. The output will be both the
% fine & coarse represenations.

% [multi] = computeSaliencyMap(IM,HDPTH,DROPOFF,APERTURE) :
% HDPTH,DROPOFF,APERTURE are all same length vectors of length m. The output will be both the
% an m-multi-layered saliency represenation.
error(nargoutchk(1, 2, nargout));
if (nargout==2)
    if (any([numel(HDPTH) numel(DROPOFF) numel(APERTURE)] ~= 1)) 
        error('HDPTH,DROPOFF,APERTURE parameters should contain exactly 1 value each');
    end
    [fine coarse] = salCoreCalc(I,HDPTH,DROPOFF,APERTURE);
    varargout{1} = fine;
    varargout{2} = coarse;
else
    mLevels = numel(HDPTH);
    if (any([numel(HDPTH) numel(DROPOFF) numel(APERTURE)] ~= mLevels)) 
        error('HDPTH,DROPOFF,APERTURE parameters should contain exactly the same number of values each');
    end
    [multi] = salCoreCalc(I,HDPTH,DROPOFF,APERTURE);
    varargout{1} = multi;
end


end



