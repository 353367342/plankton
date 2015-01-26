function [ image_out ] = im_crop( image_in )
%IM_CROP Crops image to tightest non-white extents

% Crop dim 1
first1 = find(min(image_in,[],1) ~= 255,1,'first');
last1 = find(min(image_in,[],1) ~= 255,1,'last');

% Crop dim 2
first2 = find(min(image_in,[],2) ~= 255,1,'first');
last2 = find(min(image_in,[],2) ~= 255,1,'last');

image_out = image_in(first2:last2, first1:last1);

end

