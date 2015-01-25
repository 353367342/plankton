function [ image_out ] = im_center( image_in, imsize )
%IM_CENTER im_center(image,size) pads an image to size x size and centers
%it based on its centroid.

image_out = uint8(255*ones(imsize));
centroid_struct = regionprops(double(im2bw(image_in)),'Centroid');
centroid = round(centroid_struct.Centroid);
x = ceil(imsize/2 - centroid(2));
y = ceil(imsize/2 - centroid(1));
image_out(x:(x+size(image_in,1)-1), y:(y+size(image_in,2)-1)) = image_in;

end

