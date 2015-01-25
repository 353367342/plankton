function [ image_out ] = preprocess( image_in )

% Crop image
image_in = im_crop(image_in);

% Center and pad image (typically the last preproc step)
image_out = im_center(image_in, 350);

end

