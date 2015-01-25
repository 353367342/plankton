function [ max_size ] = largest_image( in_folder )
%LARGEST_IMAGE Tool to find the largest dimensions of all images in a
%folder

max_size = [0 0];
listing = dir(in_folder);
tocompute = length(listing);

% Check if directories or just images (test or train)
switch listing(3).isdir
    case 1
        for ii = 3:length(listing)
            % Display progress
            fprintf(1,'\r%d \t/ %d',ii,tocompute);
            
            % Count
            in_subfolder = [in_folder '/' listing(ii).name];
            % Match only jpegs (ignore Thumbs.db, etc)
            sublisting = dir([in_subfolder '/*.jpg']);
            for jj = 3:length(sublisting)
                in_image = imread([in_subfolder '/' sublisting(jj).name]);
                max_size = (size(in_image) > max_size).*size(in_image) +...
                    (size(in_image) <= max_size).*max_size;
            end
        end
    case 0
        listing = dir([in_folder '/*.jpg']);
        for ii = 3:length(listing)
            % Display progress
            fprintf(1,'\r%d \t/ %d',ii,tocompute);
            
            % Count
            in_image = imread([in_folder '/' listing(ii).name]);
            max_size = (size(in_image) > max_size).*size(in_image) + ...
                (size(in_image) <= max_size).*max_size;
        end        
end

end

