function [ ] = preprocess_wrapper( in_folder, out_folder )
% PREPROCESS_WRAPPER This function is a wrapper for PREPROCESS() that
% handles files and directories.

listing = dir(in_folder);
tocompute = length(listing);
fprintf(1,'Progress:  ');

% Check if directories or just images (test or train)
switch listing(3).isdir
    case 1
        for ii = 3:length(listing)
            % Display progress
            fprintf(1,'\r%d \t/ %d',ii,tocompute);
            
            % Preprocess
            in_subfolder = [in_folder '/' listing(ii).name];
            out_subfolder = [out_folder '/' listing(ii).name];
            mkdir(out_subfolder);
            % Match only jpegs (ignore Thumbs.db, etc)
            sublisting = dir([in_subfolder '/*.jpg']);
            for jj = 3:length(sublisting)
                in_image = imread([in_subfolder '/' sublisting(jj).name]);
                out_image = preprocess(in_image);
                imwrite(out_image,[out_subfolder '/' sublisting(jj).name]);
            end
        end
    case 0
        listing = dir([in_folder '/*.jpg']);
        mkdir(out_folder);
        for ii = 3:length(listing)
            % Display Progress
            fprintf(1,'\r%d \t/ %d',ii,tocompute);
            
            % Preprocess
            in_image = imread([in_folder '/' listing(ii).name]);
            out_image = preprocess(in_image);
            imwrite(out_image,[out_folder '/' listing(ii).name]);
        end        
end

end

