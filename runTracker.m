
% RUN_TRACKER  is the external function of the tracker - does initialization and calls trackerMain
    close all;clear;
%     addpath(genpath(pwd));%添加当前路径下的所有子目录
    %% Read params.txt
    params = readParams('params.txt');
	%% load video info

%     base_path = 'F:\徐福来\野外观测\database\benchmark50';
%     base_path =数据集的文件夹，如上一行;

    [sequence_path,video_name] = choose_video(base_path);
    if isempty(sequence_path), return, end  %user cancelled
    [img_files, pos, target_sz, ground_truth, video_path] = ...
	load_video_info2013(sequence_path);

	
    img_path = [sequence_path 'img/'];
    %% Read files

    im = imread([img_path img_files{1}]);
    if(size(im,3)==1)
        params.grayscale_sequence = true;
    end
    %图像缩小处理
%     params.resize_image = (sqrt(prod(target_sz)) >= 400);
%     if params.resize_image
% 		pos = floor(pos / 2);
% 		target_sz = floor(target_sz / 2);
%         im = imresize(im,0.5 );  
%     end
%         params.s00 = sqrt(target_sz(1)*target_sz(2)/8000);%计算目标大小是否超过了4000，如果是则缩小，否则
%         params.s00  = max(params.s00 ,1);                           %按照1倍即不处理
%         target_sz = floor(target_sz/params.s00 );           %目标大小以及位置均已按照缩放处理
%         pos = floor(pos/params.s00 );
%         im = imresize(im,1/params.s00 );   
        
    params.img_files = img_files;
    params.img_path = img_path;
    params.init_pos = pos;
    params.target_sz = target_sz;

    
    [params, bg_area, fg_area, area_resize_factor] = initializeAllAreas(im, params);
    
 	if params.visualization
 		params.videoPlayer = vision.VideoPlayer('Position', [100 100 [size(im,2), size(im,1)]+30]);
 	end
%     if params.visualization  %create video interface
% 		params.update_visualization = show_video(img_files, video_path);
% 	end

    % in runTracker we do not output anything
	params.fout = -1;
	% start the actual tracking
	results = trackerMain(params, im, bg_area, fg_area, area_resize_factor);
    results.res=[results.res(:,[2,1]) + results.res(:,[4,3]) / 2 , results.res(:,[4,3])];
    outname2 = './result/';
    dlmwrite([outname2 video_name '_Ours.txt'],results.res,'delimiter',',','newline','pc');

    [distance_precision, PASCAL_precision, average_center_location_error] = ...
    compute_performance_measures(results.res, ground_truth);
    fprintf('Center Location Error: %.3g pixels\nDistance Precision: %.3g %%\nOverlap Precision: %.3g %%\nSpeed: %.3g fps\n', ...
    average_center_location_error, 100*distance_precision, 100*PASCAL_precision, results.fps);
    
    fclose('all');

