function [results] = trackerMain(p, im, bg_area, fg_area, area_resize_factor)
%TRACKERMAIN contains the main loop of the tracker, P contains all the parameters set in runTracker
  %% INITIALIZATION
        num_frames = numel(p.img_files);
        % used for OTB-13 benchmark
        OTB_rect_positions = zeros(num_frames, 4);
        pos = p.init_pos;
        target_sz = p.target_sz;
      
  %% rot参数
        K = 5;    %pool的池大小
        lambda_l = 10;  %最小熵中的控制因数
       % s0 = 1;                              % initial scale factor 初始放缩因子
        Thr_Occ = 1;                         % threshold of occlusion 遮挡阈值
        Thr_Occ_learning_rate = 0.015;       %v遮挡距离学习参数;
        trans_vec8 = [-1,0;1,0;0,-1;0,1;-1,-1;1,-1;-1,1;1,1];%8 direction 8个方向
        % discriminate occlusion 用于判断遮挡
        cell_size = 8;                                       %size of hog feature, hog特征的范围大小
        target_sz_hog = floor(target_sz/cell_size);          %用于计算hog的目标大小
  %% initialize pool 为建立分类器池（pool）申请空间
        response_pool = cell(K,1);
        loss_pool = zeros(1,K);   %用来存储分类器池中的每个分类器的最小熵
        % for test
        m_response =  zeros(num_frames,1);
  %%
        % patch of the target + padding
        patch_padded = getSubwindow(im, pos, p.norm_bg_area, bg_area);%获得中心位于pos,大小为norm_bg_area的图片
        % initialize hist model，第一次，不需要更新，
        new_pwp_model = true;
        [bg_hist, fg_hist] = updateHistModel(new_pwp_model, patch_padded, bg_area, fg_area, target_sz, p.norm_bg_area, p.n_bins, p.grayscale_sequence);
        new_pwp_model = false;
        % Hann (cosine) window
        hann_window = single(hann(p.cf_response_size(1)) * hann(p.cf_response_size(2))');
        % gaussian-shaped desired response, centred in (1,1)
        % bandwidth proportional to target size
        %p.norm_target_sz是KCF中的target_sz
        %p.cf_response_size=params.norm_bg_area /
        %params.hog_cell_size，所以norm_bg_area=kcf中的windows——sz
        output_sigma = sqrt(prod(p.norm_target_sz)) * p.output_sigma_factor / p.hog_cell_size;
        y = gaussianResponse(p.cf_response_size, output_sigma);
        yf = fft2(y);
  %% SCALE ADAPTATION INITIALIZATION
        if p.scale_adaptation
             % Code from DSST
            scale_factor = 1;
            base_target_sz = target_sz;
            scale_sigma = sqrt(p.num_scales) * p.scale_sigma_factor;
            ss = (1:p.num_scales) - ceil(p.num_scales/2);
            ys = exp(-0.5 * (ss.^2) / scale_sigma^2);
            ysf = single(fft(ys));
            if mod(p.num_scales,2) == 0
                scale_window = single(hann(p.num_scales+1));
                scale_window = scale_window(2:end);
            else
                scale_window = single(hann(p.num_scales));
            end;

            ss = 1:p.num_scales;
            scale_factors = p.scale_step.^(ceil(p.num_scales/2) - ss);

            if p.scale_model_factor^2 * prod(p.norm_target_sz) > p.scale_model_max_area
                p.scale_model_factor = sqrt(p.scale_model_max_area/prod(p.norm_target_sz));
            end

            scale_model_sz = floor(p.norm_target_sz * p.scale_model_factor);
            % find maximum and minimum scales
            min_scale_factor = p.scale_step ^ ceil(log(max(5 ./ bg_area)) / log(p.scale_step));
            max_scale_factor = p.scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ target_sz)) / log(p.scale_step));
        end

    %% %%%%%%%%%%%%%%%%%%%%%%%%%%
        t_imread = 0;
    %% MAIN LOOP
        tic;
        for frame = 1:num_frames
%             p.norm_bg_area = floor(p.norm_bg_area*s0);
%             kcf_sz_s = floor(p.norm_bg_area*s0);
%             target_sz_s = floor(target_sz*s0);
            if frame == 1   
                %% 模板TRAINING
                % extract patch of size bg_area and resize to norm_bg_area
                im_patch_bg = getSubwindow(im, pos, p.norm_bg_area, bg_area);
                % compute feature map, of cf_response_size
                xt = getFeatureMap(im_patch_bg, p.feature_type, p.cf_response_size, p.hog_cell_size);
                % apply Hann window
                xt = bsxfun(@times, hann_window, xt);
                % compute FFT
                xtf = fft2(xt);
                hf_num = bsxfun(@times, conj(yf), xtf) / prod(p.cf_response_size);
                hf_den = (conj(xtf) .* xtf) / prod(p.cf_response_size);
                %% 直方图训练，已有
%                 new_pwp_model = true;
%                 [bg_hist, fg_hist] = updateHistModel(new_pwp_model, patch_padded, bg_area, fg_area, target_sz, p.norm_bg_area, p.n_bins, p.grayscale_sequence);
%                 new_pwp_model = false;
                %% SCALE UPDATE
                 im_patch_scale = getScaleSubwindow(im, pos, base_target_sz, scale_factor*scale_factors, scale_window, scale_model_sz, p.hog_scale_cell_size);
                 xsf = fft(im_patch_scale,[],2);
                 new_sf_num = bsxfun(@times, ysf, conj(xsf));
                 new_sf_den = sum(xsf .* conj(xsf), 1);
                 sf_den = new_sf_den;
                 sf_num = new_sf_num;       
                 %% initialize classifiers-pool 初始化分类器池
                [h,w,~] = size(im);%图片的高、宽、3
                pointer = 0; indp = pointer+1;
                n_class = 1; 
                % target patch pool，target_sz_s为45*31，target_sz也是，target_sz_hog为5*3
                tpatch = get_patch_s_hog(im, pos, target_sz, base_target_sz, target_sz_hog,cell_size);
                [dp,np] = size(tpatch);%31*15
                target_pool = zeros(dp,np,K);        
                target_pool(:,:,indp) = tpatch;   %indp=1,目标池的第一个元素存放了目标的hog特征

                % DSST model pool，分别保存了DSST模型参数的分子分母
                sf_num_pool = zeros([size(sf_num),K]);
                sf_den_pool = zeros([size(sf_den),K]);  

                sf_num_pool(:,:,indp) = sf_num;
                sf_den_pool(:,:,indp) = sf_den;
                % KCF model pool  w_kcf = 1;保存了kcf跟踪器的阿拉法与目标模板x
               % kcf_xtf_pool = zeros([size(xtf),K]); 
                kcf_hf_num_pool = zeros([size(hf_num),K]);
                kcf_hf_den_pool = zeros([size(hf_den),K]);

                %kcf_xtf_pool(:,:,indp) = xtf; 
                kcf_hf_num_pool(:,:,:,indp) = hf_num;
                kcf_hf_den_pool(:,:,:,indp) = hf_den;
               %直方图
                bg_hist_pool = zeros([size(bg_hist),K]);
                fg_hist_pool = zeros([size(fg_hist),K]);  
                if p.grayscale_sequence
                     bg_hist_pool (:,:,indp) = bg_hist;
                     fg_hist_pool(:,:,indp) = fg_hist;
                else
                    bg_hist_pool (:,:,:,indp) = bg_hist;
                    fg_hist_pool(:,:,:,indp) = fg_hist;
                end
                % selecte the threshold of high variation and occlusion 选择突变与遮挡的阈值
                n_trans = size(trans_vec8,1);   %等于8
                com_patch = zeros(dp,np,n_trans);
                count = 0;
                %计算出目标周围8个图片的hog特征
            for j = 1:n_trans
                    pos1 = pos + floor(target_sz).*trans_vec8(j,:);
                    if pos1(1)-target_sz(1)/2>0 & pos1(2)-target_sz(2)/2>0 & pos1(1)+target_sz(1)/2<h & pos1(2)+target_sz(2)/2<w
                        patch1 = get_patch_s_hog(im, pos1, target_sz, base_target_sz, target_sz_hog,cell_size);
                        count = count + 1;
                        com_patch(:,:,count) = patch1;
                    end
            end
    %         d_com = sqrt(sqdist(com_patch(:,:,1:count),tpatch))./n_tp;
            [d_com] =  LW_dist(com_patch(:,:,1:count),tpatch);
            Thr_Occ = 0.95*mean(d_com);
            end
    if frame>1
            tic_imread = tic;
            im = imread([p.img_path p.img_files{frame}]);
%             im = imresize(im,1/p.s00 );  
%               if p.resize_image
%                   im = imresize(im,0.5);
%               end
            t_imread = t_imread + toc(tic_imread);
	    %% TESTING step
            % extract patch of size bg_area and resize to norm_bg_area
            im_patch_cf = getSubwindow(im, pos, p.norm_bg_area, bg_area);
            pwp_search_area = round(p.norm_pwp_search_area / area_resize_factor);
            % extract patch of size pwp_search_area and resize to norm_pwp_search_area
            im_patch_pwp = getSubwindow(im, pos, p.norm_pwp_search_area, pwp_search_area);
            % compute feature map
            xt = getFeatureMap(im_patch_cf, p.feature_type, p.cf_response_size, p.hog_cell_size);
            % apply Hann window
            xt_windowed = bsxfun(@times, hann_window, xt);
            % compute FFT
            xtf = fft2(xt_windowed);
             %% occlusion or failed prediction判断是否发生遮挡，或者预测失败
            temp = target_pool(:,:,1:n_class);%第一帧之后n_class为1
            d_pool = LW_dist(temp,tpatch);
            min_d_pool = min(d_pool);
            if min_d_pool>Thr_Occ % completely occlusion, select the optimal classifier from classifier-pool 
            % 完全遮挡,从分类器池选择最佳分类器  
                for k = 1:n_class 
                    %%%% KCF model
                    %calculate response of the classifier at all shifts
                   % model_xf = kcf_xtf_pool(:,:,:,k);
                    model_hf_num = kcf_hf_num_pool(:,:,:,k);
                    model_hf_den = kcf_hf_den_pool(:,:,:,k);
                    if p.den_per_channel
                        hf = model_hf_num ./ (model_hf_den + p.lambda);
                    else
                        hf = bsxfun(@rdivide, model_hf_num, sum(model_hf_den, 3)+p.lambda);
                    end
                    response_cf = ensure_real(ifft2(sum(conj(hf) .* xtf, 3)));
                    response_cf = cropFilterResponse(response_cf, ...
                    floor_odd(p.norm_delta_area / p.hog_cell_size));
                    if p.hog_cell_size > 1
                    % Scale up to match center likelihood resolution.
                        response_cf = mexResize(response_cf, p.norm_delta_area,'auto');
                    end
                    %% 直方图
%                      model_bg = bg_hist_pool (:,:,:,k); 
%                      model_fg = fg_hist_pool(:,:,:,k);
                     if p.grayscale_sequence
                         model_bg = bg_hist_pool (:,:,k);
                         model_fg = fg_hist_pool(:,:,k);
                     else
                         model_bg = bg_hist_pool (:,:,:,k);
                         model_fg = fg_hist_pool(:,:,:,k);
                     end
                     [likelihood_map] = getColourMap(im_patch_pwp, model_bg, model_fg, p.n_bins, p.grayscale_sequence);
                    likelihood_map(isnan(likelihood_map)) = 0;
                    response_pwp = getCenterLikelihood(likelihood_map, p.norm_target_sz);

                %% ESTIMATION
                     response = mergeResponses(response_cf, response_pwp, p.merge_factor, p.merge_method,p.grayscale_sequence);
                    response_pool{k} = response;
                    loss_pool(k) = get_loss2(response,lambda_l);%最小熵的计算
                end
             [~,idx] = min(loss_pool);
             response = response_pool{idx};   %在各个分类器的能量函数E计算完毕后，选取最小的响应
            if max(response(:)) > 0.2
            % target location is at the maximum response
                [row, col] = find(response == max(response(:)), 1);
                center = (1+p.norm_delta_area) / 2;
                pos = pos + ([row, col] - center) / area_resize_factor;
                rect_position = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
                % update target patch
                tpatch = target_pool(:,:,idx);
                % update KCF pool
                hf_num=kcf_hf_num_pool(:,:,:,idx);
                hf_den=kcf_hf_den_pool(:,:,:,idx);
                % update DSST pool
                sf_num = sf_num_pool(:,:,idx);
                sf_den = sf_den_pool(:,:,idx);
                %更新直方图
                if p.grayscale_sequence
                    bg_hist=bg_hist_pool (:,:,idx);
                    fg_hist=fg_hist_pool(:,:,idx);
                else
                    bg_hist=bg_hist_pool (:,:,:,idx); 
                    fg_hist=fg_hist_pool(:,:,:,idx);
                end
%                  bg_hist=bg_hist_pool (:,:,:,idx); 
%                  fg_hist=fg_hist_pool(:,:,:,idx);
                 
          else
                if p.den_per_channel
                    hf = hf_num ./ (hf_den + p.lambda);
                else
                    hf = bsxfun(@rdivide, hf_num, sum(hf_den, 3)+p.lambda);
                end
                response_cf = ensure_real(ifft2(sum(conj(hf) .* xtf, 3)));
                response_cf = cropFilterResponse(response_cf, ...
                    floor_odd(p.norm_delta_area / p.hog_cell_size));
                if p.hog_cell_size > 1
                 response_cf = mexResize(response_cf, p.norm_delta_area,'auto');
                end
                [likelihood_map] = getColourMap(im_patch_pwp, bg_hist, fg_hist, p.n_bins, p.grayscale_sequence);
                % (TODO) in theory it should be at 0.5 (unseen colors shoud have max entropy)
                likelihood_map(isnan(likelihood_map)) = 0;
                response_pwp = getCenterLikelihood(likelihood_map, p.norm_target_sz);
                % ESTIMATION
                    response = mergeResponses(response_cf, response_pwp, p.merge_factor, p.merge_method,p.grayscale_sequence);
                    [row, col] = find(response == max(response(:)), 1);
                    center = (1+p.norm_delta_area) / 2;
                    pos = pos + ([row, col] - center) / area_resize_factor;
                    rect_position = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
            end
          else % without complete occlusion 无完全遮挡，使用之前的分类器
                 if p.den_per_channel
                        hf = hf_num ./ (hf_den + p.lambda);
                 else
                        hf = bsxfun(@rdivide, hf_num, sum(hf_den, 3)+p.lambda);
                 end
                response_cf = ensure_real(ifft2(sum(conj(hf) .* xtf, 3)));

            % Crop square search region (in feature pixels).
                response_cf = cropFilterResponse(response_cf, ...
                floor_odd(p.norm_delta_area / p.hog_cell_size));
                if p.hog_cell_size > 1
                % Scale up to match center likelihood resolution.
                    response_cf = mexResize(response_cf, p.norm_delta_area,'auto');
                end

                [likelihood_map] = getColourMap(im_patch_pwp, bg_hist, fg_hist, p.n_bins, p.grayscale_sequence);
            % (TODO) in theory it should be at 0.5 (unseen colors shoud have max entropy)
                likelihood_map(isnan(likelihood_map)) = 0;

            % each pixel of response_pwp loosely represents the likelihood that
            % the target (of size norm_target_sz) is centred on it
                response_pwp = getCenterLikelihood(likelihood_map, p.norm_target_sz);

            %% ESTIMATION
                response = mergeResponses(response_cf, response_pwp, p.merge_factor, p.merge_method,p.grayscale_sequence);
                [row, col] = find(response == max(response(:)), 1);
                center = (1+p.norm_delta_area) / 2;
                pos = pos + ([row, col] - center) / area_resize_factor;
                rect_position = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];

                if p.scale_adaptation
                im_patch_scale = getScaleSubwindow(im, pos, base_target_sz, scale_factor * scale_factors, scale_window, scale_model_sz, p.hog_scale_cell_size);
                xsf = fft(im_patch_scale,[],2);
                scale_response = real(ifft(sum(sf_num .* xsf, 1) ./ (sf_den + p.lambda) ));
                recovered_scale = ind2sub(size(scale_response),find(scale_response == max(scale_response(:)), 1));
                %set the scale
                scale_factor = scale_factor * scale_factors(recovered_scale);

                if scale_factor < min_scale_factor
                    scale_factor = min_scale_factor;
                elseif scale_factor > max_scale_factor
                    scale_factor = max_scale_factor;
                end
                % use new scale to update bboxes for target, filter, bg and fg models
                target_sz = round(base_target_sz * scale_factor);
                avg_dim = sum(target_sz)/2;
                bg_area = round(target_sz + avg_dim);
                if(bg_area(2)>size(im,2)),  bg_area(2)=size(im,2)-1;    end
                if(bg_area(1)>size(im,1)),  bg_area(1)=size(im,1)-1;    end

                bg_area = bg_area - mod(bg_area - target_sz, 2);
                fg_area = round(target_sz - avg_dim * p.inner_padding);
                fg_area = fg_area + mod(bg_area - fg_area, 2);
                % Compute the rectangle with (or close to) params.fixed_area and
                % same aspect ratio as the target bboxgetScaleSubwindow
                area_resize_factor = sqrt(p.fixed_area/prod(bg_area));
                end
            
            end
              
            m_response(frame) = max(response(:));
  %% discrimiate high variation using the new object box 用新得到的目标框 判断是否发生突变
           tpatch = get_patch_s_hog(im,pos,target_sz,base_target_sz,target_sz_hog,cell_size);  
  %% TRAINING
        % extract patch of size bg_area and resize to norm_bg_area
        im_patch_bg = getSubwindow(im, pos, p.norm_bg_area, bg_area);
        % compute feature map, of cf_response_size
        xt = getFeatureMap(im_patch_bg, p.feature_type, p.cf_response_size, p.hog_cell_size);
        % apply Hann window
        xt = bsxfun(@times, hann_window, xt);
        % compute FFT
        xtf = fft2(xt);
 %% FILTER UPDATE
        % Compute expectations over circular shifts,
        % therefore divide by number of pixels.
		new_hf_num = bsxfun(@times, conj(yf), xtf) / prod(p.cf_response_size);
		new_hf_den = (conj(xtf) .* xtf) / prod(p.cf_response_size);
         if frame == 1
            % first frame, train with a single image
		    hf_den = new_hf_den;
		    hf_num = new_hf_num;
		else
		    % subsequent frames, update the model by linear interpolation
        	hf_den = (1 - p.learning_rate_cf) * hf_den + p.learning_rate_cf * new_hf_den;
	   	 	hf_num = (1 - p.learning_rate_cf) * hf_num + p.learning_rate_cf * new_hf_num;

            %% BG/FG MODEL UPDATE
            % patch of the target + padding
            [bg_hist, fg_hist] = updateHistModel(new_pwp_model, im_patch_bg, bg_area, fg_area, target_sz, p.norm_bg_area, p.n_bins, p.grayscale_sequence, bg_hist, fg_hist, p.learning_rate_pwp);
        end

        %% SCALE UPDATE
        if p.scale_adaptation
            im_patch_scale = getScaleSubwindow(im, pos, base_target_sz, scale_factor*scale_factors, scale_window, scale_model_sz, p.hog_scale_cell_size);
            xsf = fft(im_patch_scale,[],2);
            new_sf_num = bsxfun(@times, ysf, conj(xsf));
            new_sf_den = sum(xsf .* conj(xsf), 1);
            if frame == 1,
                sf_den = new_sf_den;
                sf_num = new_sf_num;
            else
                sf_den = (1 - p.learning_rate_scale) * sf_den + p.learning_rate_scale * new_sf_den;
                sf_num = (1 - p.learning_rate_scale) * sf_num + p.learning_rate_scale * new_sf_num;
            end
        end
        temp = target_pool(:,:,1:n_class);
        d_pool = LW_dist(temp,tpatch);
        min_d_pool = min(d_pool);
        %该处阈值需要辨识
        scale_Thr1 = 0.6;
        Thr_Occ_p = scale_Thr1*Thr_Occ;
         
        if min_d_pool<Thr_Occ_p
                pointer = mod(pointer+1,K);
                indp = pointer+1;
                n_class = min(n_class+1,K);
                % update target patch;
                target_pool(:,:,indp) = tpatch;
                
                kcf_hf_num_pool(:,:,:,indp) = hf_num;
                kcf_hf_den_pool(:,:,:,indp) = hf_den;
                if p.grayscale_sequence
                     bg_hist_pool (:,:,indp) = bg_hist;
                     fg_hist_pool(:,:,indp) = fg_hist;
                else
                    bg_hist_pool (:,:,:,indp) = bg_hist;
                    fg_hist_pool(:,:,:,indp) = fg_hist;
                end
%                 bg_hist_pool (:,:,:,indp) = bg_hist;
%                 fg_hist_pool(:,:,:,indp) = fg_hist;
                sf_num_pool(:,:,indp) = sf_num;
                sf_den_pool(:,:,indp) = sf_den;
                
                n_trans = size(trans_vec8,1);
                com_patch = zeros(dp,np,n_trans);
                count = 0;
                for j = 1:n_trans
                        pos1 = pos + floor(target_sz*1).*trans_vec8(j,:);
                        pos1 = max(pos1,floor(target_sz/2));
                        pos1 = min(pos1,floor([h,w]-target_sz/2));

                        patch1 = get_patch_s_hog(im, pos1, target_sz, base_target_sz, target_sz_hog,cell_size);
                        count = count + 1;
                        com_patch(:,:,count) = patch1;
                end
                d_com = LW_dist(com_patch(:,:,1:count),tpatch);
                Thr_Occ = (1-Thr_Occ_learning_rate)*Thr_Occ+Thr_Occ_learning_rate*0.95*min(d_com);

        end
    end
          
        % update bbox position,修改了，为了配合rot
         if frame==1, rect_position = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])]; end
         rect_position_padded = [pos([2,1]) - bg_area([2,1])/2, bg_area([2,1])];

         OTB_rect_positions(frame,:) = rect_position;

         if p.fout > 0,  fprintf(p.fout,'%.2f,%.2f,%.2f,%.2f\n', rect_position(1),rect_position(2),rect_position(3),rect_position(4));   end

%         %% VISUALIZATION
          if p.visualization == 1
                 figure(1)
                  imshow(im)
                 rectangle('Position',rect_position, 'LineWidth',2, 'EdgeColor','g');
                 rectangle('Position',rect_position_padded, 'LineWidth',2, 'LineStyle','--', 'EdgeColor','b');
%               text(rect_position_padded(1,1)-10, rect_position_padded(1,2)-15, '熊猫', 'color', [1 0 0],'FontSize',18,'FontWeight','Bold');
                 
                  drawnow
% %              end
          end
%           if p.visualization == 1
%                 stop = update_visualization(frame, rect_position);
% 			if stop, break, end  %user pressed Esc, stop early
% 			
% 			drawnow
%           end
        end
    elapsed_time = toc;
    % save data for OTB-13 benchmark
    results.type = 'rect';
    results.res = OTB_rect_positions;
    results.fps = num_frames/(elapsed_time - t_imread);
end

% We want odd regions so that the central pixel can be exact
function y = floor_odd(x)
    y = 2*floor((x-1) / 2) + 1;
end

function y = ensure_real(x)
    assert(norm(imag(x(:))) <= 1e-5 * norm(real(x(:))));
    y = real(x);
end
