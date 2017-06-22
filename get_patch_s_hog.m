function [patch_hog] = get_patch_s_hog(im, pos, target_sz_s, target_sz,target_sz_hog,cell_size)

%  get_patch_s_hog(im, pos, target_sz_s, target_sz, target_sz_hog,cell_size)
% Extracts patch from image im at position pos and
% window size sz. 

if isscalar(target_sz_s),  %square sub-window
    target_sz_s = [target_sz_s, target_sz_s];
end

xs = floor(pos(2)) + (1:target_sz_s(2)) - floor(target_sz_s(2)/2);
ys = floor(pos(1)) + (1:target_sz_s(1)) - floor(target_sz_s(1)/2);

%check for out-of-bounds coordinates, and set them to the values at
%the borders
xs(xs < 1) = 1;
ys(ys < 1) = 1;
xs(xs > size(im,2)) = size(im,2);
ys(ys > size(im,1)) = size(im,1);

%extract image
im_patch = double(im(ys, xs, :));
im_patch = imresize(im_patch,target_sz);
patch_hog = double(fhog(single(im_patch) / 255, cell_size, 9));
patch_hog(:,:,end)=[];
patch_hog = reshape(patch_hog,prod(target_sz_hog),[]);
patch_hog = patch_hog';
patch_hog = bsxfun(@rdivide,patch_hog,sum(patch_hog,1));%normalize
end

