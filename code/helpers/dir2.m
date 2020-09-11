function output_list = dir2(full_path,is_dir)
%dir2 List of folders in specified path (can include extension)
%   Same as dir but gives back names directly and removes
%   files/non-folders outputs
% Usage: output_list = dir2(full_path,is_dir)
% is_dir: 0 or 1 list files only (0, default) or folders only (1)
if nargin<2
    is_dir=0;
end
output_list=dir(full_path);
output_list={output_list([output_list.isdir]==is_dir).name};
output_list(strcmp(output_list,'.')|strcmp(output_list,'..'))=[];
end

