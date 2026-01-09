function [time,label,mid] = splitStrongEvents(str)
%SPLITSTRONGEVENTS 
%   分离形如 (0.000s - 10.000s) Music (/m/04rlf) 的字符串
%%
[idx] = regexpi(str,' ');
str(idx) = [];

% tokens = regexp(str,'\(([^()]+)\)([^()]+)\(([^()]+)\)','tokens');
inside = regexp(str, '(?<=\()[^()]+(?=\))', 'match');
label = regexp(str, '(?<=\))[^()]+(?=\()', 'match');
label = label{1};

time_str = inside{1};
% label = tokens{1}{2};
mid = inside{end};

if strcmpi(mid,'microphone')
    keyboard
end

nums = sscanf(time_str, '%fs-%fs');
start_time = nums(1);
end_time   = nums(2);
time = [start_time end_time];
end

