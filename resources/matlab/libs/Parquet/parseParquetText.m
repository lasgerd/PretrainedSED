function [time_list,label_list,mid_list] = parseParquetText(parquet_info)
%PARSEPARQUETTEXT parse the raw text in the parquet to get useful info.
%   此处显示详细说明
%%
str_num = 5;
sub_str_list = [];

% 我只关心strong events
sub_str = char(parquet_info(2));
[startIndex,endIndex] = regexpi(sub_str,':');
sub_str2 = sub_str(endIndex+1:end);

[startIndex,endIndex] = regexpi(sub_str2,'\|');

time_list = [];
label_list = [];
mid_list = [];

for i = 1:length(startIndex)
    idx = startIndex(i);
    if i == 1
        sub_str3 = sub_str2(1:idx - 1);
    else
        sub_str3 = sub_str2(startIndex(i-1) + 1:idx - 1);
    end
    [time,label,mid] = splitStrongEvents(sub_str3);

    time_list = [time_list;time];
    label_list = [label_list;string(label)];
    mid_list = [mid_list;string(mid)];
end

if ~isempty(startIndex)
    sub_str3 = sub_str2(startIndex(end) + 1:end);
else
    sub_str3 = sub_str2;
end
[time,label,mid] = splitStrongEvents(sub_str3);

time_list = [time_list;time];
label_list = [label_list;string(label)];
mid_list = [mid_list;string(mid)];

end

