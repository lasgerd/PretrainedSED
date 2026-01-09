clear;close all;clc
%% 获取感兴趣的事件
mid2name_table = readtable("mid_to_display_name.tsv", 'FileType', 'text', 'Delimiter', '\t');

class_mid_of_interest = readlines("class_mid_of_interest.txt");
class_name_of_interest = [];
for i = 1:height(class_mid_of_interest)
    mid = class_mid_of_interest(i);
    if strlength(mid) == 0
        continue
    end

    idx = find(strcmpi(mid2name_table.mid,mid));
    if isempty(idx)
        error("invalid mid")
    end
    class_name_of_interest = [class_name_of_interest;{mid2name_table.name{idx}}];
end

mid_list = [];

%%
all = dir("../test-batch0563.parquet");
num = length(all);
for i = 1:num
    file = all(i);
    data_struct = parquetread([file.folder '\' file.name]);

    for j = 1:height(data_struct)
        data_j = data_struct(j,:);
        info = data_j.raw_text{1};

        if 0
            audio_bytes = data_j.audio.bytes{1};
            % 1. 将字节流写入临时文件 (Hugging Face 的音频通常是 .flac)
            temp_filename = 'temp_audio.flac';
            fid = fopen(temp_filename, 'wb');
            fwrite(fid, audio_bytes, 'uint8');
            fclose(fid);
            [y, fs] = audioread(temp_filename);
            sound(y,fs)
        end
    end
end