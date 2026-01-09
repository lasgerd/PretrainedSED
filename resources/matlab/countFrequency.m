% 统计不同类出现的总时长
clear;close all;clc
addpath(genpath("libs\"))
%%
mid2name_table = readtable("mid_to_display_name.tsv", 'FileType', 'text', 'Delimiter', '\t');
mid2name_table.dur_s = zeros(height(mid2name_table),1);
%%
all = dir("data\train\*.parquet");
wb = waitbar(0,'Calculating');
num = length(all);
for i = 1:num
    waitbar(i/num)
    file = all(i);
    data_struct = parquetread([file.folder '\' file.name]);

    for j = 1:height(data_struct)
        data_j = data_struct(j,:);
        info = data_j.raw_text{1};
        [time_list,label_list,mid_list] = parseParquetText(info);
        
        for k = 1:length(mid_list)
            idx = find(strcmpi(mid_list(k),mid2name_table.mid));
            if isempty(idx)
                error("Invalid idx")
            end

            mid2name_table.dur_s(idx) = mid2name_table.dur_s(idx) + time_list(k,2) - time_list(k,1);
        end

        % keyboard

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
close(wb)

keyboard
writetable(mid2name_table,"mid_table_with_duration.csv");