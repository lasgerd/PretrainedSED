clear;close all;clc
%%
data_all = readtable("mid_table_with_duration.csv");
data_interest = readlines("class_of_interest.txt");

%%
dur_all = sum(data_all.dur_s);
dur_interest = 0;
class_interest_str = [];
for i = 1:length(data_interest)
    class_interest = data_interest(i);
    if strlength(class_interest) == 0
        continue
    end

    idx = find(strcmpi(data_all.mid,class_interest));
    dur_interest = dur_interest + data_all.dur_s(idx);
    class_interest_str = [class_interest_str,data_all.name(idx)];
end

percent = dur_interest/dur_all*100;