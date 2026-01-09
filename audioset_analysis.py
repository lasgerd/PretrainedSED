import pyarrow.parquet as pq
import json
import re
import pandas as pd
import io
import librosa
import numpy as np
import glob
import os

# 本文件的目的是给parquet中的每个样本增加采样权重

def bytes_to_array(b):
    try:
        # librosa 也可以直接处理 BytesIO
        y, _ = librosa.load(io.BytesIO(b), sr=16000)
        return y
    except Exception as e:
        return None

def parse_raw_text(text):
    match = re.search(r"Strong events:\s*(.*?)(?=Strong classes:|Tags:|$)", text, re.S)

    if not match:
        return []

    events_content = match.group(1).strip()

    # 2. 先按 " | " 分割各个事件，避免正则在长字符串中匹配错位
    raw_event_list = events_content.split(' | ')

    parsed_events = []

    # 3. 针对每个事件块进行精准匹配
    # 模式解释：
    # \((\d+\.\d+)s - (\d+\.\d+)s\) : 匹配起止时间
    # \s+(.*?) : 匹配标签，直到遇到后续的 MID 标识
    # \s+\((/[mt]/[^)]+)\) : 关键点！匹配以 /m/ 或 /t/ 开头的括号内容，[^)]+ 表示匹配到括号结束
    event_pattern = r"\((\d+\.\d+)s - (\d+\.\d+)s\)\s+(.*?)\s+\((/[mt]/[^)]+)\)"

    for event_str in raw_event_list:
        m = re.search(event_pattern, event_str.strip())
        if m:
            parsed_events.append({
                "onset": float(m.group(1)),
                "offset": float(m.group(2)),
                "event_label": m.group(3).strip(),
                "mid": m.group(4).strip()
            })

    return parsed_events

if __name__ == '__main__':
    # 读取整个文件
    parquet_folder_path = 'resources/'
    file_list = glob.glob(os.path.join(parquet_folder_path, "*.parquet"))
    for file_path in file_list:
        df = pd.read_parquet(file_path)
        sample_num = df.shape[0]
        sample_weight = np.zeros([sample_num,1])
        mid_table_with_dur = pd.read_csv('resources/matlab/mid_table_with_duration.csv',sep=',')
        mid_list = mid_table_with_dur['mid']
        mid_weight = mid_table_with_dur['weight']

        for i in range(sample_num):
            raw_text_i = df['raw_text'][i]
            events = parse_raw_text(raw_text_i[1])
            df_events = pd.DataFrame(events)
            for j in range(len(df_events)):
                result = mid_list[mid_list == df_events['mid'][j]]
                sample_weight[i] += (df_events['offset'][j] - df_events['onset'][j])*mid_weight[result.index[0]]

        if 'weight' not in df:
            df['weight'] = sample_weight
            df.to_parquet(file_path, index=False)

