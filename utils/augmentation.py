import os
import pandas as pd
import soundfile as sf

def stereo_channel_swap(metadata_folder_path,wav_folder_path):

    # csv的azimuth变为相反数
    # 1. 遍历所有 CSV 文件
    for filename in os.listdir(metadata_folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(metadata_folder_path, filename)

            # 2. 读取 CSV 文件
            df = pd.read_csv(file_path)

            # 3. 修改数据（下面是示例操作）
            df['azimuth'] = -df['azimuth']

            # 4. 保存回 CSV
            filename2 = filename[0:-4] + '_swap.csv'
            df.to_csv(metadata_folder_path + '\\' + filename2, index=False)

    # 交换wav的左右声道
    # 遍历文件夹中的所有文件
    for filename in os.listdir(wav_folder_path):
        if filename.endswith('.wav'):
            file_path = os.path.join(wav_folder_path, filename)

            # 读取 wav 文件（data: ndarray, samplerate: int）
            data, samplerate = sf.read(file_path)

            # 检查是否为双声道
            if data.ndim == 2 and data.shape[1] == 2:
                # 交换左右声道： [:, [1, 0]]
                data_swapped = data[:, [1, 0]]

                # 构造输出路径
                filename2 = filename[0:-4] + '_swap.wav'
                output_path = os.path.join(wav_folder_path, f"{filename2}")

                # 保存新的 wav 文件
                sf.write(output_path, data_swapped, samplerate)

# # 1. 指定要遍历的文件夹路径
# metadata_folder_path1 = 'D:\项目汇总\A2H\code\SELD\seld1\dataset\metadata_dev\dev-test-sony'  # 替换为你的路径
# wav_folder_path1 = 'D:\项目汇总\A2H\code\SELD\seld1\dataset\stereo_dev\dev-test-sony'  # 替换为你的路径
# stereo_channel_swap(metadata_folder_path1,wav_folder_path1)