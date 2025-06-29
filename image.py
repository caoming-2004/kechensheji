import pandas as pd
import numpy as np
from PIL import Image
import os
# 读取CSV文件
df = pd.read_csv('data/training_set.csv')



# 将十六进制字符串转换为二进制字节，并取前32个
df['binary_data'] = df['sha256'].apply(lambda x: bin(int(x, 16))[2:].zfill(len(x) * 4)[:32])

# 确保保存图片的目录存在
if not os.path.exists('data/images'):
    os.makedirs('data/images')

# 每32行生成一张图片
for i in range(0, len(df), 32):
    subset = df['sha256'][i:i + 32]
    if len(subset) == 32:
        # 创建一个空的32x32数组来存储灰度值
        gray_array = np.zeros((32, 32), dtype=np.uint8)
        for row_index, hex_str in enumerate(subset):
            # 将十六进制字符串转换为字节
            byte_data = bytes.fromhex(hex_str)
            byte_data_str = ''.join(format(byte, '08b') for byte in byte_data)
            # 取前32个字节（256位）
            byte_data_str = byte_data_str[:256]
            for col_index in range(0, min(len(byte_data_str), 256), 8):
                if col_index // 8 < 32:
                    # 将每8位二进制转换为灰度值
                    gray_value = int(byte_data_str[col_index:col_index + 8], 2)
                    gray_array[row_index, col_index // 8] = gray_value

        # 将数组转换为32*32的灰度图片
        img = Image.fromarray(gray_array, 'L')
        img.save(f'data/images/image_{i // 32}.png')