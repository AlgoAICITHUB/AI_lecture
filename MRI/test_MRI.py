import onnxruntime as ort
import numpy as np
import csv
import sys

# 獲取命令行參數
model_path = sys.argv[1]
output_csv = sys.argv[2]

# 加載模型
session = ort.InferenceSession(model_path)

# 加載測試數據
test_data = np.load('MRI/test_data.npy')

# 獲取模型輸入名稱
input_name = session.get_inputs()[0].name

# 推理
outputs = session.run(None, {input_name: test_data})

# 寫入CSV文件
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Output'])
    for output in outputs:
        writer.writerow([output])

print(f'Test results saved to {output_csv}')
