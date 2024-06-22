import onnxruntime as ort
import numpy as np
import csv
import sys

def main(model_path, output_csv):
    try:
        # 加載模型
        session = ort.InferenceSession(model_path)
        
        # 加載測試數據
        test_data = np.load('MRI/test_data.npy')

        # 獲取模型輸入名稱
        input_name = session.get_inputs()[0].name

        # 推理
        outputs = session.run(None, {input_name: test_data})

        # 確保輸出結果是正確的格式
        outputs = outputs[0] if isinstance(outputs, list) and len(outputs) == 1 else outputs

        # 寫入CSV文件
        with open(output_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Output'])
            for output in outputs:
                writer.writerow(output if isinstance(output, (list, np.ndarray)) else [output])

        print(f'Test results saved to {output_csv}')

    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python test_MRI.py <model_path> <output_csv>")
    else:
        main(sys.argv[1], sys.argv[2])
