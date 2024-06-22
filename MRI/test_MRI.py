import onnxruntime as ort
import numpy as np
import csv
import sys
import time

def main(model_path, output_csv):
    try:
        # 加載模型
        session = ort.InferenceSession(model_path)
        
        # 獲取模型期望的輸入形狀
        input_shape = session.get_inputs()[0].shape
        print(f"Model expects input shape: {input_shape}")

        # 加載測試數據和真實標籤
        test_data = np.load('test_data.npy', allow_pickle=True).astype(np.float32)
        true_labels = np.load('test_labels.npy', allow_pickle=True)

        # 初始化計數器
        total_predictions = 0
        correct_predictions = 0
        total_inference_time = 0

        # 獲取模型輸入名稱
        input_name = session.get_inputs()[0].name

        # 逐個樣本進行推理
        for i in range(test_data.shape[0]):
            sample = test_data[i].reshape(input_shape)  # 重塑為期望的輸入形狀
            label = true_labels[i]

            # 計算推理時間
            start_time = time.time()
            outputs = session.run(None, {input_name: sample})
            inference_time = time.time() - start_time

            # 累加推理時間
            total_inference_time += inference_time

            # 假設輸出是概率，選擇最高概率作為預測結果
            prediction = np.argmax(outputs[0])

            # 更新計數器
            total_predictions += 1
            if prediction == label:
                correct_predictions += 1

        # 計算答對率
        accuracy = correct_predictions / total_predictions

        # 寫入CSV文件
        with open(output_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Accuracy', 'Total Inference Time'])
            writer.writerow([accuracy, total_inference_time])

        print(f'Test results saved to {output_csv}')
        print(f'Accuracy: {accuracy}, Total Inference Time: {total_inference_time} seconds')

    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python test_MRI.py <model_path> <output_csv>")
    else:
        main(sys.argv[1], sys.argv[2])
