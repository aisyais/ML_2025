#Nur Aisyah_F55123042_Kelas B
#Program UAS Machine Learning
# KNN manual hanya dengan bantuan numpy
# Kelas dataset : DN, DD, B, A
#Evaluasi Model : Akurasi, Confusion Matrix, Classification Report. semua dibangun secara manual hanya dengan bantuan numpy

import numpy as np
# fungsi untuk Load Dataset
def load_dataset(path):
    data = np.genfromtxt(path, delimiter=",", dtype=str, skip_header=1)
    plate_texts = data[:, 0]
    labels = data[:, 1]  # tetap string seperti "DN", "DD"
    return plate_texts, labels

def extract_prefix(plate):
    return plate.split()[0]

def encode_prefix(prefix):
    mapping = {"DN": 0, "DD": 1, "B": 2}
    return mapping.get(prefix, 3)

def prepare_features(plates):
    return np.array([[encode_prefix(extract_prefix(p))] for p in plates])

# KNN dibangun secara manual hanya dengan bantuan numpy
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b)**2, axis=1))

def predict_knn(X_train, y_train, x_test, k=3):
    distances = euclidean_distance(X_train, x_test)
    k_indices = np.argsort(distances)[:k]
    k_labels = y_train[k_indices]
    # Ambil mayoritas label (string)
    values, counts = np.unique(k_labels, return_counts=True)
    return values[np.argmax(counts)]

# fungsi untuk melatih dan mengevaluasi model
def train_and_evaluate(csv_path):
    plates, labels = load_dataset(csv_path)
    features = prepare_features(plates)

    # Membagi dataset menjadi train dan test
    np.random.seed(42)
    indices = np.arange(len(features))
    np.random.shuffle(indices)
    split = int(0.8 * len(features))
    train_idx, test_idx = indices[:split], indices[split:]

    X_train, y_train = features[train_idx], labels[train_idx]
    X_test, y_test = features[test_idx], labels[test_idx]

    # Prediksi semua data test
    y_pred = []
    correct = 0
    for i in range(len(X_test)):
        pred = predict_knn(X_train, y_train, X_test[i])
        y_pred.append(pred)
        if pred == y_test[i]:
            correct += 1

    y_pred = np.array(y_pred)
    accuracy = correct / len(X_test)
    print(f"\n📊 Akurasi model: {accuracy * 100:.2f}%")

    # Confusion Matrix dan Classification Report. keduanya dibangun secara manual hanya dengan numpy 
    print("Confusion Matrix:")
    labels_unique = np.unique(np.concatenate((y_test, y_pred)))
    label_to_index = {label: idx for idx, label in enumerate(labels_unique)}
    matrix = np.zeros((len(labels_unique), len(labels_unique)), dtype=int)

    for actual, predicted in zip(y_test, y_pred):
        i = label_to_index[actual]
        j = label_to_index[predicted]
        matrix[i, j] += 1

    print("           Predicted")
    print("        ", "  ".join([f"{label:>5}" for label in labels_unique]))
    for idx, label in enumerate(labels_unique):
        row = "  ".join([f"{val:>5}" for val in matrix[idx]])
        print(f"Actual {label:>5}  {row}")

    # classification report using numpy only
    print("Classification Report:")
    for idx, label in enumerate(labels_unique):
        TP = matrix[idx, idx]
        FP = matrix[:, idx].sum() - TP
        FN = matrix[idx, :].sum() - TP
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"{label:>5} | Precision: {precision:.2f}  Recall: {recall:.2f}  F1-score: {f1:.2f}")

    return X_train, y_train

# fungsi untuk memprediksi plat nomor baru
def predict_new_plate(X_train, y_train):
    plate_input = input("\n🔍 Masukkan plat nomor yang ingin diprediksi: ").strip()
    encoded = np.array([[encode_prefix(extract_prefix(plate_input))]])
    result = predict_knn(X_train, y_train, encoded)
    print(f"Hasil Prediksi: {result}")

# bagian main untuk menjalankan program
if __name__ == "__main__":
    csv_path =r"D:\anggun\dataset\datasetuasmachlearning.csv"
    X_train, y_train = train_and_evaluate(csv_path)

    while True:
        predict_new_plate(X_train, y_train)
        lanjut = input("Ingin prediksi lagi? (y/n): ").strip().lower()
        if lanjut != "y":
            break
