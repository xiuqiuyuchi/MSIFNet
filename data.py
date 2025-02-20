import os
import json
from sklearn.model_selection import train_test_split


def get_image_paths_and_labels(root):
    classes = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    classes.sort()

    class_indices = {cla: idx for idx, cla in enumerate(classes)}

    image_paths = []
    labels = []
    for cla in classes:
        class_dir = os.path.join(root, cla)
        for img_name in os.listdir(class_dir):
            if img_name.endswith(('jpg', 'jpeg', 'png')):
                img_path = os.path.join(class_dir, img_name)
                image_paths.append(img_path)
                labels.append(class_indices[cla])

    return image_paths, labels, class_indices


def write_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


def sort_by_labels(paths, labels):
    sorted_data = sorted(zip(paths, labels), key=lambda x: x[1])
    sorted_paths, sorted_labels = zip(*sorted_data)

    return sorted_paths, sorted_labels


def get_data():
    root = 'E:/fl/AID'  # 修改为你的数据集路径
    train_ratio = 0.2  # 训练集比例

    image_paths, labels, class_indices = get_image_paths_and_labels(root)

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, labels, train_size=train_ratio,  stratify=labels)

    train_paths, train_labels = sort_by_labels(train_paths, train_labels)
    test_paths, test_labels = sort_by_labels(test_paths, test_labels)

    train_data = {path: label for path, label in zip(train_paths, train_labels)}
    test_data = {path: label for path, label in zip(test_paths, test_labels)}

    write_json(class_indices, 'class_indices.json')
    write_json(train_data, 'train_data.json')
    write_json(test_data, 'test_data.json')


if __name__ == "__main__":
    get_data()
