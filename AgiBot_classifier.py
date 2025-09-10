import torch
import torch.nn as nn
from transformers import ViTImageProcessor, ViTForImageClassification
from transformers import TrainingArguments, Trainer
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
import json
from datetime import datetime
import wandb
import random
from pathlib import Path

# Wandb 초기화 함수
def init_wandb(project_name="robot-task-classifier", run_name=None, config=None):
    if run_name is None:
        run_name = f"vit-base-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    wandb.init(
        project=project_name,
        name=run_name,
        config=config,
        tags=["ViT", "image-classification"]
    )
    print(f"Wandb 초기화 완료: {project_name}/{run_name}")

# ViT 모델과 프로세서 로드
def load_vit_model(num_classes, model_name="google/vit-base-patch16-224"):
    print(f"모델 로딩: {model_name}")
    print(f"분류 클래스 수: {num_classes}")

    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"모델 파라미터 수: {total_params:,}")

    return model, processor

# 커스텀 데이터셋 클래스
class RobotHeadDataset(Dataset):
    def __init__(self, image_paths, labels, processor, augment=False):
        self.image_paths = image_paths
        self.labels = labels
        self.processor = processor
        self.augment = augment
        print(f"데이터셋 생성 완료: {len(image_paths)}개 샘플, 증강: {augment}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')

            if self.augment and random.random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)

            inputs = self.processor(images=image, return_tensors="pt")

            return {
                'pixel_values': inputs['pixel_values'].squeeze(),
                'labels': torch.tensor(self.labels[idx], dtype=torch.long)
            }
        except Exception as e:
            print(f"이미지 로드 실패: {self.image_paths[idx]}, 에러: {e}")
            return self.__getitem__(0)

# 폴더명이 라벨인 구조에서 데이터 로드
def load_data_from_numbered_folders(base_dir):
    base_path = Path(base_dir).expanduser().resolve()
    print(f"데이터 로드 시작: {base_path}")

    all_image_paths = []
    all_labels = []
    class_folders = []
    class_distribution = {}

    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    # 숫자 폴더들 찾기
    for folder in sorted(base_path.iterdir()):
        if folder.is_dir() and folder.name.isdigit():
            class_label = int(folder.name)
            class_folders.append(class_label)

            print(f"클래스 {class_label} 폴더 처리 중: {folder}")

            images_found = 0
            for img_file in sorted(folder.iterdir()):
                if img_file.is_file() and img_file.suffix.lower() in supported_formats:
                    all_image_paths.append(str(img_file))
                    all_labels.append(class_label)
                    images_found += 1

            class_distribution[f"class_{class_label}"] = images_found
            print(f"  클래스 {class_label}: {images_found}개 이미지")

    # 라벨을 0부터 시작하도록 재매핑
    if class_folders:
        sorted_classes = sorted(class_folders)
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_classes)}

        print(f"라벨 재매핑: {label_mapping}")

        # 라벨 재매핑 적용
        all_labels = [label_mapping[label] for label in all_labels]

        # 클래스명
        class_names = [f"original_class_{old_label}" for old_label in sorted_classes]
        print(f"생성된 클래스명: {class_names}")
    else:
        class_names = []

    print(f"\n데이터 로드 완료:")
    print(f"  총 이미지: {len(all_image_paths)}개")
    print(f"  클래스 수: {len(class_folders)}개")

    return all_image_paths, all_labels, class_names

# 데이터 분할 함수
def prepare_and_split_data(base_dir, val_split=0.2, test_split=0.1, random_state=42):
    print(f"데이터 분할 시작: val={val_split}, test={test_split}")

    all_image_paths, all_labels, class_names = load_data_from_numbered_folders(base_dir)

    if len(all_image_paths) == 0:
        print("오류: 이미지를 찾을 수 없습니다.")
        return None

    unique_labels = list(set(all_labels))
    print(f"발견된 클래스: {sorted(unique_labels)}")

    # stratify 설정
    label_counts = {label: all_labels.count(label) for label in unique_labels}
    use_stratify = all(count >= 2 for count in label_counts.values())
    stratify_param = all_labels if use_stratify else None

    # 데이터 분할
    if test_split > 0:
        train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
            all_image_paths, all_labels,
            test_size=test_split,
            random_state=random_state,
            stratify=stratify_param
        )

        val_size_adjusted = val_split / (1 - test_split)
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_val_paths, train_val_labels,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=train_val_labels if use_stratify else None
        )

        print(f"데이터 분할 완료:")
        print(f"  훈련: {len(train_paths)}개")
        print(f"  검증: {len(val_paths)}개")
        print(f"  테스트: {len(test_paths)}개")

        return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, class_names

    else:
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            all_image_paths, all_labels,
            test_size=val_split,
            random_state=random_state,
            stratify=stratify_param
        )

        print(f"데이터 분할 완료:")
        print(f"  훈련: {len(train_paths)}개")
        print(f"  검증: {len(val_paths)}개")

        return train_paths, train_labels, val_paths, val_labels, class_names

# 모델 훈련 함수
def train_classifier(train_dataset, val_dataset, model, output_dir="./robot-task-classifier", config=None):
    if config is None:
        config = {}

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config.get('batch_size', 16),
        per_device_eval_batch_size=config.get('batch_size', 16),
        num_train_epochs=config.get('num_epochs', 10),
        learning_rate=config.get('learning_rate', 2e-5),
        weight_decay=config.get('weight_decay', 0.01),
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        warmup_steps=config.get('warmup_steps', 100),
        fp16=True,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        report_to="wandb" if wandb.run is not None else "none",
    )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(labels, predictions)
        return {"accuracy": accuracy}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print("=" * 50)
    print("훈련 시작")
    print(f"훈련 데이터: {len(train_dataset)}개")
    print(f"검증 데이터: {len(val_dataset)}개")
    print("=" * 50)

    trainer.train()
    trainer.save_model()
    print(f"모델이 {output_dir}에 저장되었습니다.")

    return trainer

# 메인 실행 함수
def main():
    config = {
        'data_base_dir': "../dataset",  # 로컬 데이터셋 경로로 변경
        'val_split': 0.2,
        'test_split': 0.1,
        'batch_size': 16,
        'num_epochs': 10,
        'learning_rate': 2e-5,
        'weight_decay': 0.01,
        'warmup_steps': 100,
        'random_state': 42
    }

    model_save_path = "./saved_models/robot_task_classifier"

    print("=" * 60)
    print("로봇 태스크 분류기 훈련 시작")
    print("=" * 60)

    # 데이터 준비
    result = prepare_and_split_data(
        config['data_base_dir'],
        val_split=config['val_split'],
        test_split=config['test_split'],
        random_state=config['random_state']
    )

    if result is None:
        print("데이터 로드 실패")
        return

    if len(result) == 7:
        train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, class_names = result
        has_test = True
    else:
        train_paths, train_labels, val_paths, val_labels, class_names = result
        has_test = False

    config['class_names'] = class_names
    print(f"발견된 클래스: {class_names}")

    if len(train_paths) == 0:
        print("오류: 훈련 데이터가 없습니다.")
        return

    # Wandb 초기화
    init_wandb(config=config)

    # 모델 로드
    model, processor = load_vit_model(len(class_names))

    # 데이터셋 생성
    train_dataset = RobotHeadDataset(train_paths, train_labels, processor, augment=False)
    val_dataset = RobotHeadDataset(val_paths, val_labels, processor, augment=False)

    # 모델 훈련
    trainer = train_classifier(train_dataset, val_dataset, model, model_save_path, config)

    # 모델 저장
    os.makedirs(model_save_path, exist_ok=True)
    model.save_pretrained(model_save_path)
    processor.save_pretrained(model_save_path)

    with open(os.path.join(model_save_path, "config.json"), "w") as f:
        json.dump({
            "class_names": class_names,
            "num_classes": len(class_names),
            "save_date": datetime.now().isoformat()
        }, f, indent=2)

    if wandb.run is not None:
        wandb.finish()

    print("=" * 60)
    print("훈련 완료!")
    print("=" * 60)

if __name__ == "__main__":
    main() 