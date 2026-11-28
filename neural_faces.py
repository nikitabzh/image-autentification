"""Выравнивание лица и признаки OpenCV YuNet/SFace."""
import io
from collections import Counter
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

MODELS = Path(__file__).resolve().parent / 'models'
DETECTOR = MODELS / 'face_detection_yunet_2023mar.onnx'
RECOGNIZER = MODELS / 'face_recognition_sface_2021dec_int8.onnx'
MAX_DISTANCE = 0.637
PREFERRED_DETECTION_SCORE = 0.90
MIN_DETECTION_SCORE = 0.80


def available():
    return DETECTOR.is_file() and RECOGNIZER.is_file()


@lru_cache(maxsize=1)
def networks():
    # Буферы позволяют открыть модели по пути с кириллицей на Windows.
    detector = cv2.FaceDetectorYN.create('onnx', DETECTOR.read_bytes(), b'', (320, 320), MIN_DETECTION_SCORE)
    recognizer = cv2.FaceRecognizerSF.create('onnx', RECOGNIZER.read_bytes(), b'')
    return detector, recognizer


def encode(rgb):
    detector, recognizer = networks()
    small = rgb.copy()
    small.thumbnail((960, 960))
    frame = cv2.cvtColor(np.asarray(small), cv2.COLOR_RGB2BGR)
    detector.setInputSize(small.size)
    _, faces = detector.detect(frame)
    if faces is not None:
        confident = faces[faces[:, -1] >= PREFERRED_DETECTION_SCORE]
        if len(confident):
            faces = confident
    count = 0 if faces is None else len(faces)
    if count != 1:
        raise ValueError(f'Найдено лиц: {count}. Выберите снимок с одним лицом или обрежьте его.')
    aligned = recognizer.alignCrop(frame, faces[0])
    vector = recognizer.feature(aligned).ravel()
    return Image.fromarray(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)), vector / np.linalg.norm(vector)

def fit(rows, fingerprint):
    """Калибрует порог, поочерёдно исключая проверяемое фото из эталонов."""
    references = []
    seen = {}
    for row in rows:
        try:
            _, vector = encode(Image.open(io.BytesIO(row['rgb'])).convert('RGB'))
        except ValueError as exc:
            raise ValueError(f"{row['filename']}: {exc}") from exc
        key = vector.tobytes()
        if key in seen:
            if seen[key] != row['user_id']:
                raise ValueError('Одинаковое фото записано у разных пользователей.')
            continue
        seen[key] = row['user_id']
        references.append((row['user_id'], vector))
    counts = Counter(user_id for user_id, _ in references)
    if not counts or min(counts.values()) < 5:
        raise ValueError('Для обучения нужно минимум 5 разных фотографий каждого пользователя.')
    labels = np.array([user_id for user_id, _ in references])
    vectors = np.stack([vector for _, vector in references])
    distances = np.clip(1 - vectors @ vectors.T, 0, 2)
    np.fill_diagonal(distances, np.inf)
    same = labels[:, None] == labels[None, :]
    genuine = np.where(same, distances, np.inf).min(axis=1)
    impostor = np.where(same, np.inf, distances).min(axis=1)
    multiple_users = len(counts) > 1
    if multiple_users:
        boundaries = np.unique(np.clip(np.r_[0, genuine, impostor, MAX_DISTANCE], 0, MAX_DISTANCE))
        candidates = [*list((boundaries[:-1] + boundaries[1:]) / 2), MAX_DISTANCE]
        def errors(limit):
            far = np.mean(impostor <= limit)
            frr = np.mean(genuine > limit)
            return (far + frr) / 2, far
        threshold = float(min(candidates, key=errors))
    else:
        threshold = float(np.clip(genuine.max() * 1.1, 0.01, MAX_DISTANCE))
    return {
        'backend': 'sface', 'fingerprint': fingerprint, 'threshold': threshold,
        'labels': labels.tolist(), 'vectors': vectors.tolist(),
        'report': {'users': len(counts), 'train': len(references), 'validation': len(references),
                   'far': float(np.mean(impostor <= threshold)) if multiple_users else None,
                   'frr': float(np.mean(genuine > threshold))}
    }


def scores(model, vector):
    distances = np.clip(1 - np.asarray(model['vectors'], dtype=np.float32) @ vector, 0, 2)
    best = {}
    for uid, distance in zip(model['labels'], distances):
        best[uid] = min(best.get(uid, 2), float(distance))
    return best
