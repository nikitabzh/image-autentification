"""Локальная база RGB-изображений и идентификация по пространственным LBP."""
from pathlib import Path
import sys
BASE = Path(__file__).resolve().parent
if (BASE / '.packages').exists():
    sys.path.insert(0, str(BASE / '.packages'))
if (BASE / '.packages4').exists():
    sys.path.insert(0, str(BASE / '.packages4'))
import json
from training import fit, fingerprint, cell_distances
import hashlib
import hmac
import io
import secrets
import sqlite3
import numpy as np
from PIL import Image, ImageOps
import cv2
import neural_faces

THRESHOLD = 0.42
MARGIN = 0.04


def descriptor(face):
    """8×8 локальных гистограмм LBP, 256 кодов; сумма каждой равна 1."""
    gray = ImageOps.equalize(face.convert('L')).resize((130, 130))
    a = np.asarray(gray)
    center = a[1:-1, 1:-1]
    codes = np.zeros((128, 128), dtype=np.uint8)
    for bit, (dy, dx) in enumerate([(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]):
        codes |= ((a[1+dy:129+dy, 1+dx:129+dx] >= center).astype(np.uint8) << bit)
    return np.concatenate([np.bincount(codes[y:y+16, x:x+16].ravel(), minlength=256) / 256
                           for y in range(0,128,16) for x in range(0,128,16)]).astype('<f4')


def prepare(path, use_sface=False):
    """Читает RGB, находит ровно одно лицо; возвращает оригинал, кроп и признаки."""
    with Image.open(path) as source:
        rgb = ImageOps.exif_transpose(source).convert('RGB')
        rgb.load()
    if use_sface:
        try:
            face, vector = neural_faces.encode(rgb)
        except ValueError as exc:
            raise ValueError(f'{Path(path).name}: {exc}') from exc
        return pack_photo(path, rgb, face, vector.astype('<f4').tobytes())
    small = rgb.copy()
    small.thumbnail((1200,1200))
    # Чтение через Python поддерживает кириллицу в пути на Windows.
    xml = (Path(cv2.data.haarcascades) / 'haarcascade_frontalface_default.xml').read_text(encoding='utf-8')
    storage = cv2.FileStorage(xml, cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
    cascade = cv2.CascadeClassifier()
    cascade.read(storage.getFirstTopLevelNode())
    storage.release()
    if cascade.empty():
        raise ValueError('Не найден классификатор лиц OpenCV. Переустановите зависимости.')
    boxes = cascade.detectMultiScale(cv2.cvtColor(np.array(small), cv2.COLOR_RGB2GRAY),
                                    scaleFactor=1.1, minNeighbors=5, minSize=(50,50))
    if len(boxes) != 1:
        raise ValueError(f'{Path(path).name}: найдено лиц: {len(boxes)}. Нужно фото с одним хорошо видимым лицом.')
    x,y,w,h = boxes[0]
    sx,sy = rgb.width / small.width, rgb.height / small.height
    face = rgb.crop((int(x*sx),int(y*sy),int((x+w)*sx),int((y+h)*sy)))
    return pack_photo(path, rgb, face, descriptor(face).tobytes())


def pack_photo(path, rgb, face, feature):
    original, crop = io.BytesIO(), io.BytesIO()
    rgb.save(original, format='JPEG', quality=95)
    face.save(crop, format='PNG')
    return Path(path).name, original.getvalue(), crop.getvalue(), feature


def distance(a, b):
    """Среднее расстояние хи-квадрат по ячейкам, диапазон [0,1]."""
    return float(0.5 * np.sum((a-b)**2 / (a+b+1e-10)) / 64)


class Database:
    def __init__(self, path=None):
        path = Path(path or BASE / 'data' / 'faces.sqlite3')
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute('PRAGMA foreign_keys=ON')
        self.conn.executescript('''
        CREATE TABLE IF NOT EXISTS model(id INTEGER PRIMARY KEY CHECK(id=1), payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS users(
          id INTEGER PRIMARY KEY, login TEXT NOT NULL UNIQUE COLLATE NOCASE,
          name TEXT NOT NULL, salt BLOB NOT NULL, password BLOB NOT NULL,
          created TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE IF NOT EXISTS photos(
          id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
          filename TEXT NOT NULL, rgb BLOB NOT NULL, face BLOB NOT NULL, feature BLOB NOT NULL);
        ''')

    def users(self):
        return self.conn.execute('''SELECT u.id,u.login,u.name,u.created,COUNT(p.id) AS count
          FROM users u LEFT JOIN photos p ON p.user_id=u.id GROUP BY u.id ORDER BY u.login''').fetchall()

    def save(self, login, name, password, photos, user_id=None):
        login, name = login.strip(), name.strip()
        if not login or not name:
            raise ValueError('Заполните логин и ФИО.')
        if user_id is None and not photos:
            raise ValueError('Добавьте хотя бы одно фото лица.')
        if (user_id is None or password) and len(password) < 6:
            raise ValueError('Пароль должен содержать не менее 6 символов.')
        try:
            with self.conn:
                if user_id is None:
                    salt = secrets.token_bytes(16)
                    hashed = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 300000)
                    user_id = self.conn.execute('INSERT INTO users(login,name,salt,password) VALUES(?,?,?,?)',
                                                (login,name,salt,hashed)).lastrowid
                else:
                    self.conn.execute('UPDATE users SET login=?,name=? WHERE id=?',(login,name,user_id))
                    if password:
                        salt = secrets.token_bytes(16)
                        hashed = hashlib.pbkdf2_hmac('sha256',password.encode(),salt,300000)
                        self.conn.execute('UPDATE users SET salt=?,password=? WHERE id=?',(salt,hashed,user_id))
                self.conn.executemany('INSERT INTO photos(user_id,filename,rgb,face,feature) VALUES(?,?,?,?,?)',
                                      [(user_id,*photo) for photo in photos])
        except sqlite3.IntegrityError as exc:
            raise ValueError('Такой логин уже существует.') from exc
        return user_id

    def delete(self, user_id):
        with self.conn:
            self.conn.execute('DELETE FROM users WHERE id=?',(user_id,))

    def photos(self, user_id):
        return self.conn.execute('SELECT id,filename,face FROM photos WHERE user_id=?',(user_id,)).fetchall()

    def delete_photo(self, photo_id, user_id):
        with self.conn:
            if len(self.photos(user_id)) <= 1:
                raise ValueError('Нельзя удалить последнее фото. Сначала добавьте новое.')
            self.conn.execute('DELETE FROM photos WHERE id=? AND user_id=?',(photo_id,user_id))

    def check_password(self, user_id, password):
        row = self.conn.execute('SELECT salt,password FROM users WHERE id=?',(user_id,)).fetchone()
        return bool(row and hmac.compare_digest(row['password'],hashlib.pbkdf2_hmac('sha256',password.encode(),row['salt'],300000)))

    def feature_rows(self):
        return self.conn.execute('SELECT id,user_id,feature FROM photos ORDER BY id').fetchall()

    def has_model(self):
        return self.conn.execute('SELECT 1 FROM model WHERE id=1').fetchone() is not None

    def model_state(self):
        row = self.conn.execute('SELECT payload FROM model WHERE id=1').fetchone()
        if row is None:
            return None, 'Модель ещё не обучена. Используется исходный LBP.'
        model = json.loads(row['payload'])
        if model['fingerprint'] != fingerprint(self.feature_rows()):
            return None, 'Фотографии изменились. Нажмите «Обучить модель» повторно.'
        if model.get('backend') == 'sface':
            return model, f"Модель SFace: {model['report']['train']} эталонов. Порог {model['threshold']:.4f}."
        return model, f"Модель обучена: {model['report']['train']} обучающих и {model['report']['validation']} проверочных фото. Порог {model['threshold']:.4f}."

    def train(self, use_sface=False):
        if use_sface:
            rows = self.conn.execute('SELECT id,user_id,filename,rgb,feature FROM photos ORDER BY id').fetchall()
            model = neural_faces.fit(rows, fingerprint(rows))
        else:
            model = fit(self.feature_rows())
        with self.conn:
            self.conn.execute('INSERT OR REPLACE INTO model(id,payload) VALUES(1,?)',
                              (json.dumps(model,ensure_ascii=False),))
        return model

    def identify(self, feature, threshold=None):
        model, state = self.model_state()
        if model is None and self.has_model():
            raise ValueError(state)
        if threshold is None:
            threshold = model['threshold'] if model else THRESHOLD
        if not 0 < threshold <= 1:
            raise ValueError('Порог должен быть в диапазоне (0, 1].')
        query = np.frombuffer(feature,dtype='<f4')
        if model and model.get('backend') == 'sface':
            if query.size != 128:
                raise ValueError('Для модели SFace нужны признаки SFace.')
            return self.rank(neural_faces.scores(model, query), threshold)
        scores = {}
        references = set(model['reference_ids']) if model else None
        for row in self.feature_rows():
            if references is not None and row['id'] not in references:
                continue
            vector = np.frombuffer(row['feature'],dtype='<f4')
            d = float(cell_distances(query,vector) @ np.array(model['weights'])) if model else distance(query,vector)
            scores[row['user_id']] = min(scores.get(row['user_id'],1.0),d)
        return self.rank(scores, threshold)

    def rank(self, scores, threshold):
        ranked = sorted(scores.items(),key=lambda pair: pair[1])
        if not ranked:
            return None,None,'База пуста.'
        uid,score = ranked[0]
        if score > threshold:
            return None,score,f'Совпадение отклонено: расстояние выше порога {threshold:.4f}.'
        if len(ranked)>1 and ranked[1][1]-score < MARGIN:
            return None,score,'Неоднозначное совпадение: предоставьте другое фото.'
        return dict(self.conn.execute('SELECT id,login,name FROM users WHERE id=?',(uid,)).fetchone()),score,'Лицо распознано.'


