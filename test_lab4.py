"""Проверки логики на синтетических изображениях, без реальных персональных данных."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from core import Database, descriptor, prepare, distance, np, cv2


class LabTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.db = Database(Path(self.temp.name)/'test.sqlite3')
        rng = np.random.default_rng(42)
        self.face = Image.fromarray(rng.integers(0,256,(160,160,3),dtype=np.uint8))
        self.feature = descriptor(self.face)
        self.photo = ('face.jpg',b'RGB',b'PNG',self.feature.tobytes())

    def tearDown(self):
        self.db.conn.close()
        self.temp.cleanup()

    def test_crud_password_and_cascade_delete(self):
        uid = self.db.save('test','Тест','secret1',[self.photo])
        self.assertTrue(self.db.check_password(uid,'secret1'))
        self.assertFalse(self.db.check_password(uid,'wrong'))
        self.db.save('new','Новое имя','secret2',[],uid)
        self.assertTrue(self.db.check_password(uid,'secret2'))
        self.assertFalse(self.db.check_password(uid,'secret1'))
        self.assertEqual(self.db.users()[0]['name'],'Новое имя')
        self.db.delete(uid)
        self.assertEqual(len(self.db.users()),0)
        self.assertEqual(self.db.conn.execute('SELECT COUNT(*) FROM photos').fetchone()[0],0)

    def test_identify_unknown_and_ambiguity(self):
        self.assertIsNone(self.db.identify(self.feature.tobytes())[0])
        uid = self.db.save('a','A','secret1',[self.photo])
        self.assertEqual(self.db.identify(self.feature.tobytes())[0]['id'],uid)
        other = descriptor(Image.new('RGB',(160,160),'white'))
        self.assertGreater(distance(other,self.feature),0.42)
        self.assertIsNone(self.db.identify(other.tobytes())[0])
        self.db.save('b','B','secret2',[self.photo])
        self.assertIn('Неоднозначное',self.db.identify(self.feature.tobytes())[2])

    def test_validation_atomicity_and_photos(self):
        with self.assertRaises(ValueError):
            self.db.save('empty','E','secret1',[])
        uid = self.db.save('test','T','secret1',[self.photo])
        with self.assertRaises(ValueError):
            self.db.save('TEST','T','secret1',[self.photo])
        self.assertEqual(len(self.db.users()),1)
        photo_id = self.db.photos(uid)[0]['id']
        with self.assertRaises(ValueError):
            self.db.delete_photo(photo_id,uid)
        self.db.save('test','T','',[self.photo],uid)
        self.db.delete_photo(photo_id,uid)
        self.assertEqual(len(self.db.photos(uid)),1)
        self.assertTrue(self.db.check_password(uid,'secret1'))

    def test_detection_and_rgb(self):
        path = Path(self.temp.name)/'снимок.jpg'
        self.face.save(path)
        class Detector:
            def read(self, node): return True
            def empty(self): return False
            def detectMultiScale(self,*args,**kwargs): return [(10,10,120,120)]
        with patch('core.cv2.CascadeClassifier',return_value=Detector()):
            photo = prepare(path)
            self.assertEqual(len(photo[3]),64*256*4)
            import io
            self.assertEqual(Image.open(io.BytesIO(photo[1])).mode,'RGB')
        # Реальный классификатор должен загрузиться, а пустой кадр — отклоняться.
        Image.new('RGB',(400,400),'white').save(path)
        with self.assertRaisesRegex(ValueError,'найдено лиц: 0'):
            prepare(path)
        with self.assertRaises(Exception):
            prepare(Path(self.temp.name)/'missing.jpg')

    def training_photos(self, offset, count=6):
        photos = []
        rng = np.random.default_rng(offset)
        for i in range(count):
            # Различимые синтетические распределения; не имитация реальных лиц.
            hist = np.full((64,256), 0.0001, dtype=np.float32)
            hist[:,offset] = 0.8
            hist[:,offset+1] = 0.1 + rng.random(64)*0.1
            hist /= hist.sum(axis=1,keepdims=True)
            photos.append((f'{i}.jpg',b'RGB',b'PNG',hist.astype('<f4').tobytes()))
        return photos

    def test_training_persistence_and_invalidation(self):
        a = self.training_photos(10)
        b = self.training_photos(100)
        uid = self.db.save('a','A','secret1',a)
        self.db.save('b','B','secret2',b)
        model = self.db.train()
        self.assertEqual(model['report']['train'],8)
        self.assertEqual(model['report']['validation'],4)
        self.assertAlmostEqual(sum(model['weights']),1)
        self.assertEqual(model['report']['far'],0)
        self.assertEqual(model['report']['frr'],0)
        # Распознавание использует веса модели и её порог без ручных параметров.
        self.assertEqual(self.db.identify(a[0][3])[0]['id'],uid)
        stored = Database(Path(self.temp.name)/'test.sqlite3')
        self.assertEqual(stored.model_state()[0],model)
        stored.conn.close()
        self.db.save('a','Новое имя','',[],uid)
        self.assertIsNotNone(self.db.model_state()[0])
        self.db.save('a','Новое имя','',[self.photo],uid)
        self.assertIsNone(self.db.model_state()[0])
        with self.assertRaisesRegex(ValueError,'повторно'):
            self.db.identify(a[0][3])
        self.db.train()
        self.assertIsNotNone(self.db.model_state()[0])

    def test_training_insufficient_duplicates_single_person(self):
        with self.assertRaises(ValueError):
            self.db.train()
        uid = self.db.save('a','A','secret1',[self.photo]*6)
        with self.assertRaisesRegex(ValueError,'5 разных'):
            self.db.train()
        self.db.save('a','A','',self.training_photos(10),uid)
        model = self.db.train()
        self.assertIsNone(model['report']['far'])
        self.assertEqual(model['report']['duplicates'],5)
        self.assertGreater(model['threshold'],0)

    def test_gui_start(self):
        from main import App
        with patch('main.Database',return_value=self.db):
            app = App()
            app.withdraw()
            app.update()
            self.assertEqual(len(app.tree.get_children()),0)
            app.destroy()


if __name__ == '__main__':
    unittest.main(verbosity=2)


