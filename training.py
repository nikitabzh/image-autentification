"""Обучение весов LBP и калибровка порога на отдельных снимках."""
import hashlib
import numpy as np


def fingerprint(rows):
    digest = hashlib.sha256(b'weighted-lbp-v1')
    for r in rows:
        digest.update(f"{r['id']}:{r['user_id']}:".encode())
        digest.update(r['feature'])
    return digest.hexdigest()


def cell_distances(a, b):
    return (0.5 * ((a-b)**2 / (a+b+1e-10)).reshape(64,256).sum(axis=1))


def fit(rows):
    groups = {}
    seen = {}
    for row in rows:
        key = hashlib.sha256(row['feature']).hexdigest()
        if key in seen and seen[key] != row['user_id']:
            raise ValueError('Одинаковое изображение записано у разных пользователей. Исправьте базу.')
        seen[key] = row['user_id']
        groups.setdefault(row['user_id'], {})[key] = row
    if not groups or any(len(g)<5 for g in groups.values()):
        raise ValueError('Для обучения нужно минимум 5 разных фотографий каждого пользователя. '
                         'Добавьте их через «Изменить». Копии одного снимка не считаются разными.')
    train, validation = [], []
    # Разделение воспроизводимо и не зависит от порядка загрузки файлов.
    for group in groups.values():
        ordered = [group[k] for k in sorted(group)]
        n = max(2, len(ordered)//4)
        validation.extend(ordered[:n])
        train.extend(ordered[n:])
    vectors = {r['id']: np.frombuffer(r['feature'],dtype='<f4') for r in train+validation}
    centers = {uid: np.mean([vectors[r['id']] for r in train if r['user_id']==uid],axis=0)
               for uid in groups}
    within = np.mean([cell_distances(vectors[r['id']], centers[r['user_id']]) for r in train],axis=0)
    if len(groups)>1:
        overall = np.mean(list(centers.values()),axis=0)
        between = np.mean([cell_distances(c,overall) for c in centers.values()],axis=0)
        raw = (between+0.001)/(within+0.001)
    else:
        raw = 1/(within+0.001)
    # Ограничение весов уменьшает переобучение на маленькой базе.
    weights = np.clip(raw/(np.mean(raw)+1e-10),0.25,4)
    weights /= weights.sum()
    genuine, impostor, correct = [], [], 0
    for row in validation:
        scores = {}
        for reference in train:
            d = float(cell_distances(vectors[row['id']],vectors[reference['id']]) @ weights)
            uid = reference['user_id']
            scores[uid] = min(scores.get(uid,1.0),d)
        genuine.append(scores[row['user_id']])
        other = [d for uid,d in scores.items() if uid!=row['user_id']]
        if other:
            impostor.append(min(other))
        correct += min(scores,key=scores.get)==row['user_id']
    if impostor:
        values = sorted(set(genuine+impostor))
        candidates = [max(1e-6,values[0]/2)] + [(a+b)/2 for a,b in zip(values,values[1:])] + [min(1,values[-1]+1e-6)]
        def errors(t):
            far = float(np.mean(np.array(impostor)<=t))
            frr = float(np.mean(np.array(genuine)>t))
            return (far+frr)/2,far,t
        threshold = min(candidates,key=errors)
    else:
        threshold = min(0.95,max(0.01,float(np.max(genuine))*1.1))
    return dict(version=1, fingerprint=fingerprint(rows), weights=weights.tolist(),
                threshold=threshold, reference_ids=[r['id'] for r in train],
                report=dict(users=len(groups),train=len(train),validation=len(validation),
                            duplicates=len(rows)-sum(len(g) for g in groups.values()),
                            far=float(np.mean(np.array(impostor)<=threshold)) if impostor else None,
                            frr=float(np.mean(np.array(genuine)>threshold)),
                            rank1=correct/len(validation)))
