"""Строгая проверка пары CatBoost .cbm + манифест JSON.

Модуль намеренно не обучает модель и не заменяет старые модели автоматически.
Он используется перед расчётом оценки: несовпадение файлов, схемы или классов
делает модель непригодной для расчёта.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path

class ModelValidationError(RuntimeError):
    pass

def sha256(path: str | Path) -> str:
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def validate_model_pair(model_path: str | Path, manifest_path: str | Path, expected_features=None):
    model=Path(model_path); manifest=Path(manifest_path)
    if not model.is_file(): raise ModelValidationError(f"Не найден файл модели: {model}")
    if not manifest.is_file(): raise ModelValidationError(f"Не найден манифест модели: {manifest}")
    try: meta=json.loads(manifest.read_text(encoding='utf-8'))
    except Exception as exc: raise ModelValidationError(f"Некорректный JSON манифеста: {manifest}") from exc
    actual=sha256(model); declared=meta.get('model_sha256') or meta.get('sha256')
    if declared and actual.lower()!=str(declared).lower():
        raise ModelValidationError('Контрольная сумма .cbm не совпадает с манифестом; модель заблокирована.')
    features=meta.get('feature_columns') or meta.get('features')
    if expected_features is not None and features is not None and list(features)!=list(expected_features):
        raise ModelValidationError('Схема признаков модели не совпадает с текущей схемой; модель заблокирована.')
    classes=meta.get('classes')
    if classes is not None and set(classes)!={0,1} and set(classes)!={'0','1'}:
        raise ModelValidationError('Модель должна содержать оба класса 0 и 1.')
    return {'valid':True,'model_path':str(model),'manifest_path':str(manifest),
            'sha256':actual,'feature_columns':features,'classes':classes,
            'calibrated':bool(meta.get('calibrated',False))}
