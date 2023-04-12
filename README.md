## Ссылка для скачивания весов модели колоризации
https://github.com/pmh9960/iColoriT/tree/main/iColoriT_demo#pretrained-icolorit

## Ссылка для скачивания весов модели удаления текста
https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt

## Установка модели для удаления текста
```console
pip install -r requirements.txt
mim install mmengine
mim install 'mmcv>=2.0.0rc1'
mim install 'mmdet>=3.0.0rc0'

git clone https://github.com/open-mmlab/mmocr.git
cd mmocr
pip install -e .
```
