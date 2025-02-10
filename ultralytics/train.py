import warnings
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("")
    model.train(data='',
                cache=False,
                imgsz=640,
                epochs=300,
                single_cls=False, 
                batch=16,
                close_mosaic=0,
                workers=0,
                device='0',
                #optimizer='AdamW',
                optimizer='SGD',
                amp=True,
                project='',
                name='',
                )

