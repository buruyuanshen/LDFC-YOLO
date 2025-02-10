from ultralytics.models import RTDETR
 
if __name__ == '__main__':
    model = RTDETR(model='')
    model.train(pretrained=True, data='', epochs=300, batch=16, device=0, imgsz=640, workers=0)