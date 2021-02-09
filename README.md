# Extreme-Tiny-Face-Detector
PyTorch implementation of EXTD , it can detect upto 800+ faces 



### Usage Example
```python
import cv2, random
from detectors import DSFD
from utils import draw_bboxes, crop_thumbnail, draw_bbox

# load detector with device(cpu or cuda)
DET = EXTD(device='cpu')

# load image in RGB
img = cv2.imread('bts.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# get bboxes with some confidence in scales for image pyramid
bboxes = DET.detect_faces(img, conf_th=0.9, scales=[0.5, 1])

# and draw bboxes on your image
img_bboxed = draw_bboxes(img, bboxes, fill=0.2, thickness=3)

# or crop thumbnail of someone
i = random.randrange(0, len(bboxes))
img_thumb, bbox_thumb = crop_thumbnail(img, bboxes[i], padding=1, size=100)

# you can use 'bbox_thumb' as bbox in thumbnail-coordinate system.
img_thumb_bboxed = draw_bbox(img_thumb, bbox_thumb)
```

### Weight Files

* [Tiny Face (trained on WIDER FACE)](https://drive.google.com/open?id=1vdKzrfQ4cXeI157NEJoeI1ECZ66GFEKE)
```
./detectors/tinyface/weights/checkpoint_50.pth
```

```

### Demo 01 : detect
```
python d_detect.py
```

![](https://github.com/kunalkarda/Extreme-Tiny-Face-Detector/blob/master/Figure_1.png)

Note that it shows bounding boxes only for default scale image *without image pyramid*. Number of bounding boxes ─ not detected faces ─ and minimum box sizes are as follows:

|                    | Tiny face| 
|         -          |    -     |    
|     # of boxes     |   686    |   
|  minimum box size  |   138    |    
|  minimum box size  |  36912   | 

### Demo 02 : crop
```
python d_crop.py
```

![](bts_demo.png)

### Face Size
* Minimum and maximum lengths of detected boxes are as follows:

|                    | MTCNN | Tiny face | S3FD  | DSFD  |
|         -          |   -   |     -     |   -   |   -   |
| min length (pixel) | 000.0 |   000.0   | 000.0 | 000.0 |
| max length (pixel) | 000.0 |   000.0   | 000.0 | 000.0 |

### References

* Tiny Face
    * [arXiv : Finding Tiny Faces](https://arxiv.org/abs/1612.04402)
    * [GitHub : tiny-faces-pytorch](https://github.com/varunagrawal/tiny-faces-pytorch)

