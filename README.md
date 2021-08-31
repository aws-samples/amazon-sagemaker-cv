## Yolov3 for SMCV

This branch is staging for merge yolov3 into smcv.

Yolov3 files include: 
  pytorch/sagemakercv/detection/backbone/darknet.py
  pytorch/sagemakercv/detection/dense_heads/inference.py
  pytorch/sagemakercv/detection/dense_heads/yolo_anchor_generator.py
  pytorch/sagemakercv/detection/dense_heads/yolo_head.py
  pytorch/sagemakercv/detection/dense_heads/yolo_loss.py
  pytorch/sagemakercv/detection/detector/yolo_detector.py

Files/notebooks to run yolov3:
  pytorch/sagemakercv/yolo_scratch.ipynb
  pytorch/sagemakercv/yolo_tune.ipynb
  pytorch/tools/ben_testing.ipynb
  pytorch/tools/configs/st_yolo.yaml
  pytorch/tools/run_yolo.ipynb
  pytorch/tools/train_yolo.py
  pytorch/tools/train_yolo.sh
  pytorch/tools/yolo_scratch.ipynb

Scripts to train darknet:
  pytorch/darknet_trainer

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

