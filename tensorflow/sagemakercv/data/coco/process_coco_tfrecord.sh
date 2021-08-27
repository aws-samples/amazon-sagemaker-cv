set -e
set -x


if [ -z "$1" ]; then
  echo "usage download_and_preprocess_coco.sh [data dir]"
  exit
fi

echo "Cloning Tensorflow models directory (for conversion utilities)"
if [ ! -e tf-models ]; then
  git clone http://github.com/tensorflow/models tf-models
fi

(cd tf-models/research && protoc object_detection/protos/*.proto --python_out=.)

mv tf-models/research/object_detection .
rm -rf tf-models

# Create the output directories.
COCO_DIR=$1
TRAIN_IMAGE_DIR=${COCO_DIR}/train2017
VAL_IMAGE_DIR=${COCO_DIR}/val2017
OUTPUT_DIR=$COCO_DIR/coco_tfrecord
TRAIN_OBJECT_ANNOTATIONS_FILE=$COCO_DIR/annotations/instances_train2017.json
VAL_OBJECT_ANNOTATIONS_FILE=$COCO_DIR/annotations/instances_val2017.json
TRAIN_CAPTION_FILE=$COCO_DIR/annotations/captions_train2017.json
VAL_CAPTION_FILE=$COCO_DIR/annotations/captions_val2017.json
mkdir -p "${OUTPUT_DIR}"

python create_coco_tf_record.py --logtostderr \
    --include_masks \
    --train_image_dir="${TRAIN_IMAGE_DIR}" \
    --val_image_dir="${VAL_IMAGE_DIR}" \
    --train_object_annotations_file="${TRAIN_OBJECT_ANNOTATIONS_FILE}" \
    --val_object_annotations_file="${VAL_OBJECT_ANNOTATIONS_FILE}" \
    --train_caption_annotations_file="${TRAIN_CAPTION_FILE}" \
    --val_caption_annotations_file="${VAL_CAPTION_FILE}" \
    --output_dir="${OUTPUT_DIR}"
    
rm -rf object_detection
