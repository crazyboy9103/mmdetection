model=$1

if [ -z "$model" ]; then
  echo "Usage: $0 <model>"
  exit 1
fi

find "/mmdetection/configs/neurocle/$model" -name "*.py" | while read -r file; do
  # Perform some action on each file
  echo "Train with config: $file"
  python tools/train.py $file
  python tools/test.py
done