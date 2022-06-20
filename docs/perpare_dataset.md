# Perpare dataset

## Pascal Context

1. Download the raw dataset (zip) from docs/download.md.
2. Unzip the raw dataset.
3. Use the following script to convert the raw dataset to tfrecord, you need to adjust the paths:

```
python ids/tools/convert_record.py \ 
--convert_datasets=pascalcontext \
--pascalcontext_path={Downloaded raw dataset dir} \
--tfrecord_outputs={Your output path}
```

## COCOStuff10k

1. Download the raw dataset (zip) from docs/download.md.
2. Unzip the raw dataset.
3. Use the following script to convert the raw dataset to tfrecord, you need to adjust the paths:

```
python ids/tools/convert_record.py \ 
--convert_datasets=cocostuff10k \
--cocostuff10k_path={Downloaded raw dataset dir} \
--tfrecord_outputs={Your output path}
--compress=True
```
