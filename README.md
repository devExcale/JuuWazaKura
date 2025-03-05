# JuuWazaKura

JuuWazaKura (柔技クラ) is an artificial intelligence model whose scope
is detecting and classifying judo techniques in video clips.

## Pipeline

1. Download competition videos from YouTube
2. Extract segments from competition videos given dataset
3. Include segments in dataset
4. Train model
5. Evaluate model

## Environment Variables

| Name                            | Description | Default          |
|---------------------------------|-------------|:-----------------|
| `concurrent_fragment_downloads` |             | `8`              |
| `dataset_source`                |             | `dataset/`       |
| `dataset_clips`                 |             | `dataset/clips/` |
| `delete_yt`                     |             | `True`           |
| `log_levelname`                 |             | `INFO`           |
