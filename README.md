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

| Name                            | Description                                                                     | Default          |
|---------------------------------|---------------------------------------------------------------------------------|:-----------------|
| `concurrent_fragment_downloads` | How many fragments to download concurrently, a yt-dlp parameter.                | `8`              |
| `concurrent_clippers`           | How many videos to clip concurrently.                                           | `4`              |
| `dataset_source`                | Path to the folder containing the csv files.                                    | `dataset/`       |
| `dataset_clips`                 | Path to the folder containing the video clips.                                  | `dataset/clips/` |
| `dataset_include`               | Comma-separated list of the YouTube videos to process. Empty list includes all. | `[]`             |
| `delete_yt`                     | Whether to delete the original competition videos after clipping.               | `True`           |
| `log_levelname`                 | The logging level name.                                                         | `INFO`           |
