# Txt2image Runner

txt2image runner aims to automate the proccess of generating the dataset images from our sentiment analysis dataset.

## Architecture

![Txt2image runner](assets/arc.jpg)

## Installing dependencies

```pip install -r requirements.txt```

## Using
1. Export needed environment variables;
```shell
$ export DATASET_PATH=~/foo/bar          # Default is ./dataset/airline_reviews.csv
$ export IS_RESTAURANT_REVIEW=true       # Use only if you are running on Google restaurant reviews dataset
```
2. Start generating samples
```shell
$ make run
```
3. If a sample coud not be generated, it will be logged to `sampling_errors.log`.