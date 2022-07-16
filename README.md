# CGVQA

## Experiment Configurations
Python 3.6.10

Tenorflow-gpu 1.2.0

Matlab R2020a

## Introduction and Download Link of the Database
In this study, we first develop a useful CG animation subjective video quality database for the validation of corresponding VQA algorithms. It consists of 27 reference videos and 397 distorted videos. The distortion types include five compression-based distortion types and one transmission-based distortion type. All the videos in are High Definition (HD) content and above. More details of the database are shown as follows:
|||
|----------------------------------------|-------------------------------------------------------------------------|
| Number of Reference / Distorted Videos | 27 /397                                                                 |
| Sources                                | animated films, games                                                   |
| Resolution                             | 1270x720 (720p), 1920x1080 (1080p), <br>3840x2160 (4K UHD), 4096x2160 (DCI 4K) |
| Distortion Types|● AVC/H.264 compression: qp=22, 32, 42, 50 <br>● HEVC/H.265 compression: qp=32, 42, 47, 50 <br>● MPEG-2 compression: q=14, 31<br>● intraframe-only compression byMJPEG codec<br>● wavelet-based compression by Snow codec<br>● additive white noise: σ=0.003, 0.005, 0.01| 
| Frame Rate | 24fps, 30fps, 60fps |
|Subjective Test Setup|ITU-R Recommendation BT.500|
|Subjective Test Method|Single Stimulus (SS)|
|Subjective Score Data|mean opinion score (MOS)|
|Display|DELL U2720Q|
|Viewing Distance|2.5 picture heights|
|Number / Age / Famale Percentage of observers|25 / 22～35 / 44%|

The corresponding newly established database is available at 
**https://pan.baidu.com/s/1_P2ZNrLzJwZfG6xa6tKnDQ**
(password:**cgvq**)

## General Flowchart
![General Flowchart](https://github.com/WeizhiXian/CGVQA/blob/main/General%20Flowchart.png)

## Content Categories
![Content Categories](https://github.com/WeizhiXian/CGVQA/blob/main/Content%20Categories.png)
