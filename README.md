## Project : Real-Time Traffic Capacity and Speed Detection
[![Author](https://img.shields.io/badge/Author-Hasib%20Al%20Muzdadid-brightgreen)](https://github.com/HasibAlMuzdadid)
[![BSD 3-Clause License](https://img.shields.io/github/license/hasibalmuzdadid/Real-Time-Traffic-Capacity-and-Speed-Detection?style=flat&color=orange)](https://github.com/HasibAlMuzdadid/Real-Time-Traffic-Capacity-and-Speed-Detection/blob/main/LICENSE)
[![Stars](https://img.shields.io/github/stars/hasibalmuzdadid/Real-Time-Traffic-Capacity-and-Speed-Detection?style=social)](https://github.com/HasibAlMuzdadid/Real-Time-Traffic-Capacity-and-Speed-Detection/stargazers)

**Author :** </br>
Hasib Al Muzdadid</br>
[Department of Computer Science & Engineering](https://www.cse.ruet.ac.bd/), </br>
[Rajshahi University of Engineering & Technology (RUET)](https://www.ruet.ac.bd/) </br>
Portfolio: https://hasibalmuzdadid.github.io  </br> 


<p align="center">
   <img src="./files/real time traffic capacity and speed detection.gif" width="450" height="300"/>
</p>

## Project Description :
This project aims for a practical system to estimate vehicle speeds using merely cutting-edge computer vision technologies. This project demonstrates real-time vehicle detection, tracking, and speed estimation, leveraging the latest YOLO11-based detection combined with perspective transformations to map vehicle positions from camera views to real-world road coordinates.

**Language used :** Python 3.11.13 </br> 
**ML Framework :** PyTorch  </br>
**Video used :** The video footage that has been used was captured from a CCTV camera overlooking the M6 highway near Knutsford, UK. The video is available on <a href="https://www.youtube.com/watch?v=PNCJQkvALVc">YouTube</a>, providing an accessible dataset for this project.  </br>
**Model used :** YOLO11  </br> 

This system also enables valuable insights to be generated from real-time traffic patterns and vehicle speed analytics.

<p align="center">
  <img src="./files/speed distribution.png" width="45%" />
  <img src="./files/speed analysis.png" width="45%" />
</p>


## Setup :
For installing the necessary requirements, use the following bash snippet
```bash
git clone https://github.com/HasibAlMuzdadid/Real-Time-Traffic-Capacity-and-Speed-Detection.git
cd Real-Time-Traffic-Capacity-and-Speed-Detection
python -m venv myenv
myenv/Scripts/Activate 
pip install -r requirements.txt
```

For running the project, use the following bash snippet
```bash
!mkdir -p ./output ./output/video ./output/Insights
python main.py \
    --video_path "./input_video.mp4" \
    --output_video_path "./output/video/output_annotated.mp4" \
    --compressed_video_path "./output/video/output_annotated_compressed.mp4" \
    --output_insights_path "./output/Insights" \
    --csv_path "./output/vehicle_speeds.csv"
```
N.B: Modify the commands appropriately based on the terminal you are using. Change the coordinate values accordingly if you use different footage.



