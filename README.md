# virtualGranny
Beased on video content generate voice and text as a story

Steps:
1. Download the complete folder and rename it to 'virtualGranny.'
2. Create a virtual environment and activate it.
3. To install necessary packages issue following command
>pip install -r requirements.txt

NOTE:  Make sure you have 'yolov3.weights' file inside folder 'yolo-coco.' If weight file is missing from the folder download it from drive link https://drive.google.com/drive/folders/1NPS8pPRlzhBnDFxVlWZTUoaZzkjepuYK?usp=sharing
and keep 'yolov3.weights' inside yolo-coco folder. At this time, you will have three files inside 'yolo-coco' folder.


Make sure you have an input video file inside a folder named 'videos.' If everything is okay, the system will generate voice as well as text as a story.

To test the script on input video issue following command
>python testVideo.py --input videos/input_video.mp4 --output output/input_video_story.mp4 --yolo yolo-coco

You can also save output video (detected objects in the video frame) by uncommenting the video writer lines in the testVideo.py



YOLO is trained on the actual images of objects, but it can also identify some objects in the cartoon images. The application will perform better if tested on real images of objects such as a person, sofa, dogs, etc.

To train with a new story, use 'trainLSTM.py' using the command:
>python trainLSTM.py

Copy the same story to the 'data' variable of 'testVideo.py' and test with the new input video. To avoid errors, try to keep only one object in a frame, with multiple object system can raise an error. However, with minor changes, you can use it to detect multiple objects
and generate a story.



Demo output : https://youtu.be/QM_XPjkS0As

----------Happy coding---------
