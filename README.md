<h1 align="center">Video Analytics Tool using YOLOv11 and Streamlit</h1>

## :innocent: Motivation
As AI engineers, we love data and we love to see graphs and numbers! So why not project the inference data on some platform to understand the inference better? When a model is deployed on the edge for some kind of monitoring, it takes up rigorous amount of frontend and backend developement apart from deep learning efforts — from getting the live data to displaying the correct output. So, I wanted to replicate a small scale video analytics tool and understand what all feature would be useful for such a tool and what could be the limitations?

## :key: Features

<ol>
    <li>Choose input source - Local, RTSP or Webcam</li>
    <li>Input class threshold</li>
    <li>Set FPS drop warning threshold</li>
    <li>Option to save inference video</li>
    <li>Input class confidence for drift detection</li>
    <li>Option to save poor performing frames</li>
    <li>Display objects in current frame</li>
    <li>Display total detected objects so far</li>
    <li>Display System stats - Ram, CPU, GPU usage</li>
    <li>Display poor performing class</li>
    <li>Display minimum, maximum, and average FPS recorded during inference</li>
</ol> 

## :dizzy: How to use?
<ol>
    <li>Clone this repo</li>
    <li>Install all the dependencies</li>
    <li>Run -> 'streamlit run app.py' or 'python -m streamlit run app.py' in Windows.</li>
</ol>

## :star: Recent changelog
<ol>
    <li>Updated YOLOv5 to YOLOv11</li>
    <li>Replaced DeepSORT with YOLO's default BoT-SORT tracker</li>
    <li>Bug fixes, refactoring, performance boosters</li>
</ol>

## Note
The input video should be in same folder where app.py is. If you want to deploy the app in cloud and use it as a webapp then - download the user uploaded video to temporary folder and pass the path and video name to the respective function in app.py . This is Streamlit bug. Check <a href="https://stackoverflow.com/questions/65612750/how-can-i-specify-the-exact-folder-in-streamlit-for-the-uploaded-file-to-be-save">Stackoverflow</a>.
