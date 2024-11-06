# Sign Language Detection 

## Purpouse

This project was made, to help me understand librarys such as:
<ul>
  <li>Scikit learn</li>
  <li>Pickle</li>
  <li>Numpy</li>
  <li>CV2</li>
</ul>

## How it works?
Using mediapipe and CV2 you can visualize hand gestures, it is added as an overlay on camera, and in real time is drawing lines on your hand, that can be later used 
to determine what letter you are currently showing.

## Data used 
Using tool that I created <a href = "https://github.com/GFuks27750/DataCollectionTool">link</a>, I collected more than 3000 pictures of me showing hand gestures to camera, this data is leter used in train_classifier.py to generate .pickle file 
