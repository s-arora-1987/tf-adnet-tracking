
In most of online trackers for moving objects, an object is recognized by choosing between bounding boxes corresponding to locations likely for the object, the boxes are called region proposals or regions of interest. Region-of-Interest (RoI) pooling has been shown to improve the processing speed of object recognizer, but it has never been tried on an object tracker. The investigation reveals whether a tool improving object recognition processes is effective in object tracking or not. The investigation reveals whether a tool improving object recognition processes is effective in object tracking or not. 

This project aims to investigate the enhancements of “Action-Decision Networks for Visual Tracking with Deep Reinforcement Learning” (ADNet). This project tries to investigate an improvement in ADNet by introducing region of interest pooling in the misprediction prevention mechanism of ADNet. After trying these changes, I compared the tracking performance of new network and that of old network (without RoI) on the OTB dataset used for ADNet paper. The results showed that RoI improves both accuracy and learning duration. 

This project is built on the code forked from https://github.com/ildoonet/tf-adnet-tracking

## Run

### OTB100 Dataset

```
$ python runner.py by_dataset  --vid-path=./data/freeman1/
```


