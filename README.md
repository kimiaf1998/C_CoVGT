# Spatio Video Grounding via Graph Transformer

## Background and Motivation
This work extends an existing video QA system, <a href="https://arxiv.org/abs/2302.13668">CoVGT</a>, by incorporating a space-time decoder specifically tailored for spatial localization. CoVGT adopts a graph-based representation for video elements, treating them as nodes and edges to capture dynamic interactions among objects for effective video reasoning. The use of graph transformers on nodes and edges enables the model to derive informative temporal relations across different timestamps. Their innovative approach to video question answering using graph neural networks motivated us to embark on this project.


** Note: The space-time decoder is inspired by <a href="https://arxiv.org/abs/2203.16434">TubeDETR</a>.

</br>
Our contributions are the following:  

* Manipulating STAR dataset annotations to present a single location as the visual gt answer.
* Parsing STAR dataset to filter out questions containing non-object answers given the questions template.
* Adding a space-time decoder to model spatial interaction over the entire video.

**For detailed information, please refer to the project <a href="final_report.pdf">report</a>.** 

| ![scr1](./misc/C_CoVGT_model_arch.png "Overview of the Proposed Model Architecture") | ![scr2](./misc/C_CoVGT_space_time_arch.png "Space-Time Decoder Architecture") |
| ------------ | ------------ |
| Proposed Model Architecture | Space-Time Decoder Architecture |

## Setup
Assume you have installed Anaconda3, cuda version > 11.0 with gpu memory >= 24G, please do the following to setup the env:
```
>conda create -n vqa python==3.8.18
>conda activate vqa
>git clone https://github.com/kimiaf1998/C_CoVGT.git
>conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
>pip install -r requirements.txt
```
## Dataset 

### Annotations
Please download the videos from this <a href="https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip">link</a>.
New annotations are stored in ```/datasets/STAR/``` directory. ```train.json``` and ```val.json``` are the finalized annotations for training and validation. Since the STAR dataset doesn't provide a test set, we use the validation set as the test set. ```clips_train.json``` and ```clips_val.json``` show the mapping of the sampled frames for each data point (question corresponding to a video). These sampled frames ensure that we capture every frame containing the visual answer (objects bboxes) for every question.
```vid_fps_mapping.json``` also indicates the mapping of fps to each video.


### Features
Please download the pre-extracted video features from <a href="https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip">link</a>. The frame features are stored in ```frame_feat``` and regional features in ```bbox``` folders. The pre-extracted regional features hold the top-10 confident features. To extract frame-wise features on your own, use ```tools/preprocess_feature.py``` and for regional features, use the provided tool in <a href="https://github.com/MILVLG/bottom-up-attention.pytorch">BUA<a>.


## Demo

Run the following command to infer and visualize results on any video of the STAR dataset. You may want to change the video information. Download and put the <a href="https://drive.google.com/file/d/1GqYjnad42-fri1lxSmT0vFWwYez6_iOv/view?usp=sharing)">checkpoint</a> for the best model in ```/data/save_models/STAR/CoVGT```.
```
./shells/star_demo.sh 0
```

## Train
To train the model, use the train script provided in the folder ```shells``` and run it by specifying the GPU IDs behind the script. (If you have multiple GPUs, you can separate them with a comma: ./shell/star_train.sh 0,1)
```
./shell/star_train.sh 0
```
It will train the model and save it to the folder 'workspace/save_models/STAR/CoVGT/' by default. 

Set some useful args:
* ```--bnum``` to the number of object queries.
* ```--mc``` to the number of multiple-choices.
* ```--qmax_words``` and ```--amax_words``` to the number of maximum question and answer tokens, respectively.
* ```--video_max_len``` to the number of frames to sample from each video.
* ```--bbox_loss_coef```, ```--giou_loss_coef``` and ```--cl_loss``` to the appropriate weight for each loss contribution. 


## Results
**<p align="center">Table 1. Quantitative results on STAR dataset.</p>**
| Model | parameters | Interaction (vIoU/vIoU@0.3/vIoU@0.5) | Sequence (vIoU/vIoU@0.3/vIoU@0.5) | Prediction (vIoU/vIoU@0.3/vIoU@0.5) | Feasibility (vIoU/vIoU@0.3/vIoU@0.5) | qa All Acc. |
|-----|-----|-----|-----|-----|-----|-----|
| [CoVGT](https://arxiv.org/abs/2302.13668)| bnum=10 | - | - | - | - | 46.20 |
| Ours | bnum=10 | 17.8/25.8/19.1 | 16.5/28.6/17.9 | 12.3/21.5/16.6 | 15.9/23.1/21.4 | 44.59 |


<div align="center">
  <b>Table 2. Qualitative examples result on STAR dataset.</b>
  <img width="80%" alt="C_CoVGT Qualitative Results" src="./misc/C_CoVGT_qualitative_results.png">
</div>

## Citations 
```
@misc{kimia2023vqa,
      title={Spatio Video Grounding via Graph Transformer}, 
      authors={Kimia Afshari, Hannah Hagen},
      year={2023}
}
```
## Notes
If you use any resources from this repo, please kindly star our repo and acknowledge the source.
