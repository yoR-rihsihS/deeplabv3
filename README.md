# deeplabv3
PyTroch Implementation of DeepLabV3.


## Generating Masks from Annotations :
- Download "gtFine_trainvaltest.zip" and "leftImg8bit_trainvaltest.zip" from the [official cityscapes website](https://www.cityscapes-dataset.com/downloads/) and extract them in "./data".
- Run [generate_gt.py](./cityscapes/generate_gt.py), it will generate masks (trainId mask, labelId mask, color mask) as per the conventions of CityScapes Dataset and save them in appropriate directory.
- Comparing the generated mask with provided masks:
![Comparing the generated mask with provided masks](./outputs/gen%20city%20acc.png)


## Dataset
Make sure the project folder looks like this:
```
Project/
├── cityscapes/
│   └── ... (python scripts for data loader and utils)
├── data/
│   ├── generated_gt
│   ├── leftImg8bit
│   ├── gtFine
│   └── ... (other files from dataset)
├── deeplabv3/
│   └── ... (python scripts for deeplabv3 model and utils)
├── outputs/
│   └── ... (output directory)
├── saved/
│   └── ... (save directory during training)
├── save_predictions.ipynb
├── test_dlv3.py
├── train_dlv3.py
├── visualize_training.ipynb
└── ... (other files from project)
```


## Training
- Run the following command to train DeepLabV3 model:
```
python train_dlv3.py --backbone "{resnet50/resnet101/resnet152}" --output_stride "{8/16}"
```

- Training Metrics:
<table style="width: 100%;">
  <tr>
    <td><img src="./outputs/dlv3_loss.png" style="width: 100%;"/></td>
    <td><img src="./outputs/dlv3_miou.png" style="width: 100%;"/></td>
  </tr>
  <tr>
    <td><img src="./outputs/dlv3_mdice.png" style="width: 100%;"/></td>
    <td><img src="./outputs/dlv3_mpacc.png" style="width: 100%;"/></td>
  </tr>
  <tr>
    <td><img src="./outputs/train os16.png" style="width: 100%;"/></td>
    <td><img src="./outputs/train os8.png" style="width: 100%;"/></td>
  </tr>
  <tr>
    <td><img src="./outputs/sum os16.png" style="width: 100%;"/></td>
    <td><img src="./outputs/sum os8.png" style="width: 100%;"/></td>
  </tr>
</table>

Comment: The above plots are generated using [visualize_training.ipynb](visualize_training.ipynb)


## Performance
- Run the following command to evaluate DeepLabV3 model on test sets:
```
python test_dlv3.py  --backbone "{resnet50/resnet101/resnet152}" --output_stride "{8/16}" --model_weights_path "./saved/{file_name}.pth"
```
- DeepLabV3 Resnet-50 Output Stride 8 Model evaluation on test set from official evaluation server:

<table style="width: 100%;">
  <tr>
    <td><img src="./outputs/avg os8.png" style="width: 100%;"/></td>
    <td rowspan="2"><img src="./outputs/class os8.png" style="width: 100%;"/></td>
  </tr>
  <tr>
    <td><img src="./outputs/cat os8.png" style="width: 100%;"/></td>
  </tr>
  <tr>
    <td><img src="./outputs/inf os16.png" style="width: 100%;"/></td>
    <td><img src="./outputs/inf os8.png" style="width: 100%;"/></td>
  </tr>
</table>

Comment: The code saves all the predictions in the directory [outputs/{model_name}](./outputs/) which can be zipped and uploaded to the official evaluation server to get the model performance evaluated.


## Predictions 
- DeepLabV3 Resnet-50 Output Stride 8 Model Predictions: 
![DeepLabV3 Resnet-50 Output Stride 8 Model Predictions](./outputs/predictions_os_8.gif)

- DeepLabV3 Resnet-50 Output Stride 16 Model Predictions:
![DeepLabV3 Resnet-50 Output Stride 16 Model Predictions](./outputs/predictions_os_16.gif)


## Model Details
- Memory Requirements of Models:
![Memory Requirements of Models](./outputs/mem.png)

Comment:
- DeepLabV3 Resnet-50 Output Stride 8 Model requires 17.5 GB VRAM for inference on complete image of size 1024 x 2048.
- DeepLabV3 Resnet-50 Output Stride 16 Model requires 9.0 GB VRAM for inference on complete image of size 1024 x 2048.