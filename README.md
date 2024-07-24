## <div align="center">RFENet</div>
### <div align="left">Our RFENet research is based on YOLOv5.</div>

<details open>
<summary>Install</summary>

Clone repo and install [requirements.txt](https://github.com/ultralytics/RFENet/blob/master/requirements.txt) in a
[**Python>=3.7.0**](https://www.python.org/) environment, including
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/).

```bash
git clone https://github.com/object9detection/RFENet.git  # clone
cd RFENet
pip install -r requirements.txt  # install
```
</details>

<details>
<summary>Training</summary>

```bash
python train.py --data coco.yaml --cfg yolov5s_All.yaml --weights '' --batch-size 64 
```

</details>


<details>
<summary>Val</summary>

```bash
python val.py --data coco.yaml --weights '' --batch-size 64 
```

</details>

