<div align="center">
  <h1>Mask Image Classification</h1>
</div>

![banner](https://user-images.githubusercontent.com/68208055/157444787-8e661ec2-369f-4fcd-bd54-6b1693ea20ca.png)


## :mask: Overview
### Background
COVID-19의 확산으로 우리나라는 물론 전 세계 사람들은 경제적, 생산적인 활동에 많은 제약을 가지게 되었습니다. 우리나라는 COVID-19 확산 방지를 위해 사회적 거리 두기를 단계적으로 시행하는 등의 많은 노력을 하고 있습니다. 하지만 넓은 공공장소에서 모든 사람들의 올바른 마스크 착용 상태를 검사하기 위해서는 추가적인 인적자원이 필요할 것입니다.

따라서, 우리는 `카메라로 비춰진 사람 얼굴 이미지 만으로 이 사람이 마스크를 쓰고 있는지, 쓰지 않았는지, 정확히 쓴 것이 맞는지 자동으로 가려낼 수 있는 시스템`이 필요합니다. 이 시스템이 공공장소 입구에 갖춰져 있다면 적은 인적자원으로도 충분히 검사가 가능할 것입니다

### Problem definition
`카메라로 비춰진 사람 얼굴 이미지만으로 이 사람이 마스크를 쓰고 있는지, 쓰지 않았는지, 정확히 쓴 것이 맞는지 자동으로 가려낼 수 있는 모델`

### Evaluation
- F1 Score

<br>

## :star: Result  

![스크린샷 2022-03-09 오후 10 12 40](https://user-images.githubusercontent.com/68208055/157448470-d6902398-32f8-4ad9-815b-826138f7fed4.png)

- F1 score : 0.7584
- Accuracy : 80.6190

<br>

## :mag: Contents

```
├── data/
|   ├── image/
|   |   ├── train/ 
|   |   └── eval/ 
├── output/
├── train.py
├── config.py
├── imbalance.py
├── Utils.py
├── models.py
├── dataset.py
├── loss.py
├── inference.py
```

### Training
```
SM_CHANNEL_TRAIN=[train image dir] SM_MODEL_DIR=[model saving dir] python train.py
```

### Inference
```
SM_CHANNEL_EVAL=[eval image dir] SM_CHANNEL_MODEL=[model saved dir] SM_OUTPUT_DATA_DIR=[inference output dir] python inference.py
```



<br>

## :family: Dataset
### Data

- 전체 사람 수 : 4500명 (train : 2700 | eval : 1800)
- 한 사람당 사진의 개수 : 7 [마스크 5장, 이상하게 착용(코스크, 턱스크...) 1장, 미착용 1장]
- 전체 이미지 수 : 31500장 (train : 18900 | eval : 12600)
- 나이 : 20대 - 70대
- 성별 : 남,여
- 이미지 크기 : (384,512)

### Data Labeling
- mask, gender, age 기준 18개의 클래스로 분류
<img src="https://user-images.githubusercontent.com/68593821/131881060-c6d16a84-1138-4a28-b273-418ea487548d.png" height="500"/>


