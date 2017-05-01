# Convert torch to pytorch
Convert torch t7 model to pytorch model and source including cuDNN supported RNNs.

## Convert
```bash
python convert_torch.py -m vgg16.t7
```
Two file will be created ```vgg16.py``` ```vgg16.pth```

## Example
```python
import vgg16

model = vgg16.vgg16
model.load_state_dict(torch.load('vgg16.pth'))
model.eval()
...
```

## Convert models including cuDNN RNN modules

```bash
th process_rnns.lua -model rnn_model.t7 -output rnn_model_processed.t7
python convert_torch -m rnn_model_processed.t7
```

## Validated
All the models in this table can be converted and the results have been validated.

| Network             | Download |
| ------------------- | -------- |
| AlexNet | [cnn-benchmarks](https://github.com/jcjohnson/cnn-benchmarks) |
| Inception-V1 | [cnn-benchmarks](https://github.com/jcjohnson/cnn-benchmarks) |
| VGG-16 | [cnn-benchmarks](https://github.com/jcjohnson/cnn-benchmarks) |
| VGG-19 | [cnn-benchmarks](https://github.com/jcjohnson/cnn-benchmarks) |
| ResNet-18 | [cnn-benchmarks](https://github.com/jcjohnson/cnn-benchmarks) |
| ResNet-200 | [cnn-benchmarks](https://github.com/jcjohnson/cnn-benchmarks) |
| ResNeXt-50 (32x4d) | [ResNeXt](https://github.com/facebookresearch/ResNeXt) |
| ResNeXt-101 (32x4d) | [ResNeXt](https://github.com/facebookresearch/ResNeXt) |
| ResNeXt-101 (64x4d) | [ResNeXt](https://github.com/facebookresearch/ResNeXt) |
