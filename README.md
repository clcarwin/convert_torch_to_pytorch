# Convert torch to pytorch
Convert torch t7 model to pytorch model and source.

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
