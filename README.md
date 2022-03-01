# SOM

This is an example of Self-Organizing-Map applied to images for color quantization. 

Package dependences can be found in requirements.txt, use the following to install them
```
pip install -r requirements.txt
```

## Launch

In order to launch the application use this command

```
python main.py -i <image_path> -c <n_quantization_colors> -e <n_epochs> -b <batch_size>
```

This will save the quantized output image named "out.jpg".

![Alt text](images/parrot.jpg?raw=true "Title") ![Alt text](images/out.jpg?raw=true "Title")
