# ResNet-networks
This is the architecture of ResNet networks.
- The ResNet type of CNN was designed by `Microsoft Research` to compete in the international `ILSVRC` competition.
- The ResNet in the 2015 contest took first place in all categories for the ImageNet and Common Objects in Context `COCO`.
- The researchers for the residual block design pattern component of the residual network proposed a new novel layer connection they called and `identity link`.
_________________________________________________________________________________________________________________________________________________________

## ResNet34
### Architecure
- ResNet34 introduced a new block layer and layer-connection pattern, residual blocks, and identity connection respectively.
- The `residual block` in ResNet34 consists of blocks of two identical convolutional layers without a pooling layer. Each block has an `identity connection` that creates a parallel path between the input of the residual block and its output.

### The following code snippet shows how a residual block can be coded in TF.Keras by using the `functional API` method:
```

shortcut = x    # Remember the input to the block
x = layers.Conv2D(64, kernel_size=(3,3), strides=(1,1), padding="same")(x)
x = ReLU()(x)
x = layers.Conv2D(64, kernel_size=(3,3), strides=(1,1), padding="same")(x)
x = ReLU()(x)     # the output of the convolutional sequence
x = layers.add([shortcut, x])   # Matrix addition of the input to the output

```

![download](https://user-images.githubusercontent.com/59202700/207303627-1aed67c6-e4ad-4ca8-a639-20dc151670ab.png)

- The `ResNet` architecture take as input a (224, 224, 3) vector, an RGB image (3 channel) of 224 (height) X 224 (width) pixels. The first layer is a basic convolutional layer, consisting of a convolutional using a fairly large filter size of 7x7, the output (feature maps) is then reduced in size by a max pooling layer.
- Each successive group doubles the number of filters (similar to VGG). Unlike VGG, though, there is no pooling layer between the groups that would reduce the size of the feature maps. Now, if we connect these blocks directly with each other, we have a problem. The input to the next block has a shape based on the previous block's filter size (let's call it X). The next block, by doubling the filters, will cause the output of thet residual block to be double in size (let's call it 2X). The identity link would attempt to add the input matrix (X) and the output matrix (2X). Yikes - we get an error, indicating we can't broadcast (for the add operation) metrices of different sizes.

![3](https://user-images.githubusercontent.com/59202700/207815068-dddb43b3-ec95-4a99-8952-e579796f474e.png)

- For ResNet, this is solved by adding a convolutional block between each "doubling" group of residual blocks. As depicted in the figure above, the convolutional block doubles the filters to reshape the size and doubles the stride to reduce the feature map size by `75%` (performs feature pooling).
- The output of the last residual block group is passed to a pooling and flattening layer `GlobalAveragePooling2D`, which is then passed to a single Dense layer of 1000 nodes (number of classes).

________________________________________________________________________________________________________________________________________________________
________________________________________________________________________________________________________________________________________________________

## ResNet50
- RestNet50 introduced a variation of the residual block referred to as the `bottleneck residual block`.
- In this versionm the group of two 3x3 convolutional layers is replaced by a group of 1x1, then 3x3, and then 1x1 convolutional layers.
- The first 1x1 convolutional performs a dimensionality reduction, reducing the computational complexity, and the last convolution restors the dimensionality, increasing the number of filters by a factor of 4. The middle 3x3 convolution is referred to as `bottleneck` convolution, like the neck of a bottle.
- The bottleneck residual block, depicted in the figure below, allows for deeper neural networks, without degradation, and furthur reduction in computational complexity.
### Here is a code snippet for writing a bottleneck residual block as a reusable function:
```
def bottleneck_block(n_filters, x):
  """ Create a Bottleneck Residual Block of Convolutions
      n_filters: number of filters.
      x: input into the block.
  """
  shortcut = x
  x = layers.Conv2D(n_filters, (1,1), strides=(1,1), padding='same', activation='relu')(x)
  x = layers.Conv2D(n_filters, (3,3), strides=(1,1), padding='same', activation='relu')(x)
  x = layers.Conv2D(n_filters*4, (1,1), strides=(1,1), padding='same', activation='relu')(x)
  x = layers.add([shortcut, x])
  return x
  
``` 
![bottleneck](https://user-images.githubusercontent.com/59202700/208291809-9527d5d9-2d2d-4083-9c29-b4c14dac8729.png)
