# HW4 - Stereo Matching

The two input images after rectification:

<img src=./testdata/Cones/img_right.png width="400"> <img src=./testdata/Cones/img_left.png width="400">

Using ``Disparity Estimation`` for Stereo Matching:

<img src=./testdata/Cones.png width="400">

## Typical Improved Pipeline

It consists of 4 steps:

* Cost computation
* Cost aggregation
* Disparity optimization
* Disparity refinement

### Cost Computation

<img src=./testdata/steps/1.png width="800">

The code computes the matching cost between corresponding pixels in the left and right images using a census transform. It converts the images to binary representations and calculates the Hamming distance between them. Out-of-bound pixels are handled by assigning them the cost of the closest valid pixel.

### Cost Aggregation

<img src=./testdata/steps/2.png width="800">

The code performs cost aggregation by applying joint bilateral filtering to refine the cost volume. It creates cost volumes for different disparity values and applies the joint bilateral filter to each volume, considering both the left and right images.

### Disparity Optimization

<img src=./testdata/steps/3.png width="800">

The code determines the disparity for each pixel based on the estimated cost. It uses the Winner-Take-All approach to select the disparity with the minimum cost from the aggregated cost volume.

### Disparity Refinement

<img src=./testdata/steps/4.png width="800">

<img src=./testdata/steps/4-2.png width="800">

<img src=./testdata/steps/4-3.png width="800">

The code performs disparity refinement by applying left-right consistency checks, hole filling, and weighted median filtering. It checks for consistency between the left and right disparity maps, fills holes in the disparity map, and applies a weighted median filter to enhance the quality of the disparity estimation.

## Result

### Disparity Map

<img src=./testdata/Cones.png width="400"><img src=./testdata/Teddy.png width="400"><img src=./testdata/Tsukuba.png width="400"> <img src=./testdata/Venus.png width="400">


### The Bad Pixel Ratio

|Image      |Bad Pixel Ratio |
|------------|---------------|
|Cones       |  7.85%        |
|Teddy       |  9.39%        |
|Tsukuba     |  4.41%        |
|Venus       |  1.52%        |