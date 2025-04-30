# Certified Visual Odometry Julia

This repo contains the julia source code to accompany the paper
```
@article{agrawal2024online,
  title={Online and certifiably correct visual odometry and mapping},
  author={Agrawal, Devansh R and Govindjee, Rajiv and Yu, Jiangbo and Ravikumar, Anurekha and Panagou, Dimitra},
  journal={arXiv preprint arXiv:2402.05254},
  year={2024}
}
```

We have released the julia source code for estimating the rototranslation and the error bounds between two point clouds. 
In the future, we hope to release the code to run the full visual odometry routine, taking in images, and returning the pose estimate and error bounds. 

## Quickstart

Clone this repo, and make sure the following (currently unregistered) packages are also installed:

```julia
] add https://github.com/dev10110/ParallelMaximumClique.jl
] add https://github.com/dev10110/GraduatedNonConvexity.jl
```

Now run 
```julia
include("test/test.jl")
```
to run the test script. 
This will generate a pointcloud, a rototranslated version of the point cloud, add some noise and outliers, and then run the estimation and error bounding calculations. 
See the `test/test.jl` to see how to use the methods presented here. 