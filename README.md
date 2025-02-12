<div align="center">
        <h1>Faster RCNN</h1>
            <p>Simple faster-RCNN implementation in Pytorch</p>
            <p>
            <a href="https://github.com/VuThanhDat14122004/Faster-RCNN/graphs/contributors">
                <img src="https://img.shields.io/github/contributors/VuThanhDat14122004/Faster-RCNN" alt="Contributors" />
            </a>
            <a href="">
                <img src="https://img.shields.io/github/last-commit/VuThanhDat14122004/Faster-RCNN" alt="last update" />
            <a href="https://github.com/VuThanhDat14122004/Faster-RCNN/network/members">
		        <img src="https://img.shields.io/github/forks/VuThanhDat14122004/Faster-RCNN" alt="forks" />
	        </a>
	        <a href="https://github.com/VuThanhDat14122004/Faster-RCNN/stargazers">
		        <img src="https://img.shields.io/github/stars/VuThanhDat14122004/Faster-RCNN" alt="stars" />
	        </a>
</div>

# Description
I have studied and re-implemented the Faster R-CNN object detection network using PyTorch. In this implementation, I have tried to clearly specify the input and output dimensions of functions and classes to enhance readability.

# Dataset
I use the <a href="https://www.kaggle.com/datasets/zaraks/pascal-voc-2007">VOC2007 Dataset</a>

# Parameters
- CNN backbone: Use the first 17 conv layer of VGG16(pretrained weights)
- Batch:(dynamic) Using all region proposals of an image to create a batch
- Optimizer: Adam with learning rate is 1e-3

# Result
- I'm still trainning ...

# References
Paper: <a href="https://arxiv.org/pdf/1506.01497">
        paper
       </a>

Blog: <a href="https://medium.com/towards-data-science/understanding-and-implementing-faster-r-cnn-a-step-by-step-guide-11acfff216b0">
        blog1
       </a>,
       <a href="https://medium.com/@fractal.ai/guide-to-build-faster-rcnn-in-pytorch-42d47cb0ecd3">
        blog2
       </a>

Video: <a href="https://www.youtube.com/watch?v=4yOcsWg-7g8">
        video
       </a>

Here are the documents, repositories, and videos I referenced for the implementation.