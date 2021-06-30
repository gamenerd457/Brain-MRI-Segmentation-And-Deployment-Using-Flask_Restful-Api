# Brain-MRI-Segmentation-And-Deployment-Using-Flask_Restful-Api
In this project I have trained a UNet model to predict masks for brain image segmentation and the model has been deployed using flask_restful api,so that we can get the output by sending a POST request

**brain_MRI.h5 is the trained model**

## Dataset:-
https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation <br />
This dataset contains brain MR images together with manual FLAIR abnormality segmentation masks. <br />
The images were obtained from The Cancer Imaging Archive (TCIA). <br />
They correspond to 110 patients included in The Cancer Genome Atlas (TCGA) lower-grade glioma collection with at least fluid-attenuated inversion recovery (FLAIR) sequence and genomic cluster data available. <br />

### Results:- 
![](https://github.com/gamenerd457/Brain-MRI-Segmentation-And-Deployment-Using-Flask_Restful-Api/blob/master/predictions.png)
![](https://github.com/gamenerd457/Brain-MRI-Segmentation-And-Deployment-Using-Flask_Restful-Api/blob/master/predictions2.png)

#### Running  inference using flask_restful_api :
* (imp) In infer.py file provide the full path to brain_MRI.h5 file
* start the app server using : python deploy.py <br />
![](https://github.com/gamenerd457/Brain-MRI-Segmentation-And-Deployment-Using-Flask_Restful-Api/blob/master/deploy_pic/pic2.png) <br />
* Run the inference using the given test_image.tif <br />
![](https://github.com/gamenerd457/Brain-MRI-Segmentation-And-Deployment-Using-Flask_Restful-Api/blob/master/deploy_pic/pic1.png)

##### Output from the api :
![](https://github.com/gamenerd457/Brain-MRI-Segmentation-And-Deployment-Using-Flask_Restful-Api/blob/master/deploy_pic/pic3.png)

