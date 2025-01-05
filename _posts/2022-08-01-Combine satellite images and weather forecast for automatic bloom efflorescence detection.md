---
title: Combine convolutional neural network, satellite imagery and weather forecast for automated blooming risk prediction 
date: 2022-08-01 00:00:00 +0000
categories: [projects]
tags: [weather forecast,bloom prediction,Auto-Encoder,Convolutional Neural Network]
image: 
    path: '/assets/project_bepsia/figures/simpleCNN_cropped.png'
    alt: Figure (1) - State of the art Convolutional Neural Network architecture for automatic bloom efflorescence prediction with satellite images.
---

---
## Information
Brief presentation of the BEPS-IA project by Scalian R&D and WaterShed Monitoring, the code is not available.

---


---
## Abstract
<div style="text-align: justify">
Cyanobacteria have the potential to develop toxins that can have a number of adverse effects on human and animal health, as well as being capable of accumulating in the aquatic food chain, so that even eating fish in affected bodies of water is a potential health hazard. As a result, swimming, or secondary contact sports such as kayaking, surfing, etc. can also present a health risk. In 2021 , the exploratory work supported by European Space Agency (ESA), SCALIAN R&D, a French engineering consultancy specialising in the development of artificial intelligence-based solutions, and WaterShed Monitoring Europe, a French micro-company specialising in the collection, management and exploitation of water quality data, revealed the potential of predicting blooming events using infrared photographs taken by satellite, under the name BEPS-IA (Bloom Event Prediction by Satellite Images Analysis). This blog presents the continuity of this project in which weather forecast data are explored to increase the blooming event prediciton accuracy.

</div>
---


---
## Data Resume
<div style="text-align: justify">
The province of Alberta (Canada) offers an open data service that provides a database of cyanobacteria bloom events in recreational lakes. As this data is not acquired through a direct partnership with the province of Alberta, only an access to bloom events and satellite images of Alberta lakes is publicly available. This dataset represent 111 lakes (see Figure (2)). Moreover, blooming event from the Lake Saint-CHarles of the Quebec province are available and added to this dataset. The satellite images come from the Landsat-8 and Sentinel-2 collections. The sensors used to obtain these images are of different types. After discussion with the experts at Watershed Monitoring, the choice of associating images preceding blooming event within 3 days with the "BLOOM" label, and the other images with the "NO BLOOM" label has been made. The bloom label is interpreted as a risk and not an obervation of blooming event. A preprocessing atmospheric correction and standardisation are applied to both satellite image collections. Moreover, an harmonasation preprocess is applied to Landsat-8 image to mimic a Sentinel-2 image using a pixel-by-pixel and band-by-band transformation. To ensure that the data we send to the convolutional network is always the same size, we resample the satellite images by bands so that they are all 100×100 using bilinear resampling. The dataset represents a total of 1578 satellite images covering the period of 2013-2021, and including 357 blooming events.

<figure>
  <img
  src="/assets/project_bepsia/figures/dataresults.png"
  alt="Presentation of the dataset composed of staellite images and weather maps."
  >
  <em>Figure (2) - Presentation of the dataset composed of staellite images and weather maps.</em>
</figure>

The weather map has been collected from the canadian meteorological office using the weather forecasting model named Regional Deterministc Prediciton System (RDPS). According to the litterature and the experts, the choice has been made of investigating the effect of three surface weather variables: direct solar downard flux (FSD), temperature (TMP), the wind speed (UVC) and the wind direction (WVC). The main objective being of the blooming event prediction, the model inputs need to describe the period preceeding the bloom event. Each of the selected weather variable are taken from 0 to 4 days of forecasting range. The weather map is a square of 16x16 forecaseting grid-points spaced of 10km representing an area of 160 km (c.f. Figure (2)). A total of 97538 weather maps covering the 111 lakes for the period of 2013-2021 are availables.
<figure>
  <img
  src="/assets/project_bepsia/figures/datasummary.png"
  alt="Key numbers of the dataset used for the 3 days blooming event prediction."
  >
  <em>Figure (3) - Key numbers of the dataset used for the 3 days blooming event prediction.</em>
</figure>

In the Figure (3), the element named "Cloud filter" refers to a preprocessing technics applied to the satellite images removing any image with a percentage of cloud cover superior to 50%.
</div>
---


---
## Neural Network architectures
In this project, two main neural network architecture has been explored to predict 3 days blooming event with satellite images and forecated weather maps: Late fusion convolutional neural network (Double CNN) and auto-encoder convolutional neural network (AE & CNN).
### Double CNN
<div style="text-align: justify">
The first architecture Double CNN described in the Figure (4) is inspired by the duplication of convolutional layers and applied independently to each dataset source. Then, the results of the last dense layers would be concatenated performing a late fusion of the information before delivering the final prediction. The advantage of this method is that it required less parameters than a 3D convolution and it cans focus on spatial caracteristic from each source before merging the results. The disadvantage of this architecture is that we learn the weights associated with the meteorological fields according to the cyanobacteria bloom prediction problem. The spatial domain is relatively large for a local phenomenon with little training data, which could lead to over-learning.
<figure>
  <img
  src="/assets/project_bepsia/figures/simpleCNN_fusionnate.PNG"
  alt="Double CNN architecture."
  >
  <em>Figure (4) - Double CNN architecture.</em>
</figure>
</div>
---


---
### AE & CNN
<div style="text-align: justify">
The next idea presented in the Figure (5) is to replace the convolutional neural network with an auto-encoder seeking to produce a latent space summarising each of the multiphysical spatial fields. This latent space would then be concatenated with the last layer of the network seeking to predict flowering from multispectral satellite images. This approach makes it possible to infer a greater quantity of meteorological information by learning the optimal transformation to reduce the meteorological fields to a simple vector. Furthermore, in this architecture, the auto-encoder is infered independently to learn the all meteorological domain instead of the Double CNN which is only trained on the dates corresponding to the satellite images, which limits the learning base enormously (830 usable satellite images compared with 97,000 meteorological fields, i.e. a factor of 100 between the two).

<figure>
  <img
  src="/assets/project_bepsia/figures/AECNN_fusionnate.PNG"
  alt="AE & CNN architecture."
  >
  <em>Figure (5) - AE & CNN architecture.</em>
</figure>
</div>
---


---
## Results
<div style="text-align: justify">
The previous models, Double CNN and AE & CNN , are now evaluated using the test dataset consisting of the period 2020-2021 removed from the global datasets. In addition, both models are evaluated by comparing their classification performance with a simple CNN (CNN Sat) trained to predict the 3-day blooming event using only the satellite images and with another CNN trained on the weather map (CNN WM).

<br>
<br>

Table (1) shows the classification metrics of recall, precision and accuracy. These metrics are derived from the confusion matrix comparing the actual blooming events with the model prediction. Recall, Precision and Accuracy close to one reflect good classification performance of the models. Accuracy provides a global perspective on the accuracy rate of the model predictions. In binary classification, recall is equivalent to the probability of detecting the event of one, while precision is the rate of correctly retrieving the class of zero. 

<br>
<br>

With the application of the Cloud filter preprocess, the CNN Sat model shows the best performances with Recall=0.99, Precision=0.63 and Accuracy=0.69 where the second best performance are allocated to the AE&CNN model with Recall=0.92, Precision=0.6 and Accuracy=0.65. Both models show a overfitting of the blooming event.When the cloud filter is removed, model performance from the simple CNN Satellite model collapse while the classification performances of the AE & CNN increases. The features extracted from the weatehr map helps the CNN prediction from the satellite images to maintain a high level of blooming event prediction.

<br>
<br>
<figure>
  <img
  src="/assets/project_bepsia/figures/latentspace.png"
  alt="2D representation of the latent spaces of the meteorological field sets of the training and validation datasets used by the CNN for predicting cyanobacterial blooms."
  >
  <em>Figure (6) - 2D representation of the latent spaces of the meteorological field sets of the training and validation datasets used by the CNN for predicting cyanobacterial blooms.</em>
</figure>
However, overlearning of blooming cases is still present in this model. Figure (6) shows the latent vectors summarised in 2D space and the datasets used by the network to predict blooms. The left part of the images shows groupings of individuals following a majority of blue dots without cyanobacterial blooms, while the middle and right parts of the figure show many local clusters of orange dots with blooming events. As can be seen in Figure (6), despite the presence of a structure that discretises the cases according to the weather, there are still a large number of individuals without efflorescence that do not belong to any structure. These individuals are diluted around the small local structures formed by the blooming situations, thus increasing the risk of false alarms.

<figure>
  <img
  src="/assets/project_bepsia/figures/dataresults2.png"
  alt="Comparison of the model's classification performances."
  >
  <em>Table (1) - Comparison of the model's classification performances.</em>
</figure>
</div>
---

---
## Discussion
<div style="text-align: justify">
The results obtained by these two approaches revealed a high degree of overlearning of bloom situations compared with the results of the BEPS-IA 2021 project. The large size of the descriptors compared with the small number of individuals means that the networks struggle to infer the data correctly. When adding new data by removing the cloud filter and also by reducing the dimension of the descriptors by removing the temporal dynamics of the meteorological fields, the results of these two approaches are clearly improved, outperforming the reference model using only satellite data. However, overlearning is still present. This undesirable effect demonstrates the complexity of the objective of predicting blooms from large-scale meteorological fields. To overcome this issue, another dataset will be add to this project: 
 <a href="https://oceandata.sci.gsfc.nasa.gov/api/cyan_file_search"> Cyanobacteria Assessment Network (CyAN)</a>. This available dataset allows citizens and policymakers to get near-real time updates on the cyanobacteria in over 2,300 lakes in the contiguous United States and more than 5,000 in Alaska. The new study, published in the journal Remote Sensing of Environment, introduces this extensive inland waters dataset that includes a time series of standardized satellite measurements starting in 2002.
 </div>
---

