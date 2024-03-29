# A Review of Deep Learning Methods for Irregularly Sampled Medical Time Series Data

Chenxi Sun, Shenda Hong, Moxian Song, and Hongyan Li. 
https://arxiv.org/abs/2010.12493

## Irregularly Sampled Medical Time Series Datasets
| Name  | Folder  | 
| :---| :---- | 
|Sepsis early diagnosis| /data/sepsis_data|
|COVID-19 mortality classification| /data/covid19_data|
|MIMIC-III in-hospital mortality classification| /data/mimiciii_inhospital_mortality_data|
|ICU mortality data| /data/icu_mortality_data|
|Synthetic patient data| /data/emrbots_data|


## Other Time Series Datasets
(Can be regular or irregular data)

| Name  | Folder  | Link
| :---| :---- | :---- |
|SILSO solor dataset| /other_data/solor_data | https://lasp.colorado.edu/lisird/data/international_sunspot_number   | 
|USHCN weather dataset| /other_data/usa_weather_data | https://www.ncei.noaa.gov/products/land-based-station/us-historical-climatology-network | 
|Synthetic medical records| /data/emrbots_data| http://www.emrbots.org   | 
|Chaotic system datasets| /other_data/chaotic_system_data |  | 
|UCR comprehensive datasets| / | https://www.cs.ucr.edu/~eamonn/time_series_data_2018 | 
|UEA comprehensive datasets| / | http://www.timeseriesclassification.com/index.php | 
|Forecasting comprehensive datasets| / | https://github.com/thuml/Time-Series-Library  | 
|MIT-BIH arrhythmia ECG dataset| / |  https://physionet.org/content/mitdb/1.0.0/  | 
|MIT-BIH ST change ECG dataset| / |  https://physionet.org/content/stdb/1.0.0/  | 
|European ST-T ECG dataset| / | https://physionet.org/content/edb/1.0.0/ | 
|Sudden cardiac death holter ECG dataset| / |  https://physionet.org/content/sddb/1.0.0/ | 




## Related Works

| ID  | Year  | Title                                                                                                                    | Method   | Venue            |
| :---| :---- | :-----------------------------------------------------------------------------------------------------------------       | :--------- | :--------------- |
| 1   | 2016  | Directly modeling missing data in sequences with rnns: Improved classification of clinical time series                   | RNN        | MLH              |
| 2   | 2016  | Automatic classification of irregularly sampled time series with unequal lengths: A case study on estimated glomerular filtration rate| /        | MLSP             |
| 3   | 2017  | Patient subtyping via time-aware lstm networks                                                                           | T-LSTM     | KDD              |
| 4   | 2017  | Multivariate time series imputation with generative adversarial networks                                                 | RNN        | BIGCOMP          |
| 5   | 2017  | A bio-statistical mining approach for classifying multivariate clinical time series data observed at irregular intervals | RNN        | EXPERT SYST APPL          |
| 6   | 2018  | Recurrent neural networks for multivariate time series with missing values                                               | GRU-D      | Sentific Reports |
| 7   | 2018  | Bidirectional recurrent imputation for time series                                                                       | Brits      | NIPS             |
| 8   | 2018  | Hierarchical deep generative models for multi-rate multivariate time series                                              | MR-HDMM    | ICML             |
| 9   | 2018  | Multivariate time series imputation with generative adversarial networks                                                 | GAN        | NIPS             |
| 10  | 2018  | missing data imputation using generative adversarial nets                                                                | GAIN       | ICML             |
| 11  | 2018  | Medical missing data imputation by stackelberg gan                                                                       | stackelberg| ICML             |
| 12  | 2019  | Interpolation-prediction networks for irregularly sampled time series                                                    | IPN        | ICLR             |
| 13  | 2019  | Temporal-clustering invariance in irregular healthcare time series                                                       | cluster    | /                |
| 14  | 2019  | Learning from incomplete data with generative adversarial networks                                                       | misGAN     | ICLR             |
| 15  | 2019  | Multi-resolution networks for flexible irregular time series modeling                                                    | multi-fit  | /                |
| 16  | 2019  |A comparison between discrete and continuous time bayesian networks in learning from clinical time series data with irregularity| BN         | /              |
| 17  | 2020  | Joint modeling of local and global temporal dynamics for multivariate time series forecasting with missing values        | LGnet      | AAAI             |
| 18  | 2020  | Dual-attention time-aware gated recurrent unit for irregu- lar multivariate time series                                  | DATA-GRU   | AAAI             |
| 19  | 2020  | A survey of missing data imputation using generative adversarial networks                                                | GAN        | ICAIIC           |
| 20  | 2020  | Medical time-series data generation using generative adversarial networks                                                | GAN        | AIM              |
| 21  | 2020  | Kernels for time series with irregularly-spaced multivariate observations                                                | Kernel     | /               |
| 22  | 2020  | Explainable uncertainty-aware convolutional recurrent neural network for irregular medical time series                   | TNNLS      | /               |

## Category of related Works   
  
![Image](https://github.com/scxhhh/ISMTS-Review/blob/main/figures/category.png)  
  
    
    
## Conclusion of related Works    
     
![Image](https://github.com/scxhhh/ISMTS-Review/blob/main/figures/related_works.png)  
  
    
## Sturcture of EHRs
  
![Image](https://github.com/scxhhh/ISMTS-Review/blob/main/figures/EHR.png)  
  
     
## Missing rates of the real-world EHRs datasets  
  
![Image](https://github.com/scxhhh/ISMTS-Review/blob/main/figures/missing_rate.png)  
  
    

## Sturctures of RNN-based methods  
  
![Image](https://github.com/scxhhh/ISMTS-Review/blob/main/figures/method_structures.png) 


