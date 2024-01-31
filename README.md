# Criminal Recidivism Predictive Modeling

In this case, we use confusion matrices to understand a recent controversy around racial equality and criminal justice system. 
Use the logistic regression skills to develop and validate a model, analogous to the proprietary COMPAS model that caused the above-mentioned controversy. 

# Is COMPAS fair?

Background : 

Correctional Offender Management Profiling for Alternative Sanctions (COMPAS) algorithm is a commercial risk assessment tool that attempts to estimate a criminal defendent’s recidivism (when a criminal reoffends, i.e. commits another crime). COMPAS is reportedly one of the most widely used tools of its kind in the US. It is often used in the US criminal justice system to inform sentencing guidelines by judges, although specific rules and regulations vary. 

In 2016, ProPublica published an investigative report arguing that racial bias was evident in the COMPAS algorithm. ProPublica had constructed a dataset from Florida public records, and used logistic regression and confusion matrix in its analysis. COMPAS’s owners disputed this analysis, and other academics noted that for people with the same COMPAS score, but different races, the recidivism rates are effectively the same. 

Dataset 

The dataset here is based on ProPublica’s dataset, compiled from public records in Florida. However, it has been cleaned up for simplicity.

-c_charge_degree Classifier for an individual’s crime–F for felony, M for misdemeanor 1 
race Classifier for the recorded race of each individual in this dataset. We will only be looking at “Caucasian”, and “African-American” here. 

-age_cat Classifies individuals as under 25, between 25 and 45, and older than 45 sex Classifier for the recorded sex of each individual in this dataset. Male or female. priors_count Numeric, the number of previous crimes the individual has committed. 

-decile_score COMPAS classification of each individual’s risk of recidivism (1 = low . . . 10 = high). This is one of the crucial variables in the data, the number that the proprietary COMPAS algorithm assigns to all defendants. 

-two_year_recid Binary variable, 1 if the individual recidivated within 2 years, 0 otherwise. 


Load the COMPAS data, and perform a basic sanity checks.


```R
library("tidyverse")
```


```R
compas<- read.delim("compas-score-data.csv.bz2")
summary(compas)
any(is.na(compas))
```


          age        c_charge_degree        race             age_cat         
     Min.   :18.00   Length:6172        Length:6172        Length:6172       
     1st Qu.:25.00   Class :character   Class :character   Class :character  
     Median :31.00   Mode  :character   Mode  :character   Mode  :character  
     Mean   :34.53                                                           
     3rd Qu.:42.00                                                           
     Max.   :96.00                                                           
         sex             priors_count     decile_score    two_year_recid  
     Length:6172        Min.   : 0.000   Min.   : 1.000   Min.   :0.0000  
     Class :character   1st Qu.: 0.000   1st Qu.: 2.000   1st Qu.:0.0000  
     Mode  :character   Median : 1.000   Median : 4.000   Median :0.0000  
                        Mean   : 3.246   Mean   : 4.419   Mean   :0.4551  
                        3rd Qu.: 4.000   3rd Qu.: 7.000   3rd Qu.:1.0000  
                        Max.   :38.000   Max.   :10.000   Max.   :1.0000  



FALSE


Filter the data to keep only only Caucasians and African-Americans. 
Select only the "African-American", "Caucasian" category


```R
compas0 <- filter(compas, compas$race %in% c("African-American", "Caucasian"))
```

Create a new dummy variable based on the COMPAS’ risk score (decile_score), which indicates if an individual was classified as low risk (score 1-4) or high risk (score 5-10).


```R
compas0$risk <- cut(compas0$decile_score, 
                    breaks = c(1, 4, 10),  
                    labels = c("low risk", "high risk"), 
                    include.lowest = TRUE, 
                    right = TRUE)


```

Now analyze the offenders across this new risk category and found that  

We can found that the recidivism rate for low-risk individuals is 0.2264 while for high-risk individuals is 0.7195
The recidivism rate for Caucasian is 0.3635 while for African-American is 0.5276


```R
risk_recidivism_rates <- summarize(
  group_by(compas0, risk),
  mean_score = mean(decile_score, na.rm = TRUE) / 10
)
print(risk_recidivism_rates)

race_recidivism_rates <- summarize(
  group_by(compas0, race),
  mean_score = mean(decile_score, na.rm = TRUE) / 10
)
print(race_recidivism_rates)

```
A tibble: 2 × 2
risk      `mean(decile_score, na.rm = TRUE)/10`
<fct>                                     <dbl>
1 low risk                                  0.226
2 high risk                                 0.719


A tibble: 2 × 2
race             `mean(decile_score, na.rm = TRUE)/10`
<chr>                                            <dbl>
1 African-American                                 0.528
2 Caucasian                                        0.364


# Confusion Matrix for comparing COMPAS predictions and actual recidivism 

Here we create a confusion matrix comparing COMPAS predictions for recidivism (is/is not low risk) and the actual two-year recidivism and interpret the results. We found the accuracy, precision and recall below:

Accuracy=65.8% The proportion of correct predictions among all predictions is 65.8%

Precision=63.45% The proportion of true positive predictions among all positive predictions is 63.45%. In this context, it represents the accuracy of predicting recidivism among those classified as low risk.

Recall=64.52% The proportion of true positive predictions among all actual positives is 64.52%. In this context, it represents the ability to capture recidivists among all individuals who actually recidivated.


```R
conf_matrix <- table(Actual = compas0$two_year_recid, Predicted = compas0$risk)
print(conf_matrix)

true_positive <- conf_matrix[2, 2]  # Actual recidivist correctly classified as low risk
false_positive <- conf_matrix[1, 2]  # Non-recidivist incorrectly classified as low risk
true_negative <- conf_matrix[1, 1]  # Non-recidivist correctly classified as low risk
false_negative <- conf_matrix[2, 1]  # Actual recidivist incorrectly classified as low risk

compas_accuracy <- ((true_positive + true_negative) / sum(conf_matrix)) 
compas_accuracy
compas_precision <- ((true_positive) / (true_positive+false_positive))
compas_precision
compas_recall <- ((true_positive) / (false_negative+true_positive)) 
compas_recall
```

          Predicted
    Actual low risk high risk
         0     1872       923
         1      881      1602



0.658203865100417



0.634455445544554



0.645187273459525


We can note the accuracy of the COMPAS classification, and also how its errors were distributed. 
According to the results presented above, the actual low recidivism is predicted as high risk, while the actual high recidivism is predicted as low risk.
Therefore, I don't think this is an good model because there are too many flaws and may misunderstand the ones who are not recidivism to be risky, which is not fair.

# Confusion matrix analysis, separately for African-Americans and for Caucasians

Here we find the accuracy, FPR and FNR fir African-American and Caucasian
FPR = FP/N = FP/(FP + TN)
FNR = FN/P = FN/(FN + TP)
 (TP: True Positive, FP: False Positive, TN: True Negative, FN: False Negative.) 

"African-American" Accuracy=64.91%, FPR=42.34%, FNR=28.48%
"Caucasian" Accuracy=67.19%, FPR=22.01%, FNR=49.64%

The accuracy here is similar in both African American and Caucasian, the FPR is higher in African American while the FNR is higher in Caucasian. COMPAS’s true negative and true positive percentages are similar for African-American and Caucasian individuals, but that false positive rates and false negative rates are different. 
FPR is the proportion of actual negatives (non-events) that are incorrectly predicted as positive (events) by the model. The FPR in "African-American" is 42% that means it is often predicted incorrectly.
FNR is the proportion of actual positives (events) that are incorrectly predicted as negative (non-events) by the model. The FNR in "African-American" is 49% that means it is often predicted incorrectly.

Conclusion:
I think it's unfair, because it is predicted incorrectly for many cases, actual low risk be predicted as high risk while actual high risk predicted as low risks. Then we may misunderstand some nice people who really wants to may changes.Also, when doing research, we do not want to misunderstand anyone. I think in reality, there are differences between races, so it is normal to have those differences between races.


```R
#"African-American"
compas1 <- compas0 %>%
  filter(race %in% c("African-American"))

conf_matrix1 <- table(Actual = compas1$two_year_recid, Predicted = compas1$risk)
print(conf_matrix1)

true_positive1 <- conf_matrix1[2, 2] 
false_positive1 <- conf_matrix1[1, 2]  
true_negative1 <- conf_matrix1[1, 1]  
false_negative1 <- conf_matrix1[2, 1]  

compas_accuracy1 <- ((true_positive1 + true_negative1) / sum(conf_matrix1)) 
FPR1<-false_positive1/(false_positive1+true_negative1)
FNR1<-false_negative1/(false_negative1+true_positive1)
compas_accuracy1
FPR1
FNR1

#"Caucasian"
compas2 <- compas0 %>%
  filter(race %in% "Caucasian")

conf_matrix2 <- table(Actual = compas2$two_year_recid, Predicted = compas2$risk)
print(conf_matrix2)

true_positive2 <- conf_matrix2[2, 2] 
false_positive2 <- conf_matrix2[1, 2]  
true_negative2 <- conf_matrix2[1, 1]  
false_negative2 <- conf_matrix2[2, 1]  

compas_accuracy2 <- ((true_positive2 + true_negative2) / sum(conf_matrix2)) 
FPR2<-false_positive2/(false_positive2+true_negative2)
FNR2<-false_negative2/(false_negative2+true_positive2)
compas_accuracy2
FPR2
FNR2
```

          Predicted
    Actual low risk high risk
         0      873       641
         1      473      1188



0.649133858267717



0.42338177014531



0.28476821192053


          Predicted
    Actual low risk high risk
         0      999       282
         1      408       414



0.671897289586305



0.220140515222482



0.496350364963504



```R

```


```R

```
