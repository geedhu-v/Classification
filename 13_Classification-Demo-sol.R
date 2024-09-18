  ##################################################
  ### PROG8430                                    ##
  ### Logistic Demonstration                      ## 
  ##################################################
  # Written by Peiyuan
  # ID: 123456789
  ##################################################
  ### Basic Set Up                                ##
  ##################################################
  # Clear plots
  if(!is.null(dev.list())) dev.off()
  # Clear console
  cat("\014") 
  # Clean workspace
  rm(list=ls())
  options(scipen=9)
  
  ##################################################
  ### Install Libraries                           ##
  ##################################################
  #For N-B Analysis
  if(!require(klaR)){install.packages("klaR")}
  library("klaR")
  
  #For DT
  if(!require(partykit)){install.packages("partykit")}
  library("partykit")
  
  #For NN
  if(!require(nnet)){install.packages("nnet")}
  library("nnet")
  
  if(!require(neuralnet)){install.packages("neuralnet")}
  library("neuralnet")
  
  #for stat.desc
  if(!require(pastecs)){install.packages("pastecs")}
  library("pastecs")
  
  #for Sampling
  if(!require(ROSE)){install.packages("ROSE")}
  library("ROSE")

  ##################################################
  ### Read data and do preliminary data checks    ##
  ##################################################
  # Read Blood Donation data set
  Blood <- read.csv("C:/Users/Geedhu/Documents/Maths _ Data Analysys/Week 13_Classification/BloodTransfusion.csv", header = TRUE, sep = ",")
  head(Blood,8)  #Print a Few Observations to Verify
  
  #Summary of Dataset
  str(Blood)
  summary(Blood)
  
  #Sampling!
  table(Blood$Donate)  # Output Class 0 has 570 and class 1 has 178 .hence Unbalanced data.
 
  Blood$Donate=as.factor(Blood$Donate)
  Blood <- ovun.sample(Donate~., data=Blood,N=356,method="under",seed=1)$data
  # Over sampling is done to add values to the class that has under data.
  # Under sampling is done to remove values of class that has more data, to make it balanced.
  #N=356, 178 for 0 and 178 for 1. 

  # To check if undersampling has affected the changes or not.
  table(Blood$Donate)
  ##################################################
  ### 1.Building a Stepwise Model                 ##
  ##################################################
  start_time_GLM <- Sys.time()
  GLM.mod <- glm(formula=Donate ~.,
                 family="binomial", data=Blood, na.action=na.omit)
  stepGLM.mod <- step(GLM.mod)
  end_time_GLM <- Sys.time()
  GLM_Time <- end_time_GLM - start_time_GLM
  summary(stepGLM.mod)
  GLM_Time
  
  ## Check the glm Model
  resp_glm <- predict(stepGLM.mod, type="response")   
  Class_glm <- ifelse(resp_glm > 0.5,"Y","N")           
  CF_GLM <- table(Blood$Donate, Class_glm, dnn=list("Act Donate","Predicted") ) 
  CF_GLM
  
  TP <- CF_GLM[2,2]
  TN <- CF_GLM[1,1]
  FP <- CF_GLM[1,2]
  FN <- CF_GLM[2,1]
  
  Precision = TP / (TP + FP)
  round(Precision,2)
  Sensitivity = TP / (TP + FN)
  round(Sensitivity,2)
  Specificity = TN / (TN + FP)
  round(Specificity,2)
  Accuracy = (TP + TN) / (TP + TN + FP + FN)
  round(Accuracy,2)
 
  ##################################################
  ### 2. NAIVE BAYES.                             ##
  ##################################################
  start_time_NB <- Sys.time()
  Blood_Naive <- NaiveBayes(Donate ~ . , data = Blood, na.action=na.omit)
  end_time_NB <- Sys.time()
  NB_Time <- end_time_NB - start_time_NB
  NB_Time

  pred_bay <- predict(Blood_Naive,Blood)
  #Creates Confusion Matrix
  CF_NB <- table(Actual=Blood$Donate, Predicted=pred_bay$class)
  CF_NB
  
  TP <- CF_NB[2,2]
  TN <- CF_NB[1,1]
  FP <- CF_NB[1,2]
  FN <- CF_NB[2,1]
  
  #Accuracy = (TP + TN) / (TP + TN + FP + FN)
  #Accuracy
  Precision = TP / (TP + FP)
  round(Precision,2)
  Sensitivity = TP / (TP + FN)
  round(Sensitivity,2)
  Specificity = TN / (TN + FP)
  round(Specificity,2)
  Accuracy = (TP + TN) / (TP + TN + FP + FN)
  round(Accuracy,2)
 
  ##################################################
  ### 3. Decision Trees                           ##
  ##################################################
  start_time_DT <- Sys.time()
  DT.mod <- ctree(Donate ~ ., data = Blood, na.action=na.omit)
  end_time_DT <- Sys.time()
  DT_Time <- end_time_DT - start_time_DT
  DT_Time
  
  DT.mod
  plot(DT.mod)

  
  pred_DT <- predict(DT.mod,newdata=Blood)
  #Creates Confusion Matrix
  CF_DT <- table(Actual=Blood$Donate, Predicted=pred_DT)
  CF_DT  
  
  TP <- CF_DT[2,2]
  TN <- CF_DT[1,1]
  FP <- CF_DT[1,2]
  FN <- CF_DT[2,1]
  
  #Accuracy = (TP + TN) / (TP + TN + FP + FN)
  #Accuracy
  Precision = TP / (TP + FP)
  round(Precision,2)
  Sensitivity = TP / (TP + FN)
  round(Sensitivity,2)
  Specificity = TN / (TN + FP)
  round(Specificity,2)
  Accuracy = (TP + TN) / (TP + TN + FP + FN)
  round(Accuracy,2)
  Balanced_Accuracy <- mean(Specificity,Sensitivity)
  round(Balanced_Accuracy,2)
  ##################################################
  ### 4. Neural Network Classification            ##
  ##################################################
  start_time_NN <- Sys.time()
  NN.mod = neuralnet(Donate~., data=Blood)
  end_time_NN <- Sys.time()
  plot(NN.mod,rep = "best")
  
  #NN.nnet.mod<- nnet(Donate ~ ., data = Blood, size = 10, maxit = 1000)
  
  NN_Time <- end_time_NN - start_time_NN
  NN_Time
  
  pred <- predict(NN.mod, newdata = Blood,linear.output = T)
  
  labels <- c("0", "1")
  max_col_pred <- max.col(pred)
  pred_NN <- data.frame(max_col_pred)
  pred_NN$pred <- labels[max_col_pred]
  pred_NN <- pred_NN$pred
  pred_NN <- unlist(pred_NN)
  pred_NN
  
  
  #Creates Confusion Matrix
  CF_NN <- table(Actual=Blood$Donate, Predicted=pred_NN)
  CF_NN  
  
  TP <- CF_NN[2,2]
  TN <- CF_NN[1,1]
  FP <- CF_NN[1,2]
  FN <- CF_NN[2,1]
  
  #Accuracy = (TP + TN) / (TP + TN + FP + FN)
  #Accuracy
  Precision = TP / (TP + FP)
  round(Precision,2)
  Sensitivity = TP / (TP + FN)
  round(Sensitivity,2)
  Specificity = TN / (TN + FP)
  round(Specificity,2)
  Accuracy = (TP + TN) / (TP + TN + FP + FN)
  round(Accuracy,2)
  
  
### Rec
  
  
  
  
  
  
  
  
  
  