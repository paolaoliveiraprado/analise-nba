library(kernlab)
library(caret)
library(gbm)

base = read.csv("https://query.data.world/s/6uhvcowok54gcj5jz3i2ccwldlb6rl", 
                header=TRUE, stringsAsFactors=FALSE)

##### Pré-Processamento #####

duplicados = duplicated(base,fromLast = TRUE)
which(duplicados) #12 linhas duplicadas
base=base[!duplicados,] 

dados = base[,-1] # o nome do jogador não é relevante para avaliar se ele será promissor ou não

dados$TARGET_5Yrs=as.factor(dados$TARGET_5Yrs)

# Tratando Na's
sum(is.na(base$x3p.))
apply(dados,2, function(x) any(is.na(x)))
preproc_NA = preProcess(dados,method = "knnImpute",k=5)
dados_na = predict(preproc_NA,dados)

# correlação ponto de corte 0.75

var_num= c(names(Filter(is.numeric,dados_na)))
descrCor =  cor(dados_na[var_num])

#Quais variaveis tem alta correlacao?
findCorrelation(descrCor, cutoff = .75, verbose=T)

#Novo banco sem var. com alta correlacao
highCor=findCorrelation(descrCor, cutoff = .75,names=T) 

dados_cor = dplyr::select(dados_na,-highCor) 

##### Regressão Logística #####

set.seed(2021)

inTrain = createDataPartition(dados_cor$TARGET_5Yrs,p=0.75,list=F)
treino = dados_cor[inTrain,]
teste  = dados_cor[-inTrain,]

library("VGAM")
library("e1071")

ctrl = trainControl(method="boot", number=3)

model = train(TARGET_5Yrs~., data=treino, trControl=ctrl, method="vglmAdjCat")

predicao = predict(model,teste)

confusionMatrix(predicao, teste$TARGET_5Yrs, positive = '1')