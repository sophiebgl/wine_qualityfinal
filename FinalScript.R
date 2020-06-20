

## Objectif / Problématique

We have a data set about red wines quality. We are going to look into this data set variables, try to find similarities between wines as per their quality and build a predictive model about the quality of the wines.

library("knitr")
opts_chunk$set(echo = TRUE, eval = TRUE, warning = FALSE, message = FALSE, cache = TRUE, fig.align = 'center', dpi = 300, out.width = '75%')

# Librairies
library('lattice')
library('ggplot2') # data visualisation
library('caret') # split dataset into train et test samples
library('corrplot')
library('cowplot')
library('dplyr') # dataframe manipulation
library('DT') # tables visualisation
library('factoextra')
library('FactoMineR')
library('pROC')
library('randomForest')
library('rpart')

# set default ggplot theme
theme_set(
  theme_light(
    base_size = 15
  ) +
    theme(
      text = element_text(family = "Arial", colour = "gray10"),
      panel.border = element_blank(),
      axis.line = element_line(colour = "gray50", size = .5),
      axis.ticks = element_blank(),
      # axis.title.y = element_text(angle = 0),
      strip.background = element_rect(colour = "gray50", fill = "transparent", size = .7),
      strip.text.x = element_text(colour = "gray10"),
      legend.key.size = unit(2, "cm")
    )
)

#Dataset

vin <- read.csv(file = "qualitevin.csv", header = T, sep = ";")


## Exploratory analysis

### Building the data set

#Data Overview

head(vin)

#Missing values

na_values <- sort(sapply(vin, function(x) sum(is.na(x))), decreasing = TRUE)
data.frame(
  x = names(na_values),
  y = na_values
) %>% 
  datatable(
    colnames = c("Variable", 'Valeurs Manquantes'),
    rownames = FALSE,
    width = '50%',
    options = list(
      pageLength = 12, 
      autoWidth = FALSE,
      lengthChange = FALSE, 
      bPaginate = FALSE, 
      bFilter = FALSE,
      bInfo = FALSE,
      columnDefs = list(list(width = '30px', targets = c(0, 1), className = 'dt-center'))
    )
  )


#No missing values
                                                      

# Transforming qualitéY as factor
vin$qualiteY <- factor(vin$qualiteY)
levels(vin$qualiteY) # un vin considéré comme qualitatif est codé 1, comme mauvais 0.

                                                      
# This function calculate the proportion of the targeted variable 
                                                      
prop_var_cible <- function(data) {
datatable(data.frame(round(prop.table(table(data$qualiteY))*100,2)),
rownames = FALSE,
colnames = c('qualiteY', 'Fréquence'),
width = '50%',
options = list(
pageLength = 12, 
autoWidth = FALSE,
lengthChange = FALSE, 
bPaginate = FALSE, 
bFilter = FALSE,
bInfo = FALSE,
columnDefs = list(list(width = '30px', targets = c(0, 1), className = 'dt-center'))))
}
                                                      
prop_var_cible(vin)


### Variables distri
                                                      
#This function plots the density of a numeric variable.
                                                      
plotDensity <- function(data, var) {
ggplot(
data = data,
mapping = aes_string(x = var)
) +
geom_density() +
labs(y = "Densité") +
theme(
axis.title.x = element_text(size = 8),
axis.text.x = element_text(size = 8),
axis.title.y = element_text(size = 8),
axis.text.y = element_text(size = 8)
)}
                                                      
density_plots <- lapply(
X = names(vin)[1:11],
FUN = function(name) plotDensity(vin, name)
)
plot_grid(plotlist = density_plots, nrow = 3)
                                                      
#Most of the variables are normally distributed, except `residual.sugar` and `chloride`.

### Correlation

#### Correlation matrix (Pearson)

mat <- cor(vin[,-12])
corrplot(mat,type = 'upper')

#### Graphs

ggplot(data = vin, 
       mapping = aes(x = fixed.acidity, y = citric.acid)) +
  geom_point(color = 'darkblue') +
  stat_smooth(method="lm") +
  ggtitle('Liens entre fixed.acidity et citric.acidity', subtitle = paste('Coefficient de corrélation :', round(cor(vin$fixed.acidity, vin$citric.acid),2))) +
  theme(plot.subtitle = element_text(hjust = 0.5))

ggplot(data = vin, 
       mapping = aes(x = fixed.acidity, y = density)) +
  geom_point(color = 'darkred') +
  stat_smooth(method="lm") +
  ggtitle('Liens entre fixed.acidity et density', subtitle = paste('Coefficient de corrélation :', round(cor(vin$fixed.acidity, vin$density),2))) +
  theme(plot.subtitle = element_text(hjust = 0.5),
        plot.title = element_text(hjust = 0.5)
  )

ggplot(data = vin, 
       mapping = aes(x = fixed.acidity, y = pH)) +
  geom_point(color = 'darkgreen') +
  stat_smooth(method="lm") +
  ggtitle('Liens entre fixed.acidity et pH', subtitle = paste('Coefficient de corrélation :', round(cor(vin$fixed.acidity, vin$pH),2))) +
  theme(plot.subtitle = element_text(hjust = 0.5),
        plot.title = element_text(hjust = 0.5)
  )

ggplot(data = vin, 
       mapping = aes(x = volatile.acidity, y = citric.acid)) +
  geom_point(color = 'orange') +
  stat_smooth(method="lm") +
  ggtitle('Liens entre volatile.acidity et citric.acid', subtitle = paste('Coefficient de corrélation :', round(cor(vin$volatile.acidity, vin$citric.acid),2))) +
  theme(plot.subtitle = element_text(hjust = 0.5),
        plot.title = element_text(hjust = 0.5)
  )


ggplot(data = vin, 
       mapping = aes(x = volatile.acidity, y = citric.acid)) +
  geom_point(color = 'darkgrey') +
  stat_smooth(method="lm") +
  ggtitle('Liens entre pH et citric.acid', subtitle = paste('Coefficient de corrélation :', round(cor(vin$pH, vin$citric.acid),2))) +
  theme(plot.subtitle = element_text(hjust = 0.5),
        plot.title = element_text(hjust = 0.5)
)


### Variable influence on wine quality

#This function creates boxplots for a numeric variable.

plotBoxplot <- function(data, var) {
  ggplot(
    data = data,
    mapping = aes_string(x = "qualiteY", y = var)
  ) +
    geom_boxplot() +
    ggtitle(var) +
    theme(
      axis.title.x = element_text(size = 8),
      axis.text.x = element_text(size = 8),
      axis.title.y = element_text(size = 8),
      axis.text.y = element_text(size = 8),
      plot.title = element_text(size = 10)
    )
}


box_plots <- lapply(
  X = names(vin)[c(2,3,10,11)],
  FUN = function(name) plotBoxplot(data = vin, var = name)
)

plot_grid(plotlist = box_plots, nrow = 2)

### Test de comparaison de moyennes

# 1 - Volatile acidity

#Variance test
var.test(volatile.acidity ~ qualiteY, data = vin, conf.level = 0.95)

#Mean test
t.test.volatile.acidity <- t.test(volatile.acidity ~ qualiteY, alternative = 'greater', var.equal = FALSE, data = vin, conf.level = 0.95)
# Les moyennes sont significativement différentes et nous rejettons H0 au seuil de 5%.

# 2 - Citric acid

var.test(citric.acid ~ qualiteY, data = vin, conf.level = 0.95)

#Mean test
t.test.citric.acide <- t.test(citric.acid ~ qualiteY, alternative = 'less', var.equal = FALSE, data = vin, conf.level = 0.95)

# 3 - Sulphates
var.test(sulphates ~ qualiteY, data = vin, conf.level = 0.95)

#Mean test
t.test.sulphates <- t.test(sulphates ~ qualiteY, alternative = 'less', var.equal = FALSE, data = vin, conf.level = 0.95)

# 4 - Alcohol
var.test(alcohol ~ qualiteY, data = vin, conf.level = 0.95)

#Mean test
t.test.alcool <- t.test(alcohol ~ qualiteY, alternative = 'less', var.equal = FALSE, data = vin, conf.level = 0.95)


## Multivariate Descriptive statistical analysis 

acp <- PCA(X = vin, quali.sup = 12, graph = FALSE)


### Analysis


acp$eig[,2:3]


### Scree plot
fviz_eig(acp, addlabels = TRUE)


### Variable analysis

#### Axes 1 and 2
plot(acp, choix = "var", axes = c(1,2), title = 'ACP - axes 1 et 2')


#### Axes 3 and 4
plot(acp, choix = "var", axes = c(3,4), title = 'ACP - axes 3 et 4')


### Analysis

fviz_pca_ind(acp,
             geom.ind = "point", # show points only (nbut not "text")
             col.ind = vin$qualiteY, # color by groups
             palette = c("#00AFBB", "#E7B800", "#FC4E07"),
             addEllipses = TRUE, # Concentration ellipses
             legend.title = "Qualité de vin",
             title = 'Représentation des individus selon la qualité de vin -  axes 1 et 2'
             )

fviz_pca_ind(acp,
             axes = c(3,4),
             geom.ind = "point", # show points only (nbut not "text")
             col.ind = vin$qualiteY, # color by groups
             palette = c("#00AFBB", "#E7B800", "#FC4E07"),
             addEllipses = TRUE, # Concentration ellipses
             legend.title = "Qualité de vin",
             title = 'Représentation des individus selon la qualité de vin - axes 3 et 4'
)




fviz_pca_ind(acp, col.ind = "cos2", 
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             title = 'Représentation des individus'
             )

fviz_pca_ind(acp, col.ind = "contrib", 
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             title = 'Contribution des individus'
             )

datatable(vin[c(152,1436,1477),], 
          options = list(
            scrollX = TRUE,
            pageLength = 12, 
            autoWidth = FALSE,
            lengthChange = FALSE, 
            bPaginate = FALSE, 
            bFilter = FALSE,
            bInfo = FALSE))


## Logistic regression


# Modèle complet'
model_complet_expl <- glm(formula = qualiteY~., data = vin, family = 'binomial')

model_select_expl <- step(model_complet_expl, direction = 'both', trace = FALSE) 
```

summary(model_select_expl)


### Predictive model


#### Test and training samples

# Training
set.seed(1) # permet de rendre reproducibles les résultats
trainIndex <- createDataPartition(vin$qualiteY, p = 0.8, list = FALSE)

vin_train <- vin[trainIndex,]

# Test
vin_test <- vin[-trainIndex,]
```

#**Train**
prop_var_cible(vin_train)
```
#**Test**
# Proportion de la variable cible dans l'échantillon de test
prop_var_cible(vin_test)

#### Training model

# Full model
model_complet <- glm(formula = qualiteY~., data = vin_train, family = 'binomial')

# Choix des variables sur le critère AIC (minimisation de l'AIC)
model_select <- step(model_complet, direction = 'both', trace = FALSE)


#### Model quality

prevision <- predict(model_select, newdata = vin_test, type = 'response')


#####  ROC 

# Prediction
roc_prevision <- roc(vin_test$qualiteY, prevision)

ggroc(roc_prevision, alpha = 0.5, colour = "red") + 
  geom_segment(aes(x = 1, xend = 0, y = 0, yend = 1), color="grey", linetype="dashed") +
  ggtitle(paste('Prévision - AUC =',round(roc_prevision$auc,2)) 
  )

##### AUC

auc_prev <- round(auc(roc_prevision),2)

prevision_result <- as.numeric(prevision>= 0.5) 

mat_conf <- function(prev, test) {
  matconft <- data.frame(matrix(table(prev, test$qualiteY),nrow = 2))
  colnames(matconft) <- c('0', '1')
  row.names(matconft) <- c('0', '1')
  datatable(matconft,
            width = '30%',
            options = list(
              autoWidth = FALSE,
              lengthChange = FALSE,
              bPaginate = FALSE,
              bFilter = FALSE,
              bInfo = FALSE))
}

mat_conf(prevision_result, vin_test)


coord <- coords(roc = roc_prevision, x = 'best')
print(coord)


##### Confusion matrix

seuil <- coord[1]

prevision_result <- as.numeric(prevision>= seuil) 

mat_conf(prevision_result, vin_test)

# accuracy / sensitivity / specificity'

confusionMatrix(data = as.factor(prevision_result), reference = vin_test$qualiteY, positive="1")



#### Residual analysis 


res <- rstudent(model_select)

plot(res, pch = 20, col = 'darkblue')
abline(h = c(2,0,-2), lty = c(2,1,2), col = 'red')
out <- which(!between(res, -2, 2))
points(x = out, y = res[out], col = 'magenta')


which.max(abs(res))

res <- residuals(model_select, type = "pearson")

plot(res, pch = 20, col = 'darkblue')
abline(h = c(2,0,-2), lty = c(2,1,2), col = 'red')
out <- which(!between(res, -2, 2))
points(x = out, y = res[out], col = 'magenta')

max <- which.max(abs(res)) #653




## Other models to compare with



#### Interactive model

modele_interaction <- glm(qualiteY~.^2, data = vin_train, family="binomial")
modele_interaction_select <- step(modele_interaction, direction = 'both', trace = FALSE) 

# Test sample
prevision_interaction <- predict(modele_interaction_select,vin_test, type="response")

# AUC
AUC.interaction <- auc(roc(vin_test$qualiteY,prevision_interaction)) 

#### Model without correlated variable

vin_train_corr <- vin_train[,c(1,2,4,5,7,8,10,11,12)]
vin_test_corr <- vin_test[,c(1,2,4,5,7,8,10,11,12)]

modele_corr <- glm(qualiteY~., data = vin_train_corr, family="binomial")
modele_corr_select <- step(modele_corr, direction = 'both', trace = FALSE) 

# Test sample
prevision_corr <- predict(modele_corr_select,vin_test_corr, type="response")

# AUC
AUC.corr <- auc(roc(vin_test_corr$qualiteY,prevision_corr)) 


#### Tree
modele_arbre <- rpart(qualiteY~., data = vin_train)

# Test sample
prevision_arbre <- predict(modele_arbre, vin_test, type='prob')[,2]

# AUC
AUC.tree <- auc(roc(vin_test$qualiteY,prevision_arbre)) 


#### Random forest
foret <- randomForest(qualiteY~., vin_train)

# Test sample
prevision_foret <- predict(foret, vin_test, type='prob')[,2]

# AUC
AUC.forest <- auc(roc(vin_test$qualiteY,prevision_foret)) 


     
    
    
#### Best model and confusion matrix
   
#MConfusion Matrix with Random Forest (0.5)
 
prevision_result_foret <- as.numeric(prevision_foret>= 0.5) 

mat_conf(prevision_result_foret, vin_test)

err.forest <- round(sum(prevision_result_foret != vin_test$qualiteY)/nrow(vin_test)*100,2)

