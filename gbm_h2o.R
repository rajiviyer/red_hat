setwd("C:/Rajiv/Work/Data Science/Kaggle/red_hat")
library(data.table)
library(xgboost)
library(h2o)
localH2O <- h2o.init()
#localH2O <- h2o.init(nthreads = -1,max_mem_size="8g")


people <- fread("people.csv")
train <- fread("act_train.csv")
test <- fread("act_test.csv")
test$outcome <- NA
full <- rbind(train,test)

full <- full[,":="(char_1 = NULL,
				   char_2 = NULL,
				   char_3 = NULL,
				   char_4 = NULL,
				   char_5 = NULL,
				   char_6 = NULL,
				   char_7 = NULL,
				   char_8 = NULL,
				   char_9 = NULL
				  )
			]
setkey(full,people_id)
setkey(people,people_id)
#intersect(names(full),names(people))
setnames(people,c("date","char_10"),c("people_date","people_char_10"))
full <- merge(full,people,all=FALSE)

full <- full[,":=" (date = NULL, people_date = NULL)]
for (col in c(3:44)) set(full,j=col, value=as.factor(full[[col]]))


train <- full[!is.na(full$outcome)]
train_id <- train$activity_id
train <- train[,":=" (activity_id = NULL)]
test <- full[is.na(full$outcome)]
test_id <- test$activity_id
test <- test[,":=" (activity_id = NULL, outcome = NULL)]

features <- names(train)
trainH2o <- as.h2o(train)
splits <- h2o.splitFrame(trainH2o,c(0.9),seed=1424)
testH2o <- as.h2o(test)
glm_mod <- h2o.glm(	x=features,
					y="outcome",
					training_frame = splits[[1]],
					validation_frame=splits[[2]],
					family="binomial",
					nfolds=10,
					keep_cross_validation_predictions=TRUE
				 )
rf_mod <- h2o.randomForest(x=features,
						   y="outcome",
						   training_frame = splits[[1]],
						   validation_frame=splits[[2]],
						   ntrees=100,
						   max_depth=4,
						   nfolds=5,
						   keep_cross_validation_predictions=TRUE,
						   seed=1424)

gbm_mod <- h2o.gbm(x=features,
						   y="outcome",
						   training_frame = splits[[1]],
						   validation_frame=splits[[2]],
						   ntrees=300,
						   max_depth=4,
						   nfolds=3,
						   keep_cross_validation_predictions=TRUE,
						   seed=1424)						   

gbm_pred <- as.data.frame(h2o.predict(gbm_mod,newdata=testH2o))

my_solution <- data.frame(activity_id=test_id,outcome=gbm_pred$p1)

write.csv(my_solution,file="gbm_h20_rstudio.csv",row.names=FALSE)
						   