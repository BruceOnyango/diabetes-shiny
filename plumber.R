library(plumber)

library(caret)

library(e1071)



model <- readRDS("logistic_model.rds")



#* @post /predict

#* @serializer json

function(req) {

  body <- jsonlite::fromJSON(req$postBody)

  

  bmi_category <- as.integer(cut(as.numeric(body$bmi),

    breaks = c(0, 18.5, 25, 30, Inf),

    labels = c(0, 1, 2, 3),

    right  = FALSE))

  

  glucose_bmi <- round((as.numeric(body$glucose) * as.numeric(body$bmi)) / 100, 4)

  

  new_data <- data.frame(

    Glucose                  = as.numeric(body$glucose),

    BMI                      = as.numeric(body$bmi),

    Age                      = as.numeric(body$age),

    Pregnancies              = as.numeric(body$pregnancies),

    Insulin                  = as.numeric(body$insulin),

    DiabetesPedigreeFunction = as.numeric(body$dpf),

    BloodPressure            = as.numeric(body$bp),

    SkinThickness            = as.numeric(body$skin),

    Glucose_BMI_Score        = glucose_bmi,

    BMI_Category             = bmi_category

  )

  

  prob  <- predict(model, new_data, type = "prob")[,"Diabetic"]

  pct   <- round(prob * 100, 1)

  level <- if (pct < 30) "Low Risk" else if (pct <= 60) "Medium Risk" else "High Risk"

  

  list(probability = pct, risk_level = level)

}
