library(shiny)
library(caret)
library(e1071)

# Load model once at startup
model <- readRDS("logistic_model.rds")

# ── UI ────────────────────────────────────────────────────────────────────────
ui <- fluidPage(
  
  tags$head(
    tags$style(HTML("
      body { background-color: #f4f6f9; font-family: 'Segoe UI', sans-serif; }
      .title-box { background-color: #1a6e2e; color: white; padding: 20px 30px;
                   border-radius: 8px; margin-bottom: 25px; }
      .result-box { padding: 20px; border-radius: 8px; margin-top: 20px;
                    font-size: 18px; font-weight: bold; text-align: center; }
      .low    { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
      .medium { background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba; }
      .high   { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
      .input-card { background: white; padding: 20px; border-radius: 8px;
                    box-shadow: 0 1px 4px rgba(0,0,0,0.1); }
    "))
  ),
  
  div(class = "title-box",
    h2("Diabetes Risk Predictor", style="margin:0"),
    p("Pima Indians Diabetes Dataset — Logistic Regression Model (AUC 0.818)",
      style="margin:5px 0 0 0; font-size:13px; opacity:0.85")
  ),
  
  fluidRow(
    
    # ── LEFT: Inputs ──────────────────────────────────────────────────────────
    column(5,
      div(class = "input-card",
        h4("Patient Details", style="margin-top:0; color:#1a6e2e"),
        
        numericInput("glucose", "Glucose (mg/dL)",
                     value = 117, min = 50, max = 250, step = 1),
        numericInput("bmi", "BMI (kg/m²)",
                     value = 32, min = 10, max = 70, step = 0.1),
        numericInput("age", "Age (years)",
                     value = 30, min = 21, max = 100, step = 1),
        numericInput("pregnancies", "Number of Pregnancies",
                     value = 1, min = 0, max = 20, step = 1),
        numericInput("insulin", "Insulin (mu U/ml)",
                     value = 125, min = 0, max = 850, step = 1),
        numericInput("dpf", "Diabetes Pedigree Function",
                     value = 0.47, min = 0.0, max = 2.5, step = 0.01),
        numericInput("bp", "Blood Pressure (mm Hg)",
                     value = 72, min = 30, max = 130, step = 1),
        numericInput("skin", "Skin Thickness (mm)",
                     value = 29, min = 0, max = 100, step = 1),
        
        br(),
        actionButton("predict", "Calculate Risk",
                     style = "background-color:#1a6e2e; color:white;
                              width:100%; font-size:15px; padding:10px;
                              border:none; border-radius:6px;")
      )
    ),
    
    # ── RIGHT: Results ────────────────────────────────────────────────────────
    column(7,
      div(class = "input-card",
        h4("Risk Assessment", style="margin-top:0; color:#1a6e2e"),
        
        uiOutput("result_box"),
        
        br(),
        h5("How this is calculated:"),
        tableOutput("input_summary"),
        
        br(),
        h5("Risk Thresholds:"),
        tags$ul(
          tags$li(tags$span(style="color:#155724; font-weight:bold", "Low Risk: "),
                  "< 30% probability"),
          tags$li(tags$span(style="color:#856404; font-weight:bold", "Medium Risk: "),
                  "30% – 60% probability"),
          tags$li(tags$span(style="color:#721c24; font-weight:bold", "High Risk: "),
                  "> 60% probability")
        ),
        
        br(),
        p("Note: This tool is for educational purposes only and is not a
          clinical diagnostic tool. Always consult a medical professional.",
          style="font-size:11px; color:#888; font-style:italic")
      )
    )
  )
)

# ── SERVER ────────────────────────────────────────────────────────────────────
server <- function(input, output) {
  
  observeEvent(input$predict, {
    
    # Recreate engineered features exactly as in training
    bmi_category <- as.integer(cut(input$bmi,
                                   breaks = c(0, 18.5, 25, 30, Inf),
                                   labels = c(0, 1, 2, 3),
                                   right  = FALSE))
    glucose_bmi  <- round((input$glucose * input$bmi) / 100, 4)
    
    new_data <- data.frame(
      Glucose                  = input$glucose,
      BMI                      = input$bmi,
      Age                      = input$age,
      Pregnancies              = input$pregnancies,
      Insulin                  = input$insulin,
      DiabetesPedigreeFunction = input$dpf,
      BloodPressure            = input$bp,
      SkinThickness            = input$skin,
      Glucose_BMI_Score        = glucose_bmi,
      BMI_Category             = bmi_category
    )
    
    # Predict probability
    prob <- predict(model, new_data, type = "prob")[,"Diabetic"]
    pct  <- round(prob * 100, 1)
    
    # Risk level
    level <- if (pct < 30) "low" else if (pct <= 60) "medium" else "high"
    label <- if (pct < 30) "Low Risk" else if (pct <= 60) "Medium Risk" else "High Risk"
    
    output$result_box <- renderUI({
      div(class = paste("result-box", level),
        paste0(label, " — ", pct, "% probability of diabetes")
      )
    })
    
    output$input_summary <- renderTable({
      data.frame(
        Feature = c("Glucose", "BMI", "Age", "Pregnancies",
                    "Insulin", "DPF", "Blood Pressure",
                    "Skin Thickness", "Glucose×BMI Score", "BMI Category"),
        Value   = c(input$glucose, input$bmi, input$age, input$pregnancies,
                    input$insulin, input$dpf, input$bp, input$skin,
                    glucose_bmi, bmi_category)
      )
    }, striped = TRUE, bordered = TRUE, hover = TRUE)
  })
}

shinyApp(ui, server)