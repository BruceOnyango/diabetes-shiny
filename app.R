library(shiny)
library(caret)
library(e1071)
library(ggplot2)
library(dplyr)
library(reshape2)
library(pROC)
library(shinycssloaders)

# ── Load everything once at startup ───────────────────────────────────────────
model    <- readRDS("logistic_model.rds")
roc_saved <- readRDS("roc_data.rds")
df       <- read.csv("diabetes_cleaned.csv")
df$Outcome <- factor(df$Outcome,
                     levels = c("Non.Diabetic", "Diabetic"))

# ── Shared plot theme ─────────────────────────────────────────────────────────
theme_app <- function() {
  theme_minimal(base_size = 13) +
    theme(
      plot.title       = element_text(face = "bold", size = 14, colour = "#1a6e2e"),
      plot.subtitle    = element_text(size = 11, colour = "#555555"),
      legend.position  = "bottom",
      panel.grid.minor = element_blank()
    )
}

COLOURS <- c("Non.Diabetic" = "#4472C4", "Diabetic" = "#C00000")

# ── UI ────────────────────────────────────────────────────────────────────────
ui <- fluidPage(

  tags$head(tags$style(HTML("
    body { background-color: #f4f6f9; font-family: 'Segoe UI', sans-serif; }
    .title-box { background-color: #1a6e2e; color: white; padding: 20px 30px;
                 border-radius: 8px; margin-bottom: 25px; }
    .result-box { padding: 20px; border-radius: 8px; margin-top: 20px;
                  font-size: 18px; font-weight: bold; text-align: center; }
    .low    { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    .medium { background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba; }
    .high   { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
    .input-card { background: white; padding: 20px; border-radius: 8px;
                  box-shadow: 0 1px 4px rgba(0,0,0,0.1); margin-bottom: 15px; }
    .nav-tabs > li > a { color: #1a6e2e; font-weight: 500; }
    .nav-tabs > li.active > a { border-top: 3px solid #1a6e2e !important; }
    .insight-box { background: #e8f5e9; border-left: 4px solid #1a6e2e;
                   padding: 10px 15px; border-radius: 4px;
                   margin-bottom: 15px; font-size: 13px; }
  "))),

  div(class = "title-box",
    h2("Diabetes Risk Predictor", style = "margin:0"),
    p("Pima Indians Diabetes Dataset — Logistic Regression Model (AUC 0.818)",
      style = "margin:5px 0 0 0; font-size:13px; opacity:0.85")
  ),

 
    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — EXECUTIVE SUMMARY
    # ══════════════════════════════════════════════════════════════════════════   
  tabsetPanel(
    tabPanel("Executive Summary",

  fluidRow(
    # KPI cards
    column(3, div(class="input-card", style="text-align:center",
      h2("69%", style="color:#C00000; margin:0"),
      p("of high-glucose patients are diabetic"),
      p("vs 23% at normal glucose", style="font-size:11px; color:#888")
    )),
    column(3, div(class="input-card", style="text-align:center",
      h2("7 in 10", style="color:#1a6e2e; margin:0"),
      p("diabetic patients correctly identified"),
      p("at tuned screening threshold", style="font-size:11px; color:#888")
    )),
    column(3, div(class="input-card", style="text-align:center",
      h2("2.94x", style="color:#C00000; margin:0"),
      p("higher diabetes risk above 140 mg/dL glucose"),
      p("vs normal glucose level", style="font-size:11px; color:#888")
    )),
    column(3, div(class="input-card", style="text-align:center",
      h2("15%", style="color:#856404; margin:0"),
      p("of patients are highest-risk"),
      p("high glucose AND high BMI", style="font-size:11px; color:#888")
    ))
  ),

  br(),

  fluidRow(
    column(6, div(class="input-card",
      h4("What the model does", style="color:#1a6e2e"),
      tags$ul(
        tags$li("Screens patients using 8 routine clinical measurements"),
        tags$li("Flags high-risk individuals before symptoms appear"),
        tags$li("Requires no specialist — works at a standard check-up"),
        tags$li("Validated on 768 patients; AUC of 0.818")
      )
    )),
    column(6, div(class="input-card",
      h4("Key recommendations", style="color:#1a6e2e"),
      tags$ul(
        tags$li("Flag any patient with glucose >= 140 mg/dL for immediate review"),
        tags$li("Prioritise patients who are also obese (BMI >= 30) — 69% are diabetic"),
        tags$li("Screen women aged 40+ annually — highest prevalence group"),
        tags$li("Early detection avoids dialysis, amputation, and specialist costs")
      )
    ))
  ),

  br(),

  fluidRow(
    column(12, div(class="input-card",
      h4("Screening efficiency by risk group", style="color:#1a6e2e"),
      withSpinner(plotOutput("plot_exec_bar", height = "220px"))
        ))
    )
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — RISK PREDICTOR (your original app, unchanged)
    # ══════════════════════════════════════════════════════════════════════════
    tabPanel("Risk Predictor",

      fluidRow(

        # LEFT: Inputs
        column(5,
          div(class = "input-card",
            h4("Patient Details", style = "margin-top:0; color:#1a6e2e"),

            numericInput("glucose",     "Glucose (mg/dL)",            value = 117, min = 50,  max = 250, step = 1),
            numericInput("bmi",         "BMI (kg/m²)",                value = 32,  min = 10,  max = 70,  step = 0.1),
            numericInput("age",         "Age (years)",                value = 30,  min = 21,  max = 100, step = 1),
            numericInput("pregnancies", "Number of Pregnancies",      value = 1,   min = 0,   max = 20,  step = 1),
            numericInput("insulin",     "Insulin (mu U/ml)",          value = 125, min = 0,   max = 850, step = 1),
            numericInput("dpf",         "Diabetes Pedigree Function", value = 0.47,min = 0.0, max = 2.5, step = 0.01),
            numericInput("bp",          "Blood Pressure (mm Hg)",     value = 72,  min = 30,  max = 130, step = 1),
            numericInput("skin",        "Skin Thickness (mm)",        value = 29,  min = 0,   max = 100, step = 1),

            br(),
            actionButton("predict", "Calculate Risk",
              style = "background-color:#1a6e2e; color:white; width:100%;
                       font-size:15px; padding:10px; border:none; border-radius:6px;")
          )
        ),

        # RIGHT: Results
        column(7,
          div(class = "input-card",
            h4("Risk Assessment", style = "margin-top:0; color:#1a6e2e"),

            uiOutput("result_box"),

            br(),
            h5("How this is calculated:"),
            tableOutput("input_summary"),

            br(),
            h5("Risk Thresholds:"),
            tags$ul(
              tags$li(tags$span(style = "color:#155724; font-weight:bold", "Low Risk: "),
                      "< 30% probability"),
              tags$li(tags$span(style = "color:#856404; font-weight:bold", "Medium Risk: "),
                      "30% to 60% probability"),
              tags$li(tags$span(style = "color:#721c24; font-weight:bold", "High Risk: "),
                      "> 60% probability")
            ),

            br(),
            p("Note: This tool is for educational purposes only and is not a
               clinical diagnostic tool. Always consult a medical professional.",
              style = "font-size:11px; color:#888; font-style:italic")
          )
        )
      )
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — DATA OVERVIEW
    # ══════════════════════════════════════════════════════════════════════════
    tabPanel("Data Overview",

      fluidRow(
        column(12,
          div(class = "insight-box",
            "768 Pima Indian women aged 21+. 500 non-diabetic (65.1%) and
             268 diabetic (34.9%). Diabetic patients consistently show higher
             glucose, BMI, and age values across the dataset."
          )
        )
      ),

      fluidRow(
        column(4,
          div(class = "input-card", withSpinner(plotOutput("plot_class", height = "280px")))
        ),
        column(8,
          div(class = "input-card", withSpinner(plotOutput("plot_means", height = "280px")))
        )
      ),

      br(),

      fluidRow(
        column(12,
          div(class = "input-card",
            fluidRow(
              column(4,
                selectInput("hist_feature", "Explore feature distribution:",
                  choices  = c("Glucose","BMI","Age","Insulin","BloodPressure",
                               "Pregnancies","DiabetesPedigreeFunction","SkinThickness"),
                  selected = "Glucose")
              )
            ),
            withSpinner(plotOutput("plot_hist", height = "260px"))
          )
        )
      )
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 — RISK FACTORS
    # ══════════════════════════════════════════════════════════════════════════
    tabPanel("Risk Factors",

      fluidRow(
        column(12,
          div(class = "insight-box",
            "Glucose x BMI Score is the strongest predictor (r = 0.52), outperforming
             raw Glucose (r = 0.49) and BMI (r = 0.31) individually. Diabetic patients
             cluster in the high-glucose, high-BMI region of the scatter plot."
          )
        )
      ),

      fluidRow(
        column(7,
          div(class = "input-card",
            fluidRow(
              column(6, selectInput("x_axis", "X axis:",
                choices  = c("Glucose","BMI","Age","Insulin","DiabetesPedigreeFunction"),
                selected = "Glucose")),
              column(6, selectInput("y_axis", "Y axis:",
                choices  = c("BMI","Glucose","Age","Insulin","DiabetesPedigreeFunction"),
                selected = "BMI"))
            ),
            withSpinner(plotOutput("plot_scatter", height = "320px"))
          )
        ),
        column(5,
          div(class = "input-card",
            withSpinner(plotOutput("plot_importance", height = "380px"))
          )
        )
      ),

      br(),

      fluidRow(
        column(12,
          div(class = "input-card",
            withSpinner(plotOutput("plot_age", height = "260px"))
          )
        )
      )
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 5 — MODEL PERFORMANCE
    # ══════════════════════════════════════════════════════════════════════════
    tabPanel("Model Performance",

      fluidRow(
        column(12,
          div(class = "insight-box",
            "Logistic Regression achieved the highest AUC (0.818). At the tuned
             threshold of 0.40, sensitivity rises from 58.5% to 71.7% — catching
             7 in every 10 diabetic patients correctly."
          )
        )
      ),

      fluidRow(
        column(6,
          div(class = "input-card", withSpinner(plotOutput("plot_roc",       height = "360px")))
        ),
        column(6,
          div(class = "input-card", withSpinner(plotOutput("plot_threshold", height = "360px")))
        )
      ),

      br(),

      fluidRow(
        column(12,
          div(class = "input-card", withSpinner(plotOutput("plot_metrics", height = "300px")))
        )
      )
    )
  )
)

# ── SERVER ────────────────────────────────────────────────────────────────────
server <- function(input, output) {

  # ── TAB 1: Executive Summary ───────────────  
  output$plot_exec_bar <- renderPlot({
  exec_data <- data.frame(
    Group = c("General population","Glucose >= 140","Glucose >= 140\nAND BMI >= 30"),
    Probability = c(34.9, 68.5, 75.0),
    Fill = c("baseline","high","highest")
  )
  ggplot(exec_data, aes(x = Group, y = Probability, fill = Fill)) +
    geom_bar(stat="identity", width=0.5) +
    geom_text(aes(label = paste0(Probability, "%")),
              vjust=-0.4, fontface="bold", size=5) +
    scale_fill_manual(values = c(
      "baseline" = "#4472C4",
      "high"     = "#856404",
      "highest"  = "#C00000"
    )) +
    scale_y_continuous(limits=c(0,100),
                       expand=expansion(mult=c(0,0.15)),
                       labels=function(x) paste0(x,"%")) +
    labs(title    = "Probability of Diabetes by Screening Group",
         subtitle = "Targeting high-glucose patients nearly doubles detection efficiency",
         x="", y="Probability of Diabetes") +
    theme_app() + theme(legend.position="none") 
    })

  # ── TAB 2: Prediction ───────────────
  observeEvent(input$predict, {

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

    prob  <- predict(model, new_data, type = "prob")[,"Diabetic"]
    pct   <- round(prob * 100, 1)
    level <- if (pct < 30) "low" else if (pct <= 60) "medium" else "high"
    label <- if (pct < 30) "Low Risk" else if (pct <= 60) "Medium Risk" else "High Risk"

    output$result_box <- renderUI({
      div(class = paste("result-box", level),
        paste0(label, " — ", pct, "% probability of diabetes")
      )
    })

    output$input_summary <- renderTable({
      data.frame(
        Feature = c("Glucose","BMI","Age","Pregnancies","Insulin",
                    "DPF","Blood Pressure","Skin Thickness",
                    "Glucose x BMI Score","BMI Category"),
        Value   = c(input$glucose, input$bmi, input$age, input$pregnancies,
                    input$insulin, input$dpf, input$bp, input$skin,
                    glucose_bmi, bmi_category)
      )
    }, striped = TRUE, bordered = TRUE, hover = TRUE)
  })

  # ── TAB 3: Data Overview ───────────────────────────────────────────────────

  output$plot_class <- renderPlot({
    counts     <- as.data.frame(table(df$Outcome))
    names(counts) <- c("Outcome","Count")
    counts$Pct <- round(counts$Count / sum(counts$Count) * 100, 1)
    ggplot(counts, aes(x = Outcome, y = Count, fill = Outcome)) +
      geom_bar(stat = "identity", width = 0.5) +
      geom_text(aes(label = paste0(Count, "\n(", Pct, "%)")),
                vjust = -0.3, fontface = "bold", size = 4) +
      scale_fill_manual(values = COLOURS) +
      scale_y_continuous(expand = expansion(mult = c(0, 0.2))) +
      labs(title = "Class Distribution", x = "", y = "Count") +
      theme_app() + theme(legend.position = "none")
  })

  output$plot_means <- renderPlot({
    means <- df %>%
      group_by(Outcome) %>%
      summarise(Glucose = mean(Glucose),
                BMI     = mean(BMI),
                Age     = mean(Age), .groups = "drop") %>%
      melt(id.vars = "Outcome", variable.name = "Feature", value.name = "Mean")
    ggplot(means, aes(x = Feature, y = Mean, fill = Outcome)) +
      geom_bar(stat = "identity", position = "dodge", width = 0.6) +
      geom_text(aes(label = round(Mean, 1)),
                position = position_dodge(width = 0.6),
                vjust = -0.4, size = 3.5, fontface = "bold") +
      scale_fill_manual(values = COLOURS) +
      scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
      labs(title    = "Mean Values by Outcome",
           subtitle = "Diabetic patients are consistently higher across all three",
           x = "", y = "Mean Value") +
      theme_app()
  })

  output$plot_hist <- renderPlot({
    feat <- input$hist_feature
    ggplot(df, aes_string(x = feat, fill = "Outcome")) +
      geom_histogram(bins = 28, alpha = 0.65, position = "identity") +
      scale_fill_manual(values = COLOURS) +
      labs(title    = paste("Distribution of", feat, "by Outcome"),
           subtitle = "Overlap shows where prediction is hardest",
           x = feat, y = "Count") +
      theme_app()
  })

  # ── TAB 4: Risk Factors ────────────────────────────────────────────────────

  output$plot_scatter <- renderPlot({
    ggplot(df, aes_string(x = input$x_axis, y = input$y_axis,
                          colour = "Outcome")) +
      geom_point(alpha = 0.4, size = 1.8) +
      geom_smooth(method = "lm", se = TRUE, linewidth = 1) +
      scale_colour_manual(values = COLOURS) +
      labs(title    = paste(input$x_axis, "vs", input$y_axis),
           subtitle = "Diabetic patients cluster in the high-risk top-right zone") +
      theme_app()
  })

  output$plot_importance <- renderPlot({
    fi <- data.frame(
      Feature = c("Glucose_BMI_Score","Glucose","Age","BMI",
                  "DPF","Pregnancies","Insulin",
                  "BloodPressure","SkinThickness","BMI_Category"),
      Score   = c(100, 88.36, 54.17, 51.34, 47.94,
                  28.87, 23.05, 21.63, 20.10, 0.00)
    )
    ggplot(fi, aes(x = reorder(Feature, Score), y = Score,
                   fill = Score > 50)) +
      geom_bar(stat = "identity") +
      geom_text(aes(label = round(Score, 1)), hjust = -0.1, size = 3.5) +
      coord_flip() +
      scale_fill_manual(values = c("TRUE" = "#1a6e2e", "FALSE" = "#4472C4")) +
      scale_y_continuous(expand = expansion(mult = c(0, 0.18))) +
      labs(title    = "Random Forest Feature Importance",
           subtitle = "Engineered feature ranks above all raw variables",
           x = "", y = "Importance Score") +
      theme_app() + theme(legend.position = "none")
  })

  output$plot_age <- renderPlot({
    ggplot(df, aes(x = Age, fill = Outcome)) +
      geom_density(alpha = 0.5) +
      scale_fill_manual(values = COLOURS) +
      geom_vline(xintercept = 35, linetype = "dashed",
                 colour = "#333333", linewidth = 0.8) +
      annotate("text", x = 37, y = 0.04,
               label = "Risk accelerates\nafter age 35",
               hjust = 0, size = 3.5, colour = "#333333") +
      labs(title    = "Age Distribution by Outcome",
           subtitle = "Non-diabetic patients peak at 22-28; diabetic patients spread into the 50s",
           x = "Age (years)", y = "Density") +
      theme_app()
  })

  # ── TAB 5: Model Performance — REAL ROC curves from saved data ─────────────

  output$plot_roc <- renderPlot({

    roc_log  <- roc(roc_saved$actual, roc_saved$prob_log,
                    levels = c("Non.Diabetic","Diabetic"), quiet = TRUE)
    roc_tree <- roc(roc_saved$actual, roc_saved$prob_tree,
                    levels = c("Non.Diabetic","Diabetic"), quiet = TRUE)
    roc_rf   <- roc(roc_saved$actual, roc_saved$prob_rf,
                    levels = c("Non.Diabetic","Diabetic"), quiet = TRUE)

    roc_data <- rbind(
      data.frame(FPR   = 1 - roc_log$specificities,
                 TPR   = roc_log$sensitivities,
                 Model = paste0("Logistic Regression (AUC = ",
                                round(auc(roc_log), 3), ")")),
      data.frame(FPR   = 1 - roc_tree$specificities,
                 TPR   = roc_tree$sensitivities,
                 Model = paste0("Decision Tree (AUC = ",
                                round(auc(roc_tree), 3), ")")),
      data.frame(FPR   = 1 - roc_rf$specificities,
                 TPR   = roc_rf$sensitivities,
                 Model = paste0("Random Forest (AUC = ",
                                round(auc(roc_rf), 3), ")"))
    )

    ggplot(roc_data, aes(x = FPR, y = TPR, colour = Model)) +
      geom_line(linewidth = 1.2) +
      geom_abline(slope = 1, intercept = 0,
                  linetype = "dashed", colour = "grey60") +
      scale_colour_manual(values = setNames(
        c("#1a6e2e", "#C00000", "#4472C4"),
        c(paste0("Logistic Regression (AUC = ", round(auc(roc_log),  3), ")"),
          paste0("Decision Tree (AUC = ",       round(auc(roc_tree), 3), ")"),
          paste0("Random Forest (AUC = ",       round(auc(roc_rf),   3), ")"))
      )) +
      labs(title    = "ROC Curves — Model Comparison",
           subtitle = "Logistic Regression dominates; higher and further left = better",
           x = "False Positive Rate (1 - Specificity)",
           y = "True Positive Rate (Sensitivity)",
           colour = "") +
      theme_app()
  })

  output$plot_threshold <- renderPlot({
    thresh_data <- data.frame(
      Metric    = rep(c("Sensitivity","Specificity","F1-Score"), 2),
      Threshold = rep(c("Default (0.50)","Tuned (0.40)"), each = 3),
      Value     = c(0.585, 0.810, 0.602,
                    0.717, 0.720, 0.639)
    )
    thresh_data$Threshold <- factor(thresh_data$Threshold,
      levels = c("Default (0.50)","Tuned (0.40)"))

    ggplot(thresh_data, aes(x = Metric, y = Value, fill = Threshold)) +
      geom_bar(stat = "identity", position = "dodge", width = 0.6) +
      geom_text(aes(label = paste0(round(Value * 100, 1), "%")),
                position = position_dodge(width = 0.6),
                vjust = -0.4, size = 3.5, fontface = "bold") +
      scale_fill_manual(values = c("Default (0.50)" = "#4472C4",
                                   "Tuned (0.40)"   = "#1a6e2e")) +
      scale_y_continuous(labels = scales::percent,
                         expand = expansion(mult = c(0, 0.15)),
                         limits = c(0, 1)) +
      labs(title    = "Effect of Threshold Tuning",
           subtitle = "Lowering to 0.40 catches 13% more diabetic patients",
           x = "", y = "", fill = "Threshold") +
      theme_app()
  })

  output$plot_metrics <- renderPlot({
    metrics <- data.frame(
      Model  = rep(c("Logistic Regression","Decision Tree","Random Forest"), 4),
      Metric = rep(c("AUC","Accuracy","Sensitivity","Specificity"), each = 3),
      Value  = c(0.8177, 0.7280, 0.7762,
                 0.7516, 0.6405, 0.7320,
                 0.5849, 0.6792, 0.5849,
                 0.8400, 0.6200, 0.8100)
    )
    metrics$Model <- factor(metrics$Model,
      levels = c("Logistic Regression","Decision Tree","Random Forest"))

    ggplot(metrics, aes(x = Model, y = Value, fill = Model)) +
      geom_bar(stat = "identity", width = 0.6) +
      geom_text(aes(label = round(Value, 3)),
                vjust = -0.4, size = 3.2, fontface = "bold") +
      facet_wrap(~Metric, nrow = 1) +
      scale_fill_manual(values = c(
        "Logistic Regression" = "#1a6e2e",
        "Decision Tree"       = "#C00000",
        "Random Forest"       = "#4472C4"
      )) +
      scale_y_continuous(expand = expansion(mult = c(0, 0.2)), limits = c(0, 1)) +
      labs(title    = "Model Comparison Across All Metrics",
           subtitle = "Logistic Regression wins on AUC; Decision Tree trades specificity for sensitivity",
           x = "", y = "Score") +
      theme_app() +
      theme(legend.position  = "none",
            axis.text.x      = element_text(angle = 25, hjust = 1, size = 9),
            strip.text       = element_text(face = "bold", size = 11))
  })
}
# ── API ───────────────────────────────────────────────────────────────────────
predict_api <- function(glucose, bmi, age, pregnancies, insulin, dpf, bp, skin) {
  
  bmi_category <- as.integer(cut(as.numeric(bmi),
    breaks = c(0, 18.5, 25, 30, Inf),
    labels = c(0, 1, 2, 3),
    right  = FALSE))
  
  glucose_bmi <- round((as.numeric(glucose) * as.numeric(bmi)) / 100, 4)
  
  new_data <- data.frame(
    Glucose                  = as.numeric(glucose),
    BMI                      = as.numeric(bmi),
    Age                      = as.numeric(age),
    Pregnancies              = as.numeric(pregnancies),
    Insulin                  = as.numeric(insulin),
    DiabetesPedigreeFunction = as.numeric(dpf),
    BloodPressure            = as.numeric(bp),
    SkinThickness            = as.numeric(skin),
    Glucose_BMI_Score        = glucose_bmi,
    BMI_Category             = bmi_category
  )
  
  prob  <- predict(model, new_data, type = "prob")[,"Diabetic"]
  pct   <- round(prob * 100, 1)
  level <- if (pct < 30) "Low Risk" else if (pct <= 60) "Medium Risk" else "High Risk"
  
  list(probability = pct, risk_level = level)
}

# ── Run Plumber in background ─────────────────────────────────────────────────
library(future)
future::plan(multisession)
library(plumber)

app_dir <- getwd()

plumber_code <- paste0('
model <- readRDS("', app_dir, '/logistic_model.rds")

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
')

tmp <- tempfile(fileext = ".R")
writeLines(plumber_code, tmp)
pr <- plumb(tmp)
pr$run(port = 6880, host = "127.0.0.1")
shinyApp(ui, server)