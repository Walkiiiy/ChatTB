# schema description
The database 'thrombosis_prediction' appears to be a medical database focused on predicting and analyzing thrombosis (blood clotting) in patients. It includes laboratory test results, patient demographics, and examination data to assess the risk and presence of thrombosis. The database consists of three tables: 'Laboratory.csv', 'Patient.csv', and 'Examination.csv'.

1. **Laboratory.csv**: This table contains laboratory test results for patients, including various biochemical and hematological parameters. Each row represents a test result for a patient on a specific date.
  - **ID**: Unique identification of the patient (integer).
  - **Date**: Date of the laboratory tests (YYMMDD format).
  - **GOT (AST)**: Measures AST levels (integer, normal range < 60).
  - **GPT (ALT)**: Measures ALT levels (integer, normal range < 60).
  - **LDH**: Measures lactate dehydrogenase levels (integer, normal range < 500).
  - **ALP**: Measures alkaliphophatase levels (integer, normal range < 300).
  - **TP**: Measures total protein levels (real, normal range 6.0-8.5).
  - **ALB**: Measures albumin levels (real, normal range 3.5-5.5).
  - **UA**: Measures uric acid levels (real, normal range varies by gender).
  - **UN**: Measures urea nitrogen levels (integer, normal range < 30).
  - **CRE**: Measures creatinine levels (real, normal range < 1.5).
  - **T-BIL**: Measures total bilirubin levels (real, normal range < 2.0).
  - **T-CHO**: Measures total cholesterol levels (integer, normal range < 250).
  - **TG**: Measures triglyceride levels (integer, normal range < 200).
  - **CPK**: Measures creatinine phosphokinase levels (integer, normal range < 250).
  - **GLU**: Measures blood glucose levels (integer, normal range < 180).
  - **WBC**: Measures white blood cell count (real, normal range 3.5-9.0).
  - **RBC**: Measures red blood cell count (real, normal range 3.5-6.0).
  - **HGB**: Measures hemoglobin levels (real, normal range 10-17).
  - **HCT**: Measures hematocrit levels (real, normal range 29-52).
  - **PLT**: Measures platelet count (integer, normal range 100-400).
  - **PT**: Measures prothrombin time (real, normal range < 14).
  - **APTT**: Measures activated partial prothrombin time (integer, normal range < 45).
  - **FG**: Measures fibrinogen levels (real, normal range 150-450).
  - **PIC, TAT, TAT2**: Measures related to thrombosis (continuous values).
  - **U-PRO**: Measures proteinuria (text, normal range 0-30).
  - **IGG, IGA, IGM**: Measures immunoglobulin levels (integer, normal ranges vary).
  - **CRP**: Measures C-reactive protein levels (text, normal range N < 1.0).
  - **RA, RF**: Measures rheumatoid factor (text, normal range N= -, +-).
  - **C3, C4**: Measures complement levels (integer, normal ranges vary).
  - **RNP, SM, SC170, SSA, SSB, CENTROMEA, DNA, DNA-II**: Measures autoantibodies (text or integer, normal ranges vary).

2. **Patient.csv**: This table contains demographic and basic information about patients.
  - **ID**: Unique identification of the patient (integer).
  - **SEX**: Gender of the patient (text: F or M).
  - **Birthday**: Patient's birth date (date).
  - **Description**: First date when patient data was recorded (date).
  - **First Date**: Date when the patient came to the hospital (date).
  - **Admission**: Indicates if the patient was admitted to the hospital (text: + or -).
  - **Diagnosis**: Disease names (text).

3. **Examination.csv**: This table contains specialized examination data related to thrombosis.
  - **ID**: Unique identification of the patient (integer).
  - **Examination Date**: Date of the examination (date).
  - **aCL IgG, aCL IgM, aCL IgA**: Measures anti-Cardiolipin antibody levels (real or integer).
  - **ANA**: Measures anti-nucleus antibody levels (integer).
  - **ANA Pattern**: Pattern observed in ANA examination (text).
  - **Diagnosis**: Disease names (text).
  - **KCT, RVVT, LAC**: Measures of coagulation (text: + or -).
  - **Symptoms**: Other symptoms observed (text).
  - **Thrombosis**: Degree of thrombosis (integer: 0-3).

The tables are linked by the 'ID' column, which uniquely identifies each patient across all tables. This allows for cross-referencing laboratory results, patient demographics, and examination data to analyze thrombosis risk and outcomes.

# background documents
