# A
### Database Overview:

The **California Schools Database** provides comprehensive information about schools in California, focusing on various attributes such as their academic status, enrollment, and resources. It contains three tables: **frpm.csv**, **satscores.csv**, and **schools.csv**, which detail specific aspects like enrollment, test scores, and administrative data for each school. This data helps in analyzing the educational landscape, academic performance, school types, and district-level details across the state.

---

### Table Descriptions and Column Details:

#### **Table 1: frpm.csv**

* **Purpose**: Contains information about schools related to Free or Reduced Price Meal (FRPM) programs in California.

* **Columns**:

  1. **CDSCode**: A unique identifier for each school (integer). This is used to reference the school across various systems. The data is continuous and varies in magnitude.
  2. **Academic Year**: The academic year represented in a format "YYYY-YYYY" (integer). It is discrete, indicating when the data is valid.
  3. **County Code**: A unique identifier for each county (integer). Discrete values represent different counties, aiding in regional classification.
  4. **District Code**: The code representing the school district (integer). Like county codes, these are discrete identifiers.
  5. **School Code**: Unique identifier for the school (integer). Continuous data representing different schools.
  6. **County Name**: Name of the county (text). Discrete values represent the different counties.
  7. **District Name**: Name of the school district (text). A discrete field with multiple values.
  8. **School Name**: Name of the school (text). Discrete data for different schools.
  9. **District Type**: Type of district (text). Identifies whether the district is elementary, high school, or unified, among others.
  10. **School Type**: Type of the school (text). This column classifies the school as K-12, alternative schools, etc.
  11. **Educational Option Type**: The educational option offered by the school (text). Values like "Alternative School of Choice" are discrete categories.
  12. **NSLP Provision Status**: Indicates the provision status of the National School Lunch Program (text). Describes the kind of meal provisions the school has.
  13. **Charter School (Y/N)**: Whether the school is a charter school (integer). The data is binary (0 = No, 1 = Yes).
  14. **Charter School Number**: The number assigned to charter schools (text).
  15. **Charter Funding Type**: Describes how the charter school is funded (text). Values like "Directly funded" or "Locally funded" indicate funding sources.
  16. **IRC**: Data not useful for analysis (integer), but marked as a discrete field.
  17. **Low Grade**: The lowest grade level offered by the school (text). Discrete values like "K", "1", etc.
  18. **High Grade**: The highest grade level offered (text). Similar to the low grade column.
  19. **Enrollment (K-12)**: The number of K-12 students enrolled in the school (real). This is continuous data.
  20. **Free Meal Count (K-12)**: The number of students receiving free meals (real). Continuous.
  21. **Percent (%) Eligible Free (K-12)**: Percentage of K-12 students eligible for free meals (real). Continuous data between 0 and 1.
  22. **FRPM Count (K-12)**: The count of students eligible for free or reduced-price meals (real). Continuous.
  23. **Percent (%) Eligible FRPM (K-12)**: Percentage of K-12 students eligible for FRPM (real). Continuous data.
  24. **Enrollment (Ages 5-17)**: The number of students aged 5-17 enrolled (real). Continuous data.
  25. **Free Meal Count (Ages 5-17)**: The number of students aged 5-17 receiving free meals (real). Continuous.
  26. **Percent (%) Eligible Free (Ages 5-17)**: Percentage of students aged 5-17 eligible for free meals (real). Continuous.
  27. **FRPM Count (Ages 5-17)**: Count of students aged 5-17 eligible for FRPM (real). Continuous.
  28. **Percent (%) Eligible FRPM (Ages 5-17)**: Percentage of students aged 5-17 eligible for FRPM (real). Continuous.
  29. **2013-14 CALPADS Fall 1 Certification Status**: Indicates the certification status (integer). It helps confirm the data's validity.

#### **Table 2: satscores.csv**

* **Purpose**: Contains SAT scores and relevant test data for schools in California.

* **Columns**:

  1. **cds**: School's unique identifier (text). This is the same as the **CDSCode** in **frpm.csv** and is used to link the data.
  2. **rtype**: A classification field which is noted as unuseful.
  3. **sname**: Name of the school (text).
  4. **dname**: Name of the district (text).
  5. **cname**: County name (text).
  6. **enroll12**: Enrollment for grades 1-12 (integer).
  7. **NumTstTakr**: Number of test takers (integer).
  8. **AvgScrRead**: Average SAT Reading score (integer).
  9. **AvgScrMath**: Average SAT Math score (integer).
  10. **AvgScrWrite**: Average SAT Writing score (integer).
  11. **NumGE1500**: Number of test takers scoring 1500 or more (integer).

#### **Table 3: schools.csv**

* **Purpose**: Contains general information about the schools in California, including geographical data and administrative details.

* **Columns**:

  1. **CDSCode**: Unique identifier for each school (text).
  2. **NCESDist**: National Center for Educational Statistics (NCES) district identifier (text).
  3. **NCESSchool**: NCES school identifier (text).
  4. **StatusType**: Status of the district (text).
  5. **County**: Name of the county (text).
  6. **District**: Name of the district (text).
  7. **School**: School name (text).
  8. **Street**: School's street address (text).
  9. **StreetAbr**: Abbreviated street address (text).
  10. **City**: City of the school (text).
  11. **Zip**: Zip code (text).
  12. **State**: The state (CA).
  13. **MailStreet**: Mailing address of the school (text).
  14. **MailStrAbr**: Abbreviated mailing address (text).
  15. **MailCity**: Mailing city (text).
  16. **MailZip**: Mailing zip code (text).
  17. **MailState**: Mailing state (text).
  18. **Phone**: School's phone number (text).
  19. **Ext**: Phone extension (text).
  20. **Website**: School website (text).
  21. **OpenDate**: Date the school opened (date).
  22. **ClosedDate**: Date the school closed (date).
  23. **Charter**: Whether the school is a charter (integer).
  24. **CharterNum**: Charter school number (text).
  25. **FundingType**: Type of funding for charter schools (text).
  26. **DOC**: District Ownership Code (text).
  27. **SOC**: School Ownership Code (text).
  28. **EdOpsCode**: Educational Option Code (text).
  29. **EdOpsName**: Educational Option Name (text).
  30. **EILCode**: Educational Instruction Level Code (text).
  31. **EILName**: Educational Instruction Level Name (text).
  32. **GSoffered**: Grade span offered (text).
  33. **GSserved**: Grade span served (text).
  34. **Virtual**: Virtual instruction type (text).
  35. **Magnet**: Whether the school is a magnet school (integer).
  36. **Latitude**: Latitude coordinate of the school (real).
  37. **Longitude**: Longitude coordinate of the school (real).
  38. **AdmFName1**: Administrator's first name (text).
  39. **AdmLName1**: Administrator's last name (text).
  40. **AdmEmail1**: Administrator's email (text).
  41. **AdmFName2**: Administrator's second first name (text).
  42. **AdmLName2**: Administrator's second last name (text).
  43. **AdmEmail2**: Administrator's second email (text).
  44. **AdmFName3**: Third administrator's first name (text).
  45. **AdmLName3**: Third administrator's last name (text).
  46. **AdmEmail3**: Third administrator's email (text).
  47. **LastUpdate**: Last record update date (date).

---

### Key Columns Needing Supplementary Background Knowledge:

* **frpm.csv**:

  1. `County Code` - Clarification on how these counties are identified.
  2. `District Code` - Further detail on how districts are organized or identified within California.

* **satscores.csv**:

  1. `cds` - How this code aligns with other identifiers in the database.
  2. `rtype` - Further information on the distinction between "S" and "D".

* **schools.csv**:

  1. `NCESDist` - Background on how these codes are structured.
  2. `NCESSchool` - Explanation of this school identifier system.
