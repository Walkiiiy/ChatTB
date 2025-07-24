# A
The **California Schools Database** contains data on various educational institutions across California, including details about the schools, districts, and related characteristics. This database appears to focus on the operations, performance, and demographic attributes of public schools, including their academic performance, enrollment statistics, and administrative details. The data is divided into several tables that categorize information regarding school performance, school and district details, and other aspects such as funding and grade levels.

---

### Table Descriptions:

1. **frpm.csv**:

   * This table contains information related to Free and Reduced-Price Meal (FRPM) eligibility for schools. It tracks various attributes such as academic year, district and school codes, and the number of students receiving free or reduced-price meals.

     * **CDSCode**: Represents the unique code for each school, an identifier for the California Department of Education.
     * **Academic Year**: The academic year of the data.
     * **County Code and Name**: Code and name of the county where the school is located.
     * **District Code and Name**: The unique district code and name.
     * **School Code**: A unique identifier for each school.
     * **School Name**: The name of the school.
     * **District Type**: The type of school district (e.g., Elementary, High School, Unified).
     * **School Type**: Defines the school type (e.g., K-12, Alternative Schools).
     * **Educational Option Type**: Specifies the type of educational option available (e.g., Alternative School).
     * **NSLP Provision Status**: Specifies the provision status under the National School Lunch Program (NSLP).
     * **Charter School (Y/N)**: Indicates whether the school is a charter school (1 = Yes, 0 = No).
     * **Enrollment (K-12)**: The number of students enrolled in K-12 grades.
     * **Free Meal Count (K-12)**: The number of students eligible for free meals in K-12.
     * **Percent (%) Eligible Free (K-12)**: The percentage of K-12 students eligible for free meals.
     * **FRPM Count (K-12)**: The number of students eligible for Free or Reduced-Price Meals in K-12.
     * **Percent (%) Eligible FRPM (K-12)**: The percentage of K-12 students eligible for FRPM.
     * **Enrollment (Ages 5-17)**: The number of students aged 5-17 enrolled.
     * **Free Meal Count (Ages 5-17)**: The number of students aged 5-17 eligible for free meals.
     * **Percent (%) Eligible Free (Ages 5-17)**: The percentage of students aged 5-17 eligible for free meals.
     * **FRPM Count (Ages 5-17)**: The number of students aged 5-17 eligible for FRPM.
     * **Percent (%) Eligible FRPM (Ages 5-17)**: The percentage of students aged 5-17 eligible for FRPM.

2. **satscores.csv**:

   * This table provides SAT scores and test-taking data for schools across California.

     * **cds**: The CDS code of the school.
     * **rtype**: The type of the SAT score data.
     * **sname**: The name of the school.
     * **dname**: The name of the district.
     * **cname**: The name of the county.
     * **enroll12**: The number of students enrolled in grades 1-12.
     * **NumTstTakr**: The number of students who took the SAT.
     * **AvgScrRead, AvgScrMath, AvgScrWrite**: Average SAT scores in reading, math, and writing.
     * **NumGE1500**: The number of test-takers with a total SAT score greater or equal to 1500.

3. **schools.csv**:

   * This table contains detailed information about schools, districts, and their geographical locations.

     * **CDSCode**: The unique identifier for the school.
     * **NCESDist and NCESSchool**: Unique identifiers for the school district and the school from the National Center for Education Statistics (NCES).
     * **StatusType**: The operational status of the district (e.g., Active, Closed).
     * **County, District, School**: Geographic information on the location of the school, including county, district, and the school itself.
     * **Street, StreetAbr**: Address information for the school, both in full and abbreviated form.
     * **Phone**: The contact phone number for the school.
     * **Zip, MailZip**: The postal codes for the school’s location and mailing address.
     * **Charter, CharterNum**: Indicates if the school is a charter and provides the charter number if applicable.
     * **FundingType**: Describes the funding model of charter schools.
     * **DOC and DOCType**: The District Ownership Code and its description.
     * **SOC and SOCType**: The School Ownership Code and its description.
     * **Website**: The school's website address.
     * **Latitude and Longitude**: Geographical coordinates of the school.

---

### Key Column Background Knowledge Needed:

These columns need additional clarification:

1. **"County Code"** from **frpm.csv**:

   * More detailed background is required to understand how the counties are classified or linked to specific educational or demographic characteristics.

2. **"District Type"** from **frpm.csv**:

   * A more thorough understanding of each district type (e.g., Elementary, High School) and the criteria for classification.

3. **"School Type"** from **frpm.csv**:

   * Clarification is needed on what defines each school type, especially for categories like "State Special Schools" or "Non-School Locations."

4. **"NSLP Provision Status"** from **frpm.csv**:

   * A deeper explanation of how different provisions work under the National School Lunch Program, and how schools are categorized.

5. **"Charter School Number"** from **frpm.csv**:

   * More context is needed on how charter school numbers are assigned and their significance.

6. **"Funding Type"** from **frpm.csv**:

   * Additional context needed on how the funding model is determined, particularly in relation to charter schools.

7. **"Grade Span Served"** from **schools.csv**:

   * Clarification on how this grade span classification is determined in relation to California Longitudinal Pupil Achievement (CALPADS) data.

8. **"Virtual"** from **schools.csv**:

   * Further details about how schools categorize the level of virtual instruction (e.g., exclusively virtual, primarily virtual).

### Columns Requiring Supplementary Background Knowledge:

```json
[
    {"originColumnName":"County Code", "fullColumnName":"County Code", "originTable":"frpm.csv"},
    {"originColumnName":"District Type", "fullColumnName":"District Type", "originTable":"frpm.csv"},
    {"originColumnName":"School Type", "fullColumnName":"School Type", "originTable":"frpm.csv"},
    {"originColumnName":"NSLP Provision Status", "fullColumnName":"NSLP Provision Status", "originTable":"frpm.csv"},
    {"originColumnName":"Charter School Number", "fullColumnName":"Charter School Number", "originTable":"frpm.csv"},
    {"originColumnName":"FundingType", "fullColumnName":"Funding Type", "originTable":"schools.csv"},
    {"originColumnName":"Grade Span Served", "fullColumnName":"Grade Span Served", "originTable":"schools.csv"},
    {"originColumnName":"Virtual", "fullColumnName":"Virtual", "originTable":"schools.csv"}
]
```
