# q
- you are a helpful data analyist. 

- you will describe a database schema very sepcifically, and extract the import columns that need to do background knowledge supplement.

- you will be given a database schema, which contains brief description of columns in the tables. the schema describes the database with every table's every column's origin column name(originColumnName), full column name(fullColumnName), brief column description(columnDescription), dataformat(dataFormat), size(size), number of empty values(emptyValueCount), continuous data or discrete genre(valType), samples(samples), uniqe value number(typeNum), average value(averageValue), max and minimum value(maximumValue,minimumValue), mathmatical varience(sampleVariance).

- however, some of the column names may lack background fields knowledge.If the column is of vital importance in the schema, even defines the main subject of the table, only a short name or short explaination may not cover the need of fully understand a schema. That's why sometimes background documents are needed.

- based on the given schema, please describe the schema in paragraph of natural language specificly,strictly in the format of below:
  - what is the database about?
  - for the database's every table:
    - what is the table about? 
    - for the table's every column:
      - what the column stands for?
      - in what form or unit is the column's value described? 
      - is the dataform strictly aligned?
      - how do the values desctibe this column?
  - which and how are the columns and tables linked or related?

- for the key columns needs to supplement their background knowledge, list them strictly in the format of below:
- [{"originColumnName":"...","fullColumnName":"...","originTable":"...csv"},......]
the input schema is :
{
    "databaseName": "california_schools",
    "table0": {
        "tableName": "frpm.csv",
        "column0": {
            "originColumnName": "CDSCode",
            "columnDescription": "CDSCode",
            "dataFormat": "integer",
            "size": 9986,
            "emptyValueCount": 0,
            "valType": "continuous",
            "samples": [
                36679596110357.0,
                10101086069488.0,
                39686766042691.0,
                30665976029268.0,
                36676523631157.0
            ],
            "averageValue": 29239035160209.906,
            "maximumValue": 58727695838305.0,
            "minimumValue": 1100170109835.0,
            "sampleVariance": 1.995776374286187e+26
        },
        "column1": {
            "originColumnName": "Academic Year",
            "columnDescription": "Academic Year",
            "dataFormat": "integer ",
            "size": 9986,
            "emptyValueCount": 0,
            "valType": "discrete",
            "typeNum": 1,
            "samples": [
                "2014-2015"
            ]
        },
        "column2": {
            "originColumnName": "County Code",
            "columnDescription": "County Code",
            "dataFormat": "integer",
            "size": 9986,
            "emptyValueCount": 0,
            "valType": "discrete",
            "typeNum": 58,
            "samples": [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0
            ],
            "averageValue": 28.583,
            "maximumValue": 58.0,
            "minimumValue": 1.0,
            "sampleVariance": 198.82
        },
        "column3": {
            "originColumnName": "District Code",
            "columnDescription": "District Code",
            "dataFormat": "integer",
            "size": 9986,
            "emptyValueCount": 0,
            "valType": "discrete",
            "typeNum": 1012,
            "samples": [
                61440.0,
                69633.0,
                67587.0,
                65540.0,
                10249.0,
                69641.0,
                67595.0,
                73742.0,
                63503.0,
                10256.0
            ],
            "averageValue": 65651.53,
            "maximumValue": 76901.0,
            "minimumValue": 10017.0,
            "sampleVariance": 122258200.87
        },
        "column4": {
            "originColumnName": "School Code ",
            "columnDescription": "School Code",
            "dataFormat": "integer",
            "size": 9986,
            "emptyValueCount": 0,
            "valType": "continuous",
            "samples": [
                6060099.0,
                6046577.0,
                6098297.0,
                6020358.0,
                6120729.0
            ],
            "averageValue": 4336056.094,
            "maximumValue": 9010745.0,
            "minimumValue": 0.0,
            "sampleVariance": 6070540262677.16
        },
        "column5": {
            "originColumnName": "County Name",
            "columnDescription": "County Code ",
            "dataFormat": "text",
            "size": 9986,
            "emptyValueCount": 0,
            "valType": "discrete",
            "typeNum": 58,
            "samples": [
                "Lassen",
                "Fresno",
                "Nevada",
                "Amador",
                "Mendocino",
                "Kings",
                "Ventura",
                "Marin",
                "Glenn",
                "San Francisco"
            ]
        },
        "column6": {
            "originColumnName": "District Name ",
            "columnDescription": "District Name ",
            "dataFormat": "text",
            "size": 9986,
            "emptyValueCount": 0,
            "valType": "discrete",
            "typeNum": 1000,
            "samples": [
                "Reed Union Elementary",
                "Bear Valley Unified",
                "Forestville Union Elementary",
                "Los Nietos",
                "Southside Elementary",
                "Oxnard",
                "Brisbane Elementary",
                "Glenn County Office of Education",
                "Lammersville Joint Unified",
                "Sunnyvale"
            ]
        },
        "column7": {
            "originColumnName": "School Name",
            "columnDescription": "School Name ",
            "dataFormat": "text",
            "size": 9986,
            "emptyValueCount": 0,
            "valType": "continuous",
            "samples": [
                "Elliott Ranch Elementary",
                "Allen Avenue Elementary",
                "Buchser Middle"
            ]
        },
        "column8": {
            "originColumnName": "District Type",
            "columnDescription": "District Type",
            "dataFormat": "text",
            "size": 9986,
            "emptyValueCount": 0,
            "valType": "discrete",
            "typeNum": 8,
            "samples": [
                "Elementary School District",
                "State Special Schools",
                "High School District",
                "Statewide Benefit Charter",
                "Non-School Locations",
                "County Office of Education (COE)",
                "State Board of Education",
                "Unified School District"
            ]
        },
        "column9": {
            "originColumnName": "School Type ",
            "columnDescription": "School Type ",
            "dataFormat": "text",
            "size": 9986,
            "emptyValueCount": 45,
            "valType": "discrete",
            "typeNum": 17,
            "samples": [
                "K-12 Schools (Public)",
                "State Special Schools",
                "Alternative Schools of Choice",
                "Intermediate/Middle Schools (Public)",
                "District Community Day Schools",
                "Opportunity Schools",
                "High Schools In 1 School Dist. (Public)",
                "High Schools (Public)",
                "Junior High Schools (Public)",
                "Special Education Schools (Public)"
            ]
        },
        "column10": {
            "originColumnName": "Educational Option Type",
            "columnDescription": "Educational Option Type",
            "dataFormat": "text",
            "size": 9986,
            "emptyValueCount": 45,
            "valType": "discrete",
            "typeNum": 12,
            "samples": [
                "Alternative School of Choice",
                "Community Day School",
                "State Special School",
                "Juvenile Court School",
                "Special Education School",
                "Continuation School",
                "District Special Education Consortia School",
                "Traditional",
                "Youth Authority School",
                "Home and Hospital"
            ]
        },
        "column11": {
            "originColumnName": "NSLP Provision Status",
            "columnDescription": "NSLP Provision Status",
            "dataFormat": "text",
            "size": 9986,
            "emptyValueCount": 8139,
            "valType": "discrete",
            "typeNum": 7,
            "samples": [
                "Provision 3",
                "Multiple Provision Types",
                "Provision 2",
                "CEP",
                "Lunch Provision 2",
                "Breakfast Provision 2",
                "Provision 1"
            ]
        },
        "column12": {
            "originColumnName": "Charter School (Y/N)",
            "columnDescription": "Charter School (Y/N)",
            "dataFormat": "integer",
            "valueDescription": "0: N;\n1: Y",
            "size": 9986,
            "emptyValueCount": 45,
            "valType": "discrete",
            "typeNum": 2,
            "samples": [
                0.0,
                1.0
            ],
            "averageValue": 0.117,
            "maximumValue": 1.0,
            "minimumValue": 0.0,
            "sampleVariance": 0.1
        },
        "column13": {
            "originColumnName": "Charter School Number",
            "columnDescription": "Charter School Number",
            "dataFormat": "text",
            "size": 9986,
            "emptyValueCount": 8819,
            "valType": "continuous",
            "samples": [
                "0252",
                "1501",
                "0173",
                "1257",
                "0848"
            ]
        },
        "column14": {
            "originColumnName": "Charter Funding Type",
            "columnDescription": "Charter Funding Type",
            "dataFormat": "text",
            "size": 9986,
            "emptyValueCount": 8819,
            "valType": "discrete",
            "typeNum": 3,
            "samples": [
                "Not in CS funding model",
                "Directly funded",
                "Locally funded"
            ]
        },
        "column15": {
            "originColumnName": "IRC",
            "dataFormat": "integer",
            "valueDescription": "Not useful",
            "size": 9986,
            "emptyValueCount": 45,
            "valType": "discrete",
            "typeNum": 2,
            "samples": [
                0.0,
                1.0
            ],
            "averageValue": 0.073,
            "maximumValue": 1.0,
            "minimumValue": 0.0,
            "sampleVariance": 0.07
        },
        "column16": {
            "originColumnName": "Low Grade",
            "columnDescription": "Low Grade",
            "dataFormat": "text",
            "size": 9986,
            "emptyValueCount": 0,
            "valType": "discrete",
            "typeNum": 15,
            "samples": [
                "K",
                "1",
                "7",
                "2",
                "12",
                "6",
                "Adult",
                "8",
                "4",
                "11"
            ]
        },
        "column17": {
            "originColumnName": "High Grade",
            "columnDescription": "High Grade",
            "dataFormat": "text",
            "size": 9986,
            "emptyValueCount": 0,
            "valType": "discrete",
            "typeNum": 17,
            "samples": [
                "7",
                "12",
                "2",
                "1",
                "K",
                "Post Secondary",
                "6",
                "Adult",
                "8",
                "4"
            ]
        },
        "column18": {
            "originColumnName": "Enrollment (K-12)",
            "columnDescription": "Enrollment (K-12)",
            "dataFormat": "real",
            "valueDescription": "commonsense evidence:\n\nK-12: 1st grade - 12nd grade ",
            "size": 9986,
            "emptyValueCount": 0,
            "valType": "discrete",
            "typeNum": 1882,
            "samples": [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0
            ],
            "averageValue": 620.826,
            "maximumValue": 5333.0,
            "minimumValue": 1.0,
            "sampleVariance": 290401.32
        },
        "column19": {
            "originColumnName": "Free Meal Count (K-12)",
            "columnDescription": "Free Meal Count (K-12)",
            "dataFormat": "real",
            "valueDescription": "commonsense evidence:\n\neligible free rate = Free Meal Count / Enrollment",
            "size": 9986,
            "emptyValueCount": 56,
            "valType": "discrete",
            "typeNum": 1216,
            "samples": [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0
            ],
            "averageValue": 312.004,
            "maximumValue": 3927.0,
            "minimumValue": 1.0,
            "sampleVariance": 98391.81
        },
        "column20": {
            "originColumnName": "Percent (%) Eligible Free (K-12)",
            "dataFormat": "real",
            "size": 9986,
            "emptyValueCount": 56,
            "valType": "continuous",
            "samples": [
                0.829248366013072,
                0.172697368421053,
                0.868898186889819,
                0.0759493670886076,
                0.60948905109489
            ],
            "averageValue": 0.53,
            "maximumValue": 1.0,
            "minimumValue": 0.0017605633802816,
            "sampleVariance": 0.07
        },
        "column21": {
            "originColumnName": "FRPM Count (K-12)",
            "columnDescription": "Free or Reduced Price Meal Count (K-12)",
            "dataFormat": "real",
            "valueDescription": "commonsense evidence:\n\neligible FRPM rate = FRPM / Enrollment",
            "size": 9986,
            "emptyValueCount": 50,
            "valType": "discrete",
            "typeNum": 1361,
            "samples": [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0
            ],
            "averageValue": 365.48,
            "maximumValue": 4419.0,
            "minimumValue": 1.0,
            "sampleVariance": 129410.51
        },
        "column22": {
            "originColumnName": "Percent (%) Eligible FRPM (K-12)",
            "dataFormat": "real",
            "size": 9986,
            "emptyValueCount": 50,
            "valType": "continuous",
            "samples": [
                0.881818181818182,
                0.764705882352941,
                0.739894551845343,
                0.728365384615385,
                0.340136054421769
            ],
            "averageValue": 0.612,
            "maximumValue": 1.0,
            "minimumValue": 0.0022050716648291,
            "sampleVariance": 0.08
        },
        "column23": {
            "originColumnName": "Enrollment (Ages 5-17)",
            "columnDescription": "Enrollment (Ages 5-17)",
            "dataFormat": "real",
            "size": 9986,
            "emptyValueCount": 14,
            "valType": "discrete",
            "typeNum": 1845,
            "samples": [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0
            ],
            "averageValue": 605.325,
            "maximumValue": 5271.0,
            "minimumValue": 1.0,
            "sampleVariance": 275665.89
        },
        "column24": {
            "originColumnName": "Free Meal Count (Ages 5-17)",
            "columnDescription": "Free Meal Count (Ages 5-17)",
            "dataFormat": "real",
            "valueDescription": "commonsense evidence:\n\neligible free rate = Free Meal Count / Enrollment",
            "size": 9986,
            "emptyValueCount": 78,
            "valType": "discrete",
            "typeNum": 1205,
            "samples": [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0
            ],
            "averageValue": 304.032,
            "maximumValue": 3864.0,
            "minimumValue": 1.0,
            "sampleVariance": 92724.85
        },
        "column25": {
            "originColumnName": " Percent (%) Eligible Free (Ages 5-17)",
            "dataFormat": "real",
            "size": 9986,
            "emptyValueCount": 78,
            "valType": "continuous",
            "samples": [
                0.131524008350731,
                0.708586296617519,
                0.50197628458498,
                0.242510699001427,
                0.400423728813559
            ],
            "averageValue": 0.532,
            "maximumValue": 1.0,
            "minimumValue": 0.0017985611510791,
            "sampleVariance": 0.07
        },
        "column26": {
            "originColumnName": "FRPM Count (Ages 5-17)",
            "dataFormat": "real",
            "size": 9986,
            "emptyValueCount": 72,
            "valType": "discrete",
            "typeNum": 1330,
            "samples": [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0
            ],
            "averageValue": 356.514,
            "maximumValue": 4347.0,
            "minimumValue": 1.0,
            "sampleVariance": 122187.7
        },
        "column27": {
            "originColumnName": "Percent (%) Eligible FRPM (Ages 5-17)",
            "dataFormat": "real",
            "size": 9986,
            "emptyValueCount": 72,
            "valType": "continuous",
            "samples": [
                0.450643776824034,
                0.898850574712644,
                0.966981132075472,
                0.248,
                0.175595238095238
            ],
            "averageValue": 0.615,
            "maximumValue": 1.0,
            "minimumValue": 0.0022050716648291,
            "sampleVariance": 0.08
        },
        "column28": {
            "originColumnName": "2013-14 CALPADS Fall 1 Certification Status",
            "columnDescription": "2013-14 CALPADS Fall 1 Certification Status",
            "dataFormat": "integer",
            "size": 9986,
            "emptyValueCount": 0,
            "valType": "discrete",
            "typeNum": 1,
            "samples": [
                1.0
            ],
            "averageValue": 1.0,
            "maximumValue": 1.0,
            "minimumValue": 1.0,
            "sampleVariance": 0.0
        }
    },
    "table1": {
        "tableName": "satscores.csv",
        "column0": {
            "originColumnName": "cds",
            "columnDescription": "California Department Schools",
            "dataFormat": "text",
            "size": 2269,
            "emptyValueCount": 0,
            "valType": "continuous",
            "samples": [
                49707060129981.0,
                13632141337609.0,
                54105460000000.0,
                10621580000000.0,
                19756970000000.0
            ],
            "averageValue": 29139356956040.285,
            "maximumValue": 58727695838305.0,
            "minimumValue": 1100170000000.0,
            "sampleVariance": 2.0973753251701384e+26
        },
        "column1": {
            "originColumnName": "rtype",
            "columnDescription": "rtype",
            "dataFormat": "text",
            "valueDescription": "unuseful",
            "size": 2269,
            "emptyValueCount": 0,
            "valType": "discrete",
            "typeNum": 2,
            "samples": [
                "S",
                "D"
            ]
        },
        "column2": {
            "originColumnName": "sname",
            "fullColumnName": "school name",
            "columnDescription": "school name",
            "dataFormat": "text",
            "size": 2269,
            "emptyValueCount": 520,
            "valType": "continuous",
            "samples": [
                "California Military Institute",
                "Westmont High",
                "University Preparatory"
            ]
        },
        "column3": {
            "originColumnName": "dname",
            "fullColumnName": "district name",
            "columnDescription": "district segment",
            "dataFormat": "text",
            "size": 2269,
            "emptyValueCount": 0,
            "valType": "discrete",
            "typeNum": 520,
            "samples": [
                "Bear Valley Unified",
                "Glenn County Office of Education",
                "Waterford Unified",
                "Butte County Office of Education",
                "Beverly Hills Unified",
                "Keyes Union",
                "Lodi Unified",
                "Merced County Office of Education",
                "Westside Elementary",
                "Sierra Sands Unified"
            ]
        },
        "column4": {
            "originColumnName": "cname",
            "fullColumnName": "county name",
            "columnDescription": "county name",
            "dataFormat": "text",
            "size": 2269,
            "emptyValueCount": 0,
            "valType": "discrete",
            "typeNum": 57,
            "samples": [
                "Lassen",
                "Fresno",
                "Nevada",
                "Amador",
                "Mendocino",
                "Kings",
                "Ventura",
                "Marin",
                "Glenn",
                "San Francisco"
            ]
        },
        "column5": {
            "originColumnName": "enroll12",
            "fullColumnName": "enrollment (1st-12nd grade)",
            "columnDescription": "enrollment (1st-12nd grade)",
            "dataFormat": "integer",
            "size": 2269,
            "emptyValueCount": 0,
            "valType": "continuous",
            "samples": [
                240.0,
                111.0,
                241.0,
                1080.0,
                1587.0
            ],
            "averageValue": 419.519,
            "maximumValue": 43324.0,
            "minimumValue": 0.0,
            "sampleVariance": 1303151.86
        },
        "column6": {
            "originColumnName": "NumTstTakr",
            "fullColumnName": "Number of Test Takers",
            "columnDescription": "Number of Test Takers in this school",
            "dataFormat": "integer",
            "valueDescription": "number of test takers in each school",
            "size": 2269,
            "emptyValueCount": 0,
            "valType": "discrete",
            "typeNum": 547,
            "samples": [
                0.0,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0
            ],
            "averageValue": 185.098,
            "maximumValue": 24305.0,
            "minimumValue": 0.0,
            "sampleVariance": 355957.83
        },
        "column7": {
            "originColumnName": "AvgScrRead",
            "fullColumnName": "average scores in Reading",
            "columnDescription": "average scores in Reading",
            "dataFormat": "integer",
            "valueDescription": "average scores in Reading",
            "size": 2269,
            "emptyValueCount": 596,
            "valType": "discrete",
            "typeNum": 269,
            "samples": [
                512.0,
                513.0,
                514.0,
                515.0,
                516.0,
                517.0,
                518.0,
                519.0,
                520.0,
                521.0
            ],
            "averageValue": 479.699,
            "maximumValue": 653.0,
            "minimumValue": 308.0,
            "sampleVariance": 3338.74
        },
        "column8": {
            "originColumnName": "AvgScrMath",
            "fullColumnName": "average scores in Math",
            "columnDescription": "average scores in Math",
            "dataFormat": "integer",
            "valueDescription": "average scores in Math",
            "size": 2269,
            "emptyValueCount": 596,
            "valType": "discrete",
            "typeNum": 295,
            "samples": [
                512.0,
                513.0,
                514.0,
                515.0,
                516.0,
                517.0,
                518.0,
                519.0,
                520.0,
                521.0
            ],
            "averageValue": 484.461,
            "maximumValue": 699.0,
            "minimumValue": 289.0,
            "sampleVariance": 3915.26
        },
        "column9": {
            "originColumnName": "AvgScrWrite",
            "fullColumnName": "average scores in writing",
            "columnDescription": "average scores in writing",
            "dataFormat": "integer",
            "valueDescription": "average scores in writing",
            "size": 2269,
            "emptyValueCount": 596,
            "valType": "discrete",
            "typeNum": 267,
            "samples": [
                512.0,
                513.0,
                514.0,
                515.0,
                516.0,
                517.0,
                518.0,
                519.0,
                520.0,
                521.0
            ],
            "averageValue": 472.529,
            "maximumValue": 671.0,
            "minimumValue": 312.0,
            "sampleVariance": 3116.68
        },
        "column10": {
            "originColumnName": "NumGE1500",
            "fullColumnName": "Number of Test Takers Whose Total SAT Scores Are Greater or Equal to 1500",
            "columnDescription": "Number of Test Takers Whose Total SAT Scores Are Greater or Equal to 1500",
            "dataFormat": "integer",
            "valueDescription": "Number of Test Takers Whose Total SAT Scores Are Greater or Equal to 1500\n\ncommonsense evidence:\n\nExcellence Rate = NumGE1500 / NumTstTakr",
            "size": 2269,
            "emptyValueCount": 596,
            "valType": "discrete",
            "typeNum": 368,
            "samples": [
                0.0,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0
            ],
            "averageValue": 111.079,
            "maximumValue": 5837.0,
            "minimumValue": 0.0,
            "sampleVariance": 53106.46
        }
    },
    "table2": {
        "tableName": "schools.csv",
        "column0": {
            "originColumnName": "CDSCode",
            "fullColumnName": "CDSCode",
            "columnDescription": "CDSCode",
            "dataFormat": "text",
            "size": 17686,
            "emptyValueCount": 0,
            "valType": "continuous",
            "samples": [
                19647330124412.0,
                50711670000000.0,
                34673066119796.0,
                54718606053904.0,
                7617540730408.0
            ],
            "averageValue": 29117072973703.785,
            "maximumValue": 58727695838305.0,
            "minimumValue": 1100170000000.0,
            "sampleVariance": 2.095918712828605e+26
        },
        "column1": {
            "originColumnName": "NCESDist",
            "fullColumnName": "National Center for Educational Statistics school district identification number",
            "columnDescription": "This field represents the 7-digit National Center for Educational Statistics (NCES) school district identification number. The first 2 digits identify the state and the last 5 digits identify the school district. Combined, they make a unique 7-digit ID for each school district.",
            "dataFormat": "text",
            "size": 17686,
            "emptyValueCount": 1030,
            "valType": "discrete",
            "typeNum": 1193,
            "samples": [
                614400.0,
                634880.0,
                600064.0,
                600067.0,
                638980.0,
                600068.0,
                600070.0,
                600071.0,
                612360.0,
                600073.0
            ],
            "averageValue": 627945.395,
            "maximumValue": 691137.0,
            "minimumValue": 600001.0,
            "sampleVariance": 535949991.73
        },
        "column2": {
            "originColumnName": "NCESSchool",
            "fullColumnName": "National Center for Educational Statistics school identification number",
            "columnDescription": "This field represents the 5-digit NCES school identification number. The NCESSchool combined with the NCESDist form a unique 12-digit ID for each school.",
            "dataFormat": "text",
            "size": 17686,
            "emptyValueCount": 5040,
            "valType": "continuous",
            "samples": [
                731.0,
                5900.0,
                6970.0,
                13541.0,
                7936.0
            ],
            "averageValue": 6888.113,
            "maximumValue": 14003.0,
            "minimumValue": 1.0,
            "sampleVariance": 17095503.66
        },
        "column3": {
            "originColumnName": "StatusType",
            "columnDescription": "This field identifies the status of the district. ",
            "dataFormat": "text",
            "valueDescription": "Definitions of the valid status types are listed below:\n\u00b7       Active: The district is in operation and providing instructional services.\n\u00b7       Closed: The district is not in operation and no longer providing instructional services.\n\u00b7       Merged: The district has combined with another district or districts.\n\u00b7       Pending: The district has not opened for operation and instructional services yet, but plans to open within the next 9\u201312 months.",
            "size": 17686,
            "emptyValueCount": 0,
            "valType": "discrete",
            "typeNum": 4,
            "samples": [
                "Active",
                "Pending",
                "Merged",
                "Closed"
            ]
        },
        "column4": {
            "originColumnName": "County",
            "columnDescription": "County name",
            "dataFormat": "text",
            "size": 17686,
            "emptyValueCount": 0,
            "valType": "discrete",
            "typeNum": 58,
            "samples": [
                "Lassen",
                "Fresno",
                "Nevada",
                "Amador",
                "Mendocino",
                "Kings",
                "Ventura",
                "Marin",
                "Glenn",
                "San Francisco"
            ]
        },
        "column5": {
            "originColumnName": "District",
            "columnDescription": "District",
            "dataFormat": "text",
            "size": 17686,
            "emptyValueCount": 0,
            "valType": "discrete",
            "typeNum": 1411,
            "samples": [
                "Reed Union Elementary",
                "Bear Valley Unified",
                "Southside Elementary",
                "Oxnard",
                "Glenn County Office of Education",
                "First Covenant Church",
                "Upland Elementary",
                "Keyes Union",
                "Red Bluff Union Elementary",
                "Sierra Sands Unified"
            ]
        },
        "column6": {
            "originColumnName": "School",
            "columnDescription": "School",
            "dataFormat": "text",
            "size": 17686,
            "emptyValueCount": 1369,
            "valType": "continuous",
            "samples": [
                "Sarah Anthony",
                "Jefferson Middle",
                "Coast Union High"
            ]
        },
        "column7": {
            "originColumnName": "Street",
            "columnDescription": "Street",
            "dataFormat": "text",
            "size": 17686,
            "emptyValueCount": 294,
            "valType": "continuous",
            "samples": [
                "3701 Lester Road",
                "39925 Harveston Drive",
                "19010 Napa Street",
                "707 Morse Avenue",
                "4410 Old Farm Road"
            ]
        },
        "column8": {
            "originColumnName": "StreetAbr",
            "fullColumnName": "street address ",
            "columnDescription": "The abbreviated street address of the school, district, or administrative authority\u2019s physical location.",
            "dataFormat": "text",
            "valueDescription": "The abbreviated street address of the school, district, or administrative authority\u2019s physical location. Note: Some records (primarily records of closed or retired schools) may not have data in this field.",
            "size": 17686,
            "emptyValueCount": 294,
            "valType": "continuous",
            "samples": [
                "Rt. 7, Box 300",
                "2800 West Fruitvale Ave.",
                "3934 Broadway Rd.",
                "2000 Railroad Ave.",
                "15405 Sunset Ave."
            ]
        },
        "column9": {
            "originColumnName": "City",
            "columnDescription": "City",
            "dataFormat": "text",
            "size": 17686,
            "emptyValueCount": 293,
            "valType": "discrete",
            "typeNum": 1165,
            "samples": [
                "Leona Valley",
                "Lancaster",
                "Los Nietos",
                "Dublin",
                "Oxnard",
                "Fort Irwin",
                "Helm",
                "Rancho Cucamonga",
                "Friant",
                "Sunnyvale"
            ]
        },
        "column10": {
            "originColumnName": "Zip",
            "columnDescription": "Zip",
            "dataFormat": "text",
            "size": 17686,
            "emptyValueCount": 293,
            "valType": "continuous",
            "samples": [
                "95037",
                "94044",
                "91762",
                "95959",
                "92806-3767"
            ]
        },
        "column11": {
            "originColumnName": "State",
            "columnDescription": "State",
            "dataFormat": "text",
            "size": 17686,
            "emptyValueCount": 293,
            "valType": "discrete",
            "typeNum": 1,
            "samples": [
                "CA"
            ]
        },
        "column12": {
            "originColumnName": "MailStreet",
            "columnDescription": "MailStreet",
            "dataFormat": "text",
            "valueDescription": "The unabbreviated mailing address of the school, district, or administrative authority. Note: 1) Some entities (primarily closed or retired schools) may not have data in this field; 2) Many active entities have not provided a mailing street address. For your convenience we have filled the unpopulated MailStreet cells with Street data.",
            "size": 17686,
            "emptyValueCount": 292,
            "valType": "continuous",
            "samples": [
                "25151 Pradera Drive",
                "515 West San Antonio Drive",
                "2253 Fifth Street",
                "801 High Street",
                "4121 Mackin Woods Lane"
            ]
        },
        "column13": {
            "originColumnName": "MailStrAbr",
            "fullColumnName": "mailing street address ",
            "dataFormat": "text",
            "valueDescription": "the abbreviated mailing street address of the school, district, or administrative authority.Note: Many active entities have not provided a mailing street address. For your convenience we have filled the unpopulated MailStrAbr cells with StreetAbr data.",
            "size": 17686,
            "emptyValueCount": 292,
            "valType": "continuous",
            "samples": [
                "4400 Coliseum St.",
                "200 Monte Vista Ave.",
                "2601 May Rd.",
                "540 G St.",
                "257 Grand Army"
            ]
        },
        "column14": {
            "originColumnName": "MailCity",
            "fullColumnName": "mailing city",
            "dataFormat": "text",
            "valueDescription": "The city associated with the mailing address of the school, district, or administrative authority. Note: Many entities have not provided a mailing address city. For your convenience we have filled the unpopulated MailCity cells with City data.",
            "size": 17686,
            "emptyValueCount": 292,
            "valType": "discrete",
            "typeNum": 1132,
            "samples": [
                "Leona Valley",
                "Lancaster",
                "Los Nietos",
                "Dublin",
                "Oxnard",
                "Fort Irwin",
                "Helm",
                "Rancho Cucamonga",
                "Friant",
                "Sunnyvale"
            ]
        },
        "column15": {
            "originColumnName": "MailZip",
            "fullColumnName": "mailing zip ",
            "dataFormat": "text",
            "valueDescription": "The zip code associated with the mailing address of the school, district, or administrative authority. Note: Many entities have not provided a mailing address zip code. For your convenience we have filled the unpopulated MailZip cells with Zip data.",
            "size": 17686,
            "emptyValueCount": 292,
            "valType": "continuous",
            "samples": [
                "93203",
                "95626-9580",
                "92555-3802",
                "92119",
                "93405-8003"
            ]
        },
        "column16": {
            "originColumnName": "MailState",
            "fullColumnName": "mailing state",
            "dataFormat": "text",
            "valueDescription": "The state within the mailing address. For your convenience we have filled the unpopulated MailState cells with State data.",
            "size": 17686,
            "emptyValueCount": 292,
            "valType": "discrete",
            "typeNum": 1,
            "samples": [
                "CA"
            ]
        },
        "column17": {
            "originColumnName": "Phone",
            "columnDescription": "Phone",
            "dataFormat": "text",
            "size": 17686,
            "emptyValueCount": 5969,
            "valType": "continuous",
            "samples": [
                "(310) 970-9914",
                "(714) 730-7348",
                "(559) 490-4290",
                "(805) 498-6748",
                "(949) 515-6975"
            ]
        },
        "column18": {
            "originColumnName": "Ext",
            "fullColumnName": "extension",
            "columnDescription": "The phone number extension of the school, district, or administrative authority.",
            "dataFormat": "text",
            "valueDescription": "The phone number extension of the school, district, or administrative authority.",
            "size": 17686,
            "emptyValueCount": 17146,
            "valType": "continuous",
            "samples": [
                5238.0,
                220.0,
                206.0,
                2551.0,
                551.0
            ],
            "averageValue": 6868.009,
            "maximumValue": 630100.0,
            "minimumValue": 0.0,
            "sampleVariance": 904572837.81
        },
        "column19": {
            "originColumnName": "Website",
            "columnDescription": "The website address of the school, district, or administrative authority.",
            "dataFormat": "text",
            "valueDescription": "The website address of the school, district, or administrative authority.",
            "size": 17686,
            "emptyValueCount": 10722,
            "valType": "continuous",
            "samples": [
                "http://ro.centralunified.org/",
                "www.smcoe.org",
                "www.iusd.org/eclc"
            ]
        },
        "column20": {
            "originColumnName": "OpenDate",
            "columnDescription": "The date the school opened.",
            "dataFormat": "date",
            "size": 17686,
            "emptyValueCount": 1369,
            "valType": "discrete",
            "typeNum": 1406,
            "samples": [
                "1990-07-09",
                "1994-09-08",
                "1974-09-01",
                "2003-08-14",
                "1998-10-16",
                "1998-08-31",
                "2013-08-28",
                "2003-11-04",
                "2000-08-22",
                "1991-08-28"
            ]
        },
        "column21": {
            "originColumnName": "ClosedDate",
            "columnDescription": "The date the school closed.",
            "dataFormat": "date",
            "size": 17686,
            "emptyValueCount": 11992,
            "valType": "discrete",
            "typeNum": 899,
            "samples": [
                "2007-03-02",
                "2014-07-25",
                "2009-06-12",
                "1998-01-13",
                "2009-06-16",
                "2002-09-16",
                "1989-03-22",
                "2003-09-05",
                "2002-10-24",
                "2006-07-12"
            ]
        },
        "column22": {
            "originColumnName": "Charter",
            "columnDescription": "This field identifies a charter school. ",
            "dataFormat": "integer",
            "valueDescription": "The field is coded as follows:\n\n\u00b7\u00a0\u00a0\u00a0\u00a0\u00a0\u00a0 1 = The school is a charter\n\n\u00b7\u00a0\u00a0\u00a0\u00a0\u00a0\u00a0 0 = The school is not a charter",
            "size": 17686,
            "emptyValueCount": 1369,
            "valType": "discrete",
            "typeNum": 2,
            "samples": [
                0.0,
                1.0
            ],
            "averageValue": 0.106,
            "maximumValue": 1.0,
            "minimumValue": 0.0,
            "sampleVariance": 0.09
        },
        "column23": {
            "originColumnName": "CharterNum",
            "columnDescription": "The charter school number,",
            "dataFormat": "text",
            "valueDescription": "4-digit number assigned to a charter school.",
            "size": 17686,
            "emptyValueCount": 15885,
            "valType": "continuous",
            "samples": [
                "1109",
                "1100",
                "0850",
                "0425",
                "1451"
            ]
        },
        "column24": {
            "originColumnName": "FundingType",
            "columnDescription": "Indicates the charter school funding type",
            "dataFormat": "text",
            "valueDescription": "Values are as follows:\n\n\u00b7\u00a0\u00a0\u00a0\u00a0\u00a0\u00a0 Not in CS (California School) funding model\n\n\u00b7\u00a0\u00a0\u00a0\u00a0\u00a0\u00a0 Locally funded\n\n\u00b7\u00a0\u00a0\u00a0\u00a0\u00a0\u00a0 Directly funded",
            "size": 17686,
            "emptyValueCount": 16044,
            "valType": "discrete",
            "typeNum": 3,
            "samples": [
                "Not in CS funding model",
                "Directly funded",
                "Locally funded"
            ]
        },
        "column25": {
            "originColumnName": "DOC",
            "fullColumnName": "District Ownership Code",
            "columnDescription": "District Ownership Code",
            "dataFormat": "text",
            "valueDescription": "The District Ownership Code (DOC) is the numeric code used to identify the category of the Administrative Authority.\n\u2022       00 - County Office of Education\n\u2022       02 \u2013 State Board of Education\n\u2022       03 \u2013 Statewide Benefit Charter\n\u2022       31 \u2013 State Special Schools\n\u2022       34 \u2013 Non-school Location*\n\u2022       52 \u2013 Elementary School District\n\u2022       54 \u2013 Unified School District\n\u2022       56 \u2013 High School District\n\u2022       98 \u2013 Regional Occupational Center/Program (ROC/P)\ncommonsense evidence:\n*Only the California Education Authority has been included in the non-school location category.",
            "size": 17686,
            "emptyValueCount": 0,
            "valType": "discrete",
            "typeNum": 12,
            "samples": [
                0.0,
                34.0,
                98.0,
                2.0,
                99.0,
                3.0,
                42.0,
                52.0,
                54.0,
                56.0
            ],
            "averageValue": 48.549,
            "maximumValue": 99.0,
            "minimumValue": 0.0,
            "sampleVariance": 284.04
        },
        "column26": {
            "originColumnName": "DOCType",
            "fullColumnName": "The District Ownership Code Type",
            "columnDescription": "The District Ownership Code Type is the text description of the DOC category.",
            "dataFormat": "text",
            "valueDescription": "(See text values in DOC field description above)",
            "size": 17686,
            "emptyValueCount": 0,
            "valType": "discrete",
            "typeNum": 12,
            "samples": [
                "Administration Only",
                "State Special Schools",
                "Elementary School District",
                "High School District",
                "Community College District",
                "Statewide Benefit Charter",
                "Non-School Locations",
                "Joint Powers Authority (JPA)",
                "County Office of Education (COE)",
                "Regional Occupation Center/Program (ROC/P)"
            ]
        },
        "column27": {
            "originColumnName": "SOC",
            "fullColumnName": "School Ownership Code",
            "columnDescription": "The School Ownership Code is a numeric code used to identify the type of school.",
            "dataFormat": "text",
            "valueDescription": "\u2022      08 - Preschool      \n\u2022       09 \u2013 Special Education Schools (Public)\n\u2022      11 \u2013 Youth Authority Facilities (CEA)\n\u2022       13 \u2013 Opportunity Schools\n\u2022       14 \u2013 Juvenile Court Schools\n\u2022       15 \u2013 Other County or District Programs\n\u2022       31 \u2013 State Special Schools\n\u2022       60 \u2013 Elementary School (Public)\n\u2022       61 \u2013 Elementary School in 1 School District (Public)\n\u2022       62 \u2013 Intermediate/Middle Schools (Public)\n\u2022       63 \u2013 Alternative Schools of Choice\n\u2022       64 \u2013 Junior High Schools (Public)\n\u2022       65 \u2013 K-12 Schools (Public)\n\u2022       66 \u2013 High Schools (Public)\n\u2022       67 \u2013 High Schools in 1 School District (Public)\n\u2022       68 \u2013 Continuation High Schools\n\u2022       69 \u2013 District Community Day Schools\n\u2022       70 \u2013 Adult Education Centers\n\u2022       98 \u2013 Regional Occupational Center/Program (ROC/P)",
            "size": 17686,
            "emptyValueCount": 1369,
            "valType": "discrete",
            "typeNum": 20,
            "samples": [
                8.0,
                9.0,
                10.0,
                11.0,
                13.0,
                14.0,
                15.0,
                31.0,
                60.0,
                61.0
            ],
            "averageValue": 56.408,
            "maximumValue": 98.0,
            "minimumValue": 8.0,
            "sampleVariance": 307.81
        },
        "column28": {
            "originColumnName": "SOCType",
            "fullColumnName": "School Ownership Code Type",
            "columnDescription": "The School Ownership Code Type is the text description of the type of school.",
            "dataFormat": "text",
            "valueDescription": "The School Ownership Code Type is the text description of the type of school.",
            "size": 17686,
            "emptyValueCount": 1369,
            "valType": "discrete",
            "typeNum": 20,
            "samples": [
                "Opportunity Schools",
                "Intermediate/Middle Schools (Public)",
                "Adult Education Centers",
                "Preschool",
                "Other County Or District Programs",
                "Alternative Schools of Choice",
                "Special Education Schools (Public)",
                "Juvenile Court Schools",
                "County Community",
                "Junior High Schools (Public)"
            ]
        },
        "column29": {
            "originColumnName": "EdOpsCode",
            "fullColumnName": "Education Option Code",
            "columnDescription": "The Education Option Code is a short text description of the type of education offered.",
            "dataFormat": "text",
            "valueDescription": "\n\u2022      ALTSOC \u2013 Alternative School of Choice\n\u2022      COMM \u2013 County Community School\n\u2022       COMMDAY \u2013 Community Day School\n\u2022       CON \u2013 Continuation School\n\u2022       JUV \u2013 Juvenile Court School\n\u2022       OPP \u2013 Opportunity School\n\u2022       YTH \u2013 Youth Authority School\n\u2022       SSS \u2013 State Special School\n\u2022       SPEC \u2013 Special Education School\n\u2022       TRAD \u2013 Traditional\n\u2022       ROP \u2013 Regional Occupational Program\n\u2022       HOMHOS \u2013 Home and Hospital\n\u2022       SPECON \u2013 District Consortia Special Education School",
            "size": 17686,
            "emptyValueCount": 5711,
            "valType": "discrete",
            "typeNum": 13,
            "samples": [
                "ROP",
                "TRAD",
                "SPECON",
                "COMM",
                "ALTSOC",
                "JUV",
                "HOMHOS",
                "OPP",
                "SPEC",
                "YTH"
            ]
        },
        "column30": {
            "originColumnName": "EdOpsName",
            "fullColumnName": "Educational Option Name",
            "columnDescription": "Educational Option Name",
            "dataFormat": "text",
            "valueDescription": "The Educational Option Name is the long text description of the type of education being offered.",
            "size": 17686,
            "emptyValueCount": 5711,
            "valType": "discrete",
            "typeNum": 13,
            "samples": [
                "ROP",
                "Alternative School of Choice",
                "Community Day School",
                "State Special School",
                "Special Education School",
                "Juvenile Court School",
                "Continuation School",
                "District Special Education Consortia School",
                "Traditional",
                "Youth Authority School"
            ]
        },
        "column31": {
            "originColumnName": "EILCode",
            "fullColumnName": "Educational Instruction Level Code",
            "columnDescription": "The Educational Instruction Level Code is a short text description of the institution's type relative to the grade range served.",
            "dataFormat": "text",
            "valueDescription": "\u2022       A \u2013 Adult\n\u2022       ELEM \u2013 Elementary\n\u2022       ELEMHIGH \u2013 Elementary-High Combination\n\u2022       HS \u2013 High School\n\u2022       INTMIDJR \u2013 Intermediate/Middle/Junior High\n\u2022       PS \u2013 Preschool\n\u2022       UG \u2013 Ungraded",
            "size": 17686,
            "emptyValueCount": 1369,
            "valType": "discrete",
            "typeNum": 7,
            "samples": [
                "PS",
                "INTMIDJR",
                "HS",
                "ELEM",
                "UG",
                "A",
                "ELEMHIGH"
            ]
        },
        "column32": {
            "originColumnName": "EILName",
            "fullColumnName": "Educational Instruction Level Name ",
            "columnDescription": "The Educational Instruction Level Name is the long text description of the institution\u2019s type relative to the grade range served.",
            "dataFormat": "text",
            "valueDescription": "The Educational Instruction Level Name is the long text description of the institution\u2019s type relative to the grade range served.",
            "size": 17686,
            "emptyValueCount": 1369,
            "valType": "discrete",
            "typeNum": 7,
            "samples": [
                "High School",
                "Adult",
                "Ungraded",
                "Intermediate/Middle/Junior High",
                "Preschool",
                "Elementary",
                "Elementary-High Combination"
            ]
        },
        "column33": {
            "originColumnName": "GSoffered",
            "fullColumnName": "grade span offered",
            "columnDescription": "The grade span offered is the lowest grade and the highest grade offered or supported by the school, district, or administrative authority. This field might differ from the grade span served as reported in the most recent certified California Longitudinal Pupil Achievement (CALPADS) Fall 1 data collection.",
            "dataFormat": "text",
            "valueDescription": "For example XYZ School might display the following data:\n\nGSoffered = P\u2013Adult\n\nGSserved = K\u201312",
            "size": 17686,
            "emptyValueCount": 3882,
            "valType": "discrete",
            "typeNum": 94,
            "samples": [
                "K-8",
                "1-3",
                "5-12",
                "8-12",
                "6-12",
                "K-2",
                "2-4",
                "P-12",
                "10-12",
                "K-7"
            ]
        },
        "column34": {
            "originColumnName": "GSserved",
            "fullColumnName": "grade span served.",
            "columnDescription": "It is the lowest grade and the highest grade of student enrollment as reported in the most recent certified CALPADS Fall 1 data collection. Only K\u201312 enrollment is reported through CALPADS. This field may differ from the grade span offered.",
            "dataFormat": "text",
            "valueDescription": "commonsense evidence:\n\n1.\u00a0\u00a0\u00a0\u00a0 Only K\u201312 enrollment is reported through CALPADS\n\n2.\u00a0\u00a0\u00a0\u00a0 Note: Special programs at independent study, alternative education, and special education schools will often exceed the typical grade span for schools of that type",
            "size": 17686,
            "emptyValueCount": 5743,
            "valType": "discrete",
            "typeNum": 81,
            "samples": [
                "K-8",
                "1-3",
                "5-12",
                "8-12",
                "6-12",
                "K-2",
                "2-4",
                "10-12",
                "K-7",
                "7"
            ]
        },
        "column35": {
            "originColumnName": "Virtual",
            "columnDescription": "This field identifies the type of virtual instruction offered by the school. Virtual instruction is instruction in which students and teachers are separated by time and/or location, and interaction occurs via computers and/or telecommunications technologies. ",
            "dataFormat": "text",
            "valueDescription": "The field is coded as follows:\n\n\u00b7\u00a0\u00a0\u00a0\u00a0\u00a0\u00a0 F = Exclusively Virtual \u2013 The school has no physical building where students meet with each other or with teachers, all instruction is virtual.\n\n\u00b7\u00a0\u00a0\u00a0\u00a0\u00a0\u00a0 V = Primarily Virtual \u2013 The school focuses on a systematic program of virtual instruction but includes some physical meetings among students or with teachers.\n\n\u00b7\u00a0\u00a0\u00a0\u00a0\u00a0\u00a0 C = Primarily Classroom \u2013 The school offers virtual courses but virtual instruction is not the primary means of instruction.\n\n\u00b7\u00a0\u00a0\u00a0\u00a0\u00a0\u00a0 N = Not Virtual \u2013 The school does not offer any virtual instruction.\n\n\u00b7\u00a0\u00a0\u00a0\u00a0\u00a0\u00a0 P = Partial Virtual \u2013 The school offers some, but not all, instruction through virtual instruction. Note: This value was retired and replaced with the Primarily Virtual and Primarily Classroom values beginning with the 2016\u201317 school year.",
            "size": 17686,
            "emptyValueCount": 6868,
            "valType": "discrete",
            "typeNum": 3,
            "samples": [
                "N",
                "F",
                "P"
            ]
        },
        "column36": {
            "originColumnName": "Magnet",
            "columnDescription": "This field identifies whether a school is a magnet school and/or provides a magnet program. ",
            "dataFormat": "integer",
            "valueDescription": "The field is coded as follows:\n\n\u00b7\u00a0\u00a0\u00a0\u00a0\u00a0\u00a0 1 = Magnet - The school is a magnet school and/or offers a magnet program.\n\n\u00b7\u00a0\u00a0\u00a0\u00a0\u00a0\u00a0 0 = Not Magnet - The school is not a magnet school and/or does not offer a magnet program.\n\ncommonsense evidence:\n\nNote: Preschools and adult education centers do not contain a magnet school indicator.",
            "size": 17686,
            "emptyValueCount": 7076,
            "valType": "discrete",
            "typeNum": 2,
            "samples": [
                0.0,
                1.0
            ],
            "averageValue": 0.049,
            "maximumValue": 1.0,
            "minimumValue": 0.0,
            "sampleVariance": 0.05
        },
        "column37": {
            "originColumnName": "Latitude",
            "columnDescription": "The angular distance (expressed in degrees) between the location of the school, district, or administrative authority and the equator measured north to south.",
            "dataFormat": "real",
            "valueDescription": "The angular distance (expressed in degrees) between the location of the school, district, or administrative authority and the equator measured north to south.",
            "size": 17686,
            "emptyValueCount": 4823,
            "valType": "continuous",
            "samples": [
                38.735106,
                36.337837,
                37.2041,
                34.13204,
                34.125125
            ],
            "averageValue": 36.008,
            "maximumValue": 44.219305,
            "minimumValue": 32.547737,
            "sampleVariance": 5.22
        },
        "column38": {
            "originColumnName": "Longitude",
            "columnDescription": "The angular distance (expressed in degrees) between the location of the school, district, or administrative authority and the prime meridian (Greenwich, England) measured from west to east.",
            "dataFormat": "real",
            "valueDescription": "The angular distance (expressed in degrees) between the location of the school, district, or administrative authority and the prime meridian (Greenwich, England) measured from west to east.",
            "size": 17686,
            "emptyValueCount": 4823,
            "valType": "continuous",
            "samples": [
                -119.77373,
                -118.84449,
                -121.33475,
                -121.25523,
                -122.39938
            ],
            "averageValue": -119.694,
            "maximumValue": -83.781133,
            "minimumValue": -124.28481,
            "sampleVariance": 4.6
        },
        "column39": {
            "originColumnName": "AdmFName1",
            "fullColumnName": "administrator's first name",
            "columnDescription": "administrator's first name",
            "dataFormat": "text",
            "valueDescription": "The superintendent\u2019s or principal\u2019s first name.\n\ncommonsense evidence:\n\nOnly active and pending districts and schools will display administrator information, if applicable.",
            "size": 17686,
            "emptyValueCount": 5986,
            "valType": "discrete",
            "typeNum": 2327,
            "samples": [
                "Johnny",
                "Santa",
                "Adriana",
                "Beatriz",
                "Ildefonso",
                "Dario",
                "Adaina",
                "Rita",
                "Steven",
                "Janeen"
            ]
        },
        "column40": {
            "originColumnName": "AdmLName1",
            "fullColumnName": "administrator's last name",
            "columnDescription": "administrator's last name",
            "dataFormat": "text",
            "valueDescription": "The superintendent\u2019s or principal\u2019s last name.\n\ncommonsense evidence:\nOnly active and pending districts and schools will display administrator information, if applicable.",
            "size": 17686,
            "emptyValueCount": 5986,
            "valType": "continuous",
            "samples": [
                "Drechsler",
                "Shriner",
                "Ferry",
                "Aramburo",
                "Johnson"
            ]
        },
        "column41": {
            "originColumnName": "AdmEmail1",
            "fullColumnName": "administrator's email address",
            "columnDescription": "administrator's email address",
            "dataFormat": "text",
            "valueDescription": "The superintendent\u2019s or principal\u2019s email address.\n\ncommonsense evidence:\n\nOnly active and pending districts and schools will display administrator information, if applicable.",
            "size": 17686,
            "emptyValueCount": 6012,
            "valType": "continuous",
            "samples": [
                "jales@saratogausd.org",
                "rmorris@kingsburghigh.com",
                "lisacooper@ccusd.org"
            ]
        },
        "column42": {
            "originColumnName": "AdmFName2",
            "dataFormat": "text",
            "valueDescription": "SAME as 1",
            "size": 17686,
            "emptyValueCount": 17255,
            "valType": "continuous",
            "samples": [
                "Janet",
                "Rachel",
                "Todd",
                "Vicki",
                "Merry"
            ]
        },
        "column43": {
            "originColumnName": "AdmLName2",
            "dataFormat": "text",
            "size": 17686,
            "emptyValueCount": 17255,
            "valType": "continuous",
            "samples": [
                "Syverson",
                "Tuter",
                "Torres",
                "Rizo",
                "Hanna"
            ]
        },
        "column44": {
            "originColumnName": "AdmEmail2",
            "dataFormat": "text",
            "size": 17686,
            "emptyValueCount": 17262,
            "valType": "continuous",
            "samples": [
                "cynthia.orr@cvesd.org",
                "sarah.sanchez@kippbayarea.org",
                "gmoraga@lbschools.net"
            ]
        },
        "column45": {
            "originColumnName": "AdmFName3",
            "dataFormat": "text",
            "valueDescription": "not useful",
            "size": 17686,
            "emptyValueCount": 17644,
            "valType": "continuous",
            "samples": [
                "Dustin",
                "Joseph",
                "Joy",
                "Alex",
                "Bridget"
            ]
        },
        "column46": {
            "originColumnName": "AdmLName3",
            "dataFormat": "text",
            "valueDescription": "not useful",
            "size": 17686,
            "emptyValueCount": 17644,
            "valType": "continuous",
            "samples": [
                "Wolk",
                "Martarano",
                "Zerpoli",
                "Rivera",
                "Yount"
            ]
        },
        "column47": {
            "originColumnName": "AdmEmail3",
            "dataFormat": "text",
            "valueDescription": "not useful",
            "size": 17686,
            "emptyValueCount": 17644,
            "valType": "continuous",
            "samples": [
                "bwolk@husd.com",
                "bmandelbaum@bcoe.org",
                "rlemmon@ghsd.k12.ca.us"
            ]
        },
        "column48": {
            "originColumnName": "LastUpdate",
            "dataFormat": "date",
            "valueDescription": "when is this record updated last time",
            "size": 17686,
            "emptyValueCount": 0,
            "valType": "discrete",
            "typeNum": 757,
            "samples": [
                "2000-05-11",
                "2006-09-07",
                "2014-07-25",
                "2000-06-15",
                "2010-08-17",
                "2003-12-05",
                "2016-03-17",
                "2000-06-19",
                "2016-01-09",
                "2014-08-13"
            ]
        }
    }
}
# A

### Overview of the Database

The database is called **"california\_schools"**, which contains detailed data about schools in California, particularly related to school districts, school names, SAT scores, and various attributes such as enrollment, free meal count, school types, and administrative details. The data is organized across three tables: **frpm.csv**, **satscores.csv**, and **schools.csv**.

---

### Table 1: **frpm.csv**

* **Description**: This table contains information about school districts, school names, enrollment numbers, and free/reduced-price meal data (FRPM). It also includes other details such as school type, district type, and meal provision status, among others.

  #### Columns:

  1. **CDSCode**: A unique identifier for the school district. It is an integer and represents the unique California Department of Education School Code.
  2. **Academic Year**: The academic year for the data entry, given in a discrete format as a text representation of the school year (e.g., "2014-2015").
  3. **County Code**: An identifier for the county, represented as an integer.
  4. **District Code**: A unique identifier for the school district, represented as an integer.
  5. **School Code**: A unique identifier for the school itself, represented as an integer.
  6. **County Name**: The name of the county, represented as text.
  7. **District Name**: The name of the school district, represented as text.
  8. **School Name**: The name of the school, represented as text.
  9. **District Type**: Describes the type of school district (e.g., "Elementary School District"), represented as text.
  10. **School Type**: Describes the type of school (e.g., "K-12 Schools (Public)"), represented as text.
  11. **Educational Option Type**: Describes the educational option type (e.g., "Alternative School of Choice"), represented as text.
  12. **NSLP Provision Status**: Describes the National School Lunch Program provision status (e.g., "Provision 3"), represented as text.
  13. **Charter School (Y/N)**: Indicates if the school is a charter school (1 = Yes, 0 = No), represented as an integer.
  14. **Charter School Number**: The number assigned to a charter school, represented as text.
  15. **Charter Funding Type**: Describes the funding type for charter schools (e.g., "Not in CS funding model"), represented as text.
  16. **IRC**: A column marked as "Not useful," represented as an integer.
  17. **Low Grade**: The lowest grade offered at the school, represented as text.
  18. **High Grade**: The highest grade offered at the school, represented as text.
  19. **Enrollment (K-12)**: The total number of K-12 students enrolled in the school, represented as a real number.
  20. **Free Meal Count (K-12)**: The number of students receiving free meals (K-12), represented as a real number.
  21. **Percent (%) Eligible Free (K-12)**: The percentage of K-12 students eligible for free meals, represented as a real number.
  22. **FRPM Count (K-12)**: The count of students eligible for free or reduced-price meals (K-12), represented as a real number.
  23. **Percent (%) Eligible FRPM (K-12)**: The percentage of K-12 students eligible for free or reduced-price meals, represented as a real number.
  24. **Enrollment (Ages 5-17)**: The total number of students aged 5-17 enrolled, represented as a real number.
  25. **Free Meal Count (Ages 5-17)**: The number of students aged 5-17 receiving free meals, represented as a real number.
  26. **Percent (%) Eligible Free (Ages 5-17)**: The percentage of students aged 5-17 eligible for free meals, represented as a real number.
  27. **FRPM Count (Ages 5-17)**: The number of students aged 5-17 eligible for free or reduced-price meals, represented as a real number.
  28. **Percent (%) Eligible FRPM (Ages 5-17)**: The percentage of students aged 5-17 eligible for free or reduced-price meals, represented as a real number.
  29. **2013-14 CALPADS Fall 1 Certification Status**: The status of the CALPADS certification, represented as an integer.

---

### Table 2: **satscores.csv**

* **Description**: This table contains SAT scores for various schools in California, including the district name, school name, and test scores in different subjects. It also includes a field for the number of students who took the test and their results.

  #### Columns:

  1. **cds**: California Department of Education School Code, represented as a text.
  2. **rtype**: A status field that indicates some level of record distinction (not highly useful), represented as text.
  3. **sname**: The name of the school, represented as text.
  4. **dname**: The district name, represented as text.
  5. **cname**: The county name, represented as text.
  6. **enroll12**: The number of students enrolled in grades 1-12, represented as an integer.
  7. **NumTstTakr**: The number of students who took the SAT, represented as an integer.
  8. **AvgScrRead**: The average SAT reading score, represented as an integer.
  9. **AvgScrMath**: The average SAT math score, represented as an integer.
  10. **AvgScrWrite**: The average SAT writing score, represented as an integer.
  11. **NumGE1500**: The number of students whose SAT scores were greater than or equal to 1500, represented as an integer.

---

### Table 3: **schools.csv**

* **Description**: This table contains school-specific data such as location information (address, city, zip code), administrator information (name, email), and general school details (such as school ownership codes and educational type). The table serves as a central directory of various schools in California.

  #### Columns:

  1. **CDSCode**: The California Department of Education School Code, represented as text.
  2. **NCESDist**: The National Center for Educational Statistics district ID, represented as text.
  3. **NCESSchool**: The National Center for Educational Statistics school ID, represented as text.
  4. **StatusType**: The status of the district, such as "Active" or "Closed," represented as text.
  5. **County**: The name of the county, represented as text.
  6. **District**: The district name, represented as text.
  7. **School**: The name of the school, represented as text.
  8. **Street**: The physical street address, represented as text.
  9. **StreetAbr**: The abbreviated street address, represented as text.
  10. **City**: The city of the school, represented as text.
  11. **Zip**: The zip code of the school, represented as text.
  12. **State**: The state of the school, represented as text (California).
  13. **MailStreet**: The mailing street address, represented as text.
  14. **MailStrAbr**: The abbreviated mailing street address, represented as text.
  15. **MailCity**: The city of the mailing address, represented as text.
  16. **MailZip**: The zip code of the mailing address, represented as text.
  17. **MailState**: The state of the mailing address (California), represented as text.
  18. **Phone**: The phone number of the school, represented as text.
  19. **Ext**: The phone number extension, represented as text.
  20. **Website**: The website address of the school, represented as text.
  21. **OpenDate**: The date when the school opened, represented as a date.
  22. **ClosedDate**: The date when the school closed, represented as a date.
  23. **Charter**: Indicates whether the school is a charter school (1 = Yes, 0 = No), represented as an integer.
  24. **CharterNum**: The charter number for charter schools, represented as text.
  25. **FundingType**: Indicates the funding type for charter schools (e.g., "Locally funded"), represented as text.
  26. **DOC**: The District Ownership Code, represented as text.
  27. **DOCType**: The District Ownership Code Type, represented as text.
  28. **SOC**: The School Ownership Code, represented as text.
  29. **SOCType**: The School Ownership Code Type, represented as text.
  30. **EdOpsCode**: The Education Option Code, represented as text.
  31. **EdOpsName**: The Education Option Name, represented as text.
  32. **EILCode**: The Educational Instruction Level Code, represented as text.
  33. **EILName**: The Educational Instruction Level Name, represented as text.
  34. **GSoffered**: The grade span offered, represented as text.
  35. **GSserved**: The grade span served, represented as text.
  36. **Virtual**: The type of virtual instruction offered by the school, represented as text.
  37. **Magnet**: Indicates whether the school is a magnet school, represented as an integer.
  38. **Latitude**: The latitude coordinate of the school, represented as a real number.
  39. **Longitude**: The longitude coordinate of the school, represented as a real number.
  40. **AdmFName1**: The first name of the administrator, represented as text.
  41. **AdmLName1**: The last name of the administrator, represented as text.
  42. **AdmEmail1**: The email address of the administrator, represented as text.
  43. **AdmFName2**: The second name of the administrator, represented as text.
  44. **AdmLName2**: The second last name of the administrator, represented as text.
  45. **AdmEmail2**: The second email address of the administrator, represented as text.
  46. **AdmFName3**: The third name of the administrator, represented as text.
  47. **AdmLName3**: The third last name of the administrator, represented as text.
  48. **AdmEmail3**: The third email address of the administrator, represented as text.
  49. **LastUpdate**: The last update timestamp for the record, represented as a date.

---

### Key Columns Needing Supplementary Information

* **CDSCode** (all tables): This code appears in multiple tables and represents a key identifier for California schools. More information about its exact structure or what it stands for would help provide better understanding.

* **School Type** (frpm.csv): While the column describes various school types, knowing the full context of what each "School Type" entails would be useful.

* **FundingType** (schools.csv): The specific funding model for charter schools, as it impacts how schools operate under different funding conditions.

* **Charter Funding Type** (frpm.csv): Provides details on the funding of charter schools, but further explanation would clarify how these different funding types function in relation to state policies.

* **School Ownership Code (SOC)** (schools.csv): Clarification on how different school ownership categories are defined would enhance understanding.

---

### Related Tables and Columns

* **frpm.csv** and **satscores.csv** share the **CDSCode**, which links the school-related information from both tables (enrollment data, SAT scores, etc.).

* **frpm.csv** and **schools.csv** share the **CDSCode**, linking detailed school and district information (such as school types and educational options).
