"""Calculate student grades by combining data from many sources.

Using Pandas, this script combines data from the:

* Roster
* Homework & Exam grades
* Quiz grades

to calculate final grades for a class.
"""
#Importing Libraries and Setting Paths
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

HERE = Path(__file__).parent
DATA_FOLDER = HERE / "data"

#Data Importation and Cleaning
roster = pd.read_csv(
    DATA_FOLDER / "roster.csv", 
    converters={'NetID': lambda x: x.lower(), 'Email Address': lambda x: x.lower()},
    index_col='NetID'
)

hw_exam_grades = pd.read_csv(
    DATA_FOLDER / "hw_exam_grades.csv",  
    converters={'SID': lambda x: x.lower()},
    index_col='SID'
)

quiz_grades = pd.DataFrame()
quiz_files = DATA_FOLDER.glob("quiz_*_grades.csv")
for file in quiz_files:
    quiz_data = pd.read_csv(file, index_col='Email')
    quiz_name = "Quiz " + file.stem.split('_')[1]
    quiz_grades[quiz_name] = quiz_data['Grade']

#Data Merging: roaster and homework
final_data = pd.merge(
    roster, hw_exam_grades, left_index=True, right_index=True
)

#Data Merging: Final data and quiz grades
final_data = pd.merge(
    final_data, quiz_grades, left_on='Email Address', right_index=True, how='left'
)

final_data = final_data.fillna(0)

#Data Processing and Score Calculation
n_exams = 3
#For each exam, calculate the score as a proportion of the maximum points possible.
#Remove pass once you have cerated written the for loop
for n in range(1, n_exams + 1):
    final_data[f"Exam {n} Score"] = final_data[f"Exam {n}"] / final_data[f"Exam {n} - Max Points"]

#Calculating Exam Scores:
#Filter homework and Homework - for max points
homework_scores = final_data.filter(regex="Homework \d+$", axis=1)
homework_max_points = final_data.filter(regex="Homework \d+ - Max Points$", axis=1)

#Calculating Total Homework score
sum_of_hw_scores = homework_scores.sum(axis=1)
sum_of_hw_max = homework_max_points.sum(axis=1)
final_data["Total Homework"] = (sum_of_hw_scores / sum_of_hw_max)

#Calculating Average Homework Scores
hw_max_renamed = homework_max_points.rename(lambda x: x.replace(" - Max Points", ""), axis=1)
average_hw_scores = (homework_scores / hw_max_renamed).mean(axis=1)
final_data["Average Homework"] = average_hw_scores

#Final Homework Score Calculation
final_data["Homework Score"] = final_data[["Total Homework", "Average Homework"]].max(axis=1)

#Calculating Total and Average Quiz Scores:
#Filter the data for Quiz scores
quiz_scores = final_data.filter(like="Quiz", axis=1)
quiz_max_points = pd.Series(
    {"Quiz 1": 11, "Quiz 2": 15, "Quiz 3": 17, "Quiz 4": 14, "Quiz 5": 12}
)

#Final Quiz Score Calculation:
sum_of_quiz_scores = quiz_scores.sum(axis=1)
sum_of_quiz_max = quiz_max_points.sum()
final_data["Total Quizzes"] = (sum_of_quiz_scores / sum_of_quiz_max)

average_quiz_scores = quiz_scores.div(quiz_max_points).sum(axis=1) / len(quiz_max_points)
final_data["Average Quizzes"] = average_quiz_scores

final_data["Quiz Score"] = final_data[["Total Quizzes", "Average Quizzes"]].max(axis=1)

#Calculating the Final Score:
weightings = pd.Series(
    {
        "Exam 1 Score": 0.05,
        "Exam 2 Score": 0.1,
        "Exam 3 Score": 0.15,
        "Quiz Score": 0.30,
        "Homework Score": 0.4,
    }
)

final_data["Final Score"] = final_data[weightings.index].mul(weightings).sum(axis=1)
# Rounding Up the Final Score
final_data["Ceiling Score"] = np.ceil(final_data["Final Score"].mul(100))

#Defining Grade Mapping:
grades = {
    90: "A",
    80: "B",
    70: "C",
    60: "D",
    0: "F",
}

#Applying Grade Mapping to Data:
def grade_mapping(value):
    for score, grade in grades.items():
        if value >= score:
            return grade

letter_grades = final_data["Ceiling Score"].apply(grade_mapping)
final_data["Final Grade"] = pd.Categorical(letter_grades, categories=grades.values(), ordered=True)

#Processing Data by Sections:
for section, table in final_data.groupby("Section"):
    section_file = DATA_FOLDER / f"section_{section}_data.csv"
    table.sort_values(by=['Last Name', 'First Name']).to_csv(section_file, index=False)
    print(f"\nSection {section}:\n{table}")
    print(f"Number of students in Section {section}: {table.shape[0]}")
    print(f"Data saved to: {section_file}")

#Visualizing Grade Distribution: Get Grade Counts and use plot to plot the grades
grade_counts = final_data["Final Grade"].value_counts()
sorted_grades = sorted(grade_counts.index)
print(grade_counts) 
plt.bar(sorted_grades, grade_counts[sorted_grades])
plt.xlabel("Letter Grade")
plt.ylabel("Count")
plt.title("Distribution of Letter Grades")
plt.show()

#Visualize the data on with Histogram and use Matplot lib density function to print Kernel Density Estimate
final_data["Final Score"].plot.hist(bins=20, label="Histogram")
final_data["Final Score"].plot.density(
    linewidth=4, label="Kernel Density Estimate"
)

#Plotting Normal Distribution:
final_mean = final_data["Final Score"].mean()
final_std = final_data["Final Score"].std()
print("Final Mean " + str(final_mean))
print("Final Std " + str(final_std))
x = np.linspace(final_mean - 5 * final_std, final_mean + 5 * final_std, 100)
y = scipy.stats.norm.pdf(x, final_mean, final_std)
plt.plot(x, y, label="Normal Distribution")

#Plot the normal distribution of final_mean and final_std
plt.legend()
plt.xlabel("Final Score")
plt.ylabel("Density")
plt.show()